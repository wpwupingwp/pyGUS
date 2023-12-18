from multiprocessing import Process, Queue
from sys import argv
from time import sleep

from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm
import cv2
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import scipy.sparse.linalg.interface

from pyGUS.utils import draw_dots, draw_box, imshow, resize
from pyGUS.global_vars import log, debug

from numba import njit
class SolveForeAndBack:
    # https://github.com/MarcoForte/closed-form-matting
    def __init__(self, image, alpha):
        self.CONST_ALPHA_MARGIN = 0.02
        self.image = image
        self.alpha = alpha

    @staticmethod
    def __spdiagonal(diag):
        return scipy.sparse.spdiags(diag, (0,), len(diag), len(diag))

    def get_const_conditions(self, image, alpha):
        falpha = alpha.flatten()
        weights = (
                (falpha < self.CONST_ALPHA_MARGIN) * 100.0 +
                0.03 * (1.0 - falpha) * (falpha < 0.3) +
                0.01 * (falpha > 1.0 - self.CONST_ALPHA_MARGIN)
        )
        conditions = self.__spdiagonal(weights)

        mask = falpha < 1.0 - self.CONST_ALPHA_MARGIN
        right_hand = (weights * mask)[:, np.newaxis] * image.reshape(
            (alpha.size, -1))
        return conditions, right_hand

    def get_grad_operator(self, mask):
        horizontal_left = np.ravel_multi_index(
            np.nonzero(mask[:, :-1] | mask[:, 1:]), mask.shape)
        horizontal_right = horizontal_left + 1

        vertical_top = np.ravel_multi_index(
            np.nonzero(mask[:-1, :] | mask[1:, :]), mask.shape)
        vertical_bottom = vertical_top + mask.shape[1]

        diag_main_1 = np.ravel_multi_index(
            np.nonzero(mask[:-1, :-1] | mask[1:, 1:]), mask.shape)
        diag_main_2 = diag_main_1 + mask.shape[1] + 1

        diag_sub_1 = np.ravel_multi_index(
            np.nonzero(mask[:-1, 1:] | mask[1:, :-1]), mask.shape) + 1
        diag_sub_2 = diag_sub_1 + mask.shape[1] - 1

        indices = np.stack((
            np.concatenate(
                (horizontal_left, vertical_top, diag_main_1, diag_sub_1)),
            np.concatenate(
                (horizontal_right, vertical_bottom, diag_main_2, diag_sub_2))
        ), axis=-1)
        return scipy.sparse.coo_matrix(
            (np.tile([-1, 1], len(indices)),
             (np.arange(indices.size) // 2, indices.flatten())),
            shape=(len(indices), mask.size))

    def run(self):
        consts = (self.alpha < self.CONST_ALPHA_MARGIN) | (
                self.alpha > 1.0 - self.CONST_ALPHA_MARGIN)
        grad = self.get_grad_operator(~consts)
        grad_weights = np.power(np.abs(grad * self.alpha.flatten()), 0.5)

        grad_only_positive = grad.maximum(0)
        grad_weights_f = grad_weights + 0.003 * grad_only_positive * (
                1.0 - self.alpha.flatten())
        grad_weights_b = (grad_weights + 0.003 * grad_only_positive *
                          self.alpha.flatten())

        grad_pad = scipy.sparse.coo_matrix(grad.shape)

        smoothness_conditions = scipy.sparse.vstack((
            scipy.sparse.hstack(
                (self.__spdiagonal(grad_weights_f) * grad, grad_pad)),
            scipy.sparse.hstack(
                (grad_pad, self.__spdiagonal(grad_weights_b) * grad))
        ))

        composite_conditions = scipy.sparse.hstack((
            self.__spdiagonal(self.alpha.flatten()),
            self.__spdiagonal(1.0 - self.alpha.flatten())
        ))

        const_conditions_f, b_const_f = self.get_const_conditions(
            self.image, 1.0 - self.alpha)
        const_conditions_b, b_const_b = self.get_const_conditions(
            self.image, self.alpha)

        non_zero_conditions = scipy.sparse.vstack((
            composite_conditions,
            scipy.sparse.hstack((
                const_conditions_f,
                scipy.sparse.coo_matrix(const_conditions_f.shape)
            )),
            scipy.sparse.hstack((
                scipy.sparse.coo_matrix(const_conditions_b.shape),
                const_conditions_b
            ))
        ))

        b_composite = self.image.reshape(self.alpha.size, -1)

        right_hand = non_zero_conditions.transpose() * np.concatenate(
            (b_composite,
             b_const_f,
             b_const_b))

        conditons = scipy.sparse.vstack((
            non_zero_conditions,
            smoothness_conditions
        ))
        left_hand = conditons.transpose() * conditons

        solution = scipy.sparse.linalg.spsolve(left_hand, right_hand).reshape(
            2, *self.image.shape)
        foreground = solution[0, :, :, :].reshape(*self.image.shape)
        background = solution[1, :, :, :].reshape(*self.image.shape)
        return foreground, background


@njit
def _rolling_block(A, block=(3, 3)):
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


@njit
def compute_laplacian(img, mask=None, eps=10 ** (-7), win_rad=1):
    # https://github.com/MarcoForte/closed-form-matting
    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1

    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    if mask is not None:
        mask = cv2.dilate(
            mask.astype(np.uint8),
            np.ones((win_diam, win_diam), np.uint8)
        ).astype(bool)
        win_mask = np.sum(mask.ravel()[win_inds], axis=2)
        win_inds = win_inds[win_mask > 0, :]
    else:
        win_inds = win_inds.reshape(-1, win_size)

    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=1, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI,
                        winI) / win_size - np.einsum('...ji,...jk ->...ik',
                                                     win_mu, win_mu)

    inv = np.linalg.inv(win_var + (eps / win_size) * np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1.0 / win_size) * (
            1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)),
                                shape=(h * w, h * w))
    return L


def cfm_with_prior(image, prior, prior_confidence, consts_map=None):
    # https://github.com/MarcoForte/closed-form-matting
    assert image.shape[:2] == prior.shape, (
        'prior must be 2D matrix with height and width equal '
        'to image.')
    assert image.shape[:2] == prior_confidence.shape, (
        'prior_confidence must be 2D matrix with '
        'height and width equal to image.')
    assert (consts_map is None) or image.shape[:2] == consts_map.shape, (
        'consts_map must be 2D matrix with height and width equal to image.')

    log.debug('Computing Matting Laplacian.')
    laplacian = compute_laplacian(
        image, ~consts_map if consts_map is not None else None)

    confidence = scipy.sparse.diags(prior_confidence.ravel())
    log.debug('Solving for alpha.')
    solution = scipy.sparse.linalg.spsolve(
        laplacian + confidence,
        prior.ravel() * prior_confidence.ravel())
    alpha = np.minimum(np.maximum(solution.reshape(prior.shape), 0), 1)
    return alpha


def cfm_with_scribbles(image, scribbles, scribbles_confidence=100.0):
    assert image.shape == scribbles.shape, 'scribbles must have exactly same ' \
                                           'shape as image.'
    prior = np.sign(np.sum(scribbles - image, axis=2)) / 2 + 0.5
    consts_map = prior != 0.5
    return cfm_with_prior(image, prior, scribbles_confidence * consts_map,
                          consts_map)


def closed_form_matting(image_raw: np.ndarray, scribbles_raw: np.ndarray, results):
    # https://github.com/MarcoForte/closed-form-matting
    # bgr
    img = image_raw / 255.0
    scribbles = scribbles_raw / 255.0
    alpha = cfm_with_scribbles(img, scribbles)
    results.put(alpha)
    return alpha


def get_scribbles(img: np.array):
    drawed = img.copy()
    drawed = draw_dots(drawed)
    drawed = draw_box(drawed)
    if debug:
        imshow('scribbles', drawed)
    return drawed


def run_cfm(img: np.array, quick):
    log.info('Start CFM...')
    # at least 1366x768 screen
    scribbles_raw = get_scribbles(img)
    log.info('Calculating...')
    # reduce time
    if quick:
        size = (512, 512)
        img = resize(img, *size)
        scribbles_raw = resize(scribbles_raw, *size)
    results = Queue()
    a = Process(target=closed_form_matting, args=(img, scribbles_raw, results))
    a.start()
    total = img.size
    with tqdm(total=total, desc='Calculating', unit='pixels',
              dynamic_ncols=True) as t:
        for i in range(total):
            if not results.empty():
                t.update(total-t.n)
                t.close()
            else:
                sleep(0.1)
                t.update(n=total//1000)
    alpha = results.get()
    # alpha = closed_form_matting(img, scribbles_raw)
    # alpha = closed_form_matting(img, scribbles_raw)
    alpha_256 = alpha * 255
    alpha_256 = alpha_256.astype('uint8')
    log.info('CFM done.')
    return alpha_256, img


def get_cfm_masks(image: np.array, quick=False) -> np.array:
    height, width = image.shape[:2]

    def try_to_get():
        gray, resized = run_cfm(image)
        gray = cv2.resize(gray, (width, height),
                          interpolation=cv2.INTER_LANCZOS4)
        th, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary
    binary = try_to_get()
    if debug:
        # imshow('bin', binary)
        # imshow('bin_edge', bin_edge2)
        masked = cv2.bitwise_and(image, image, mask=binary)
        imshow('masked', masked)
        # Esc
        while cv2.waitKey() == 27:
            binary = try_to_get()
            masked = cv2.bitwise_and(image, image, mask=binary)
            imshow('masked', masked)
    return binary


if __name__ == '__main__':
    img = cv2.imread(argv[1], cv2.IMREAD_COLOR)
    run_cfm(img)
