import cv2
import numpy as np


from pyGUS.core import make_clean, fill_boundary, imshow, show_error, revert
from pyGUS.global_vars import debug


def test(gray):
    # get edge, fail
    equal = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(equal, (15, 15), 0)
    threshold3, blur_bin = cv2.threshold(blur, 0, 255,
                                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bin_edge = cv2.Canny(blur_bin, 0, 255)
    bin_edge_ = make_clean(bin_edge)
    fill = fill_boundary(blur_bin)

    s_cnt, _ = cv2.findContours(bin_edge_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    s_img = np.zeros(gray.shape[:2])
    for i in s_cnt:
        cv2.drawContours(s_img, [i], 0, (255, 255, 255), 2)
    imshow('equal', equal)
    imshow('blur', blur)
    imshow('blur bin', blur_bin)
    imshow('blur bin_edge', bin_edge_)
    imshow('blur cnt', s_img)
    imshow('fill', fill)
    cv2.waitKey()
    return


def get_edge2(image: np.array) -> np.array:
    b, g, r = cv2.split(image)
    combine = revert(g // 2 + r // 2)
    sobel_add = get_sobel(image)
    s2_clean = make_clean(sobel_add)
    # s_equal = cv2.equalizeHist(s_erode)
    imshow('sobel2', sobel_add)
    imshow('s2_clean', s2_clean)
    scharr_x = cv2.Scharr(combine, cv2.CV_8U, 1, 0)
    scharr_y = cv2.Scharr(combine, cv2.CV_8U, 1, 0)
    scharr = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)
    sc_equal = cv2.equalizeHist(scharr)
    s_cnt, _ = cv2.findContours(scharr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    s_img = np.zeros(image.shape[:2])
    for i in s_cnt:
        cv2.drawContours(s_img, [i], 0, (255, 255, 255), 2)
    imshow('scnt', s_img)
    if debug:
        imshow('sobel_add', sobel_add)
    laplacian = cv2.Laplacian(combine, cv2.CV_64F)
    imshow('laplacian', laplacian)
    return s2_clean


def old_get_yellow(b, g, r):
    # deprecated
    yellow_part = np.minimum(g, r)
    b2 = b.astype('int')
    b2 -= yellow_part
    b2[b2 < 0] = 255
    b2 = b2.astype('uint8')
    return b2


def get_real_blue2(original_image: np.array, neg_ref_value: float,
                   pos_ref_value: float) -> (np.array, int):
    """
    255-b is not real blue part
    Paused
    """
    if neg_ref_value > pos_ref_value or pos_ref_value <= 0:
        show_error('Bad negative and positive reference values.')
    factor = 1
    b, g, r = cv2.split(original_image)
    pos_ref_value = round(pos_ref_value)
    neg_ref_value = max(255, round(neg_ref_value))
    revert_b = revert(b)
    # amplify
    # revert_b = revert_b.astype('float')
    # log.info(f'Factor {factor}')
    # make sure express ratio <= 100%
    revert_b = revert_b.astype('float')
    factor = 255 // pos_ref_value
    revert_b = (revert_b - neg_ref_value) * factor
    revert_b[revert_b > 255] = 255
    revert_b = revert_b.astype('uint8')
    amplified_neg_ref = int(factor * neg_ref_value)
    return revert_b, amplified_neg_ref


