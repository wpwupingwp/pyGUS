#!/usr/bin/python3.10
import argparse
import csv
from pathlib import Path

# for nuitka
from matplotlib.backends import backend_pdf
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


from pyGUS.global_vars import log, debug
from pyGUS.utils import select_box, select_polygon, draw_lines, resize
from pyGUS.utils import color_calibrate, if_exist, imshow, show_error, hex2bgr
from pyGUS.utils import get_CCT
from pyGUS.cfm import get_cfm_masks

matplotlib.use('Agg')
plt.set_loglevel('error')
MANUAL = 'manual'
NEG_TEXT = 'Click mouse to select negative expression region as reference'
POS_TEXT = 'Click mouse to select positive expression region as reference'
GENERAL_TEXT = ('Failed to detect target with extremely low contrast. '
                'Please manually select target region.')
SHORT_TEXT = 'Click mouse to select target region'


# todo: update manual
# todo mode 1 test: single object for each image, manually select positive,
#  negative, targets
# todo mode 2 test: two object for each image, left target, right positive
# todo mode 3 test: two object for each image, left target, right color card
# todo: mode 4 test: select area by mouse
# todo: manuscript


def parse_arg(arg_str):
    """
    Args:
        arg_str: None or str
    Returns:
        arg.parse_args()
    """
    arg = argparse.ArgumentParser(prog='pyGUS')
    arg.add_argument('-ref1', help='Negative expression reference image')
    arg.add_argument('-ref2', help='Positive expression reference image')
    arg.add_argument('-auto_ref', action='store_true',
                     help='auto detect objects (deprecated)')
    arg.add_argument('-images', nargs='*', help='Input images', required=True)
    arg.add_argument('-quick', action='store_true',
                     help='quick but rough')
    arg.add_argument('-mode', type=int, choices=(1, 2, 3, 4), required=True,
                     help=('1. Normal: single target in each image; '
                           '2. With reference: target and positive reference; '
                           '3. With color checker: target and color checker; '
                           '4. Manually: manually select targets with mouse'))
    if arg_str is None:
        return arg.parse_args()
    else:
        return arg.parse_args(arg_str.split(' '))


def get_input(arg) -> (str, str, list[str], bool, bool, str):
    message = None
    negative = None
    positive = None
    targets = None
    if arg.mode in (1, 3):
        if arg.ref1 is None or arg.ref2 is None:
            message = (f'Bad input. Mode {arg.mode} requires ref1 (negative) '
                       f'and ref2 (positive)')
    elif arg.mode == 2:
        if arg.ref1 is None:
            message = 'Bad input. Mode 2 requires ref1 (negative vs positive)'
    if message is not None:
        return negative, positive, targets, message
    targets_ = [Path(i).absolute() for i in arg.images]
    targets = [if_exist(i) for i in targets_]
    if arg.mode != 4:
        negative = Path(arg.ref1).absolute()
        negative = if_exist(negative)
    if arg.mode not in (2, 4):
        positive = Path(arg.ref2).absolute()
        positive = if_exist(positive)
    auto_ref = bool(arg.auto_ref)
    quick = bool(arg.quick)
    return negative, positive, targets, auto_ref, quick, message


def mode_1(negative: str, positive: str, targets: list, auto_ref: bool,
           quick: bool) -> (list, list, list):
    """
    Ignore light change
    Args:
        negative: negative reference image
        positive: positive reference
        targets: target images
        auto_ref: automatic detect reference region or no
    Returns:
        neg_result
        pos_result
        target_results
    deprecated parameters:
        convex: use convex hull for region detection or not
    """
    target_results = []
    if auto_ref:
        # old methods
        (neg_result, neg_mask, neg_filtered_result, neg_level_cnt,
         neg_img) = get_contour_wrapper(negative, 0, 255, 1, NEG_TEXT)
        neg_ref_value = neg_result[0]
        pos_level_cnt, pos_img = get_contour(positive)
        pos_filtered_result = filter_contours(pos_img, pos_level_cnt, big=1)
        pos_mask = get_single_mask(pos_filtered_result, pos_level_cnt, pos_img)
        pos_result, _ = calculate(pos_img, pos_mask,
                                  neg_ref_value=neg_ref_value)
        pos_ref_value = pos_result[0]
        for target in targets:
            (target_result, target_mask, filtered_result, level_cnt,
             img) = get_contour_wrapper(target, neg_ref_value, pos_ref_value, 1)
            target_results.append(target_result)
            img_dict = draw_images(filtered_result, level_cnt, img, simple=True,
                                   show=False, filename=target)
            if debug:
                pass
                # imshow('mask', target_mask)
                # imshow('masked', cv2.bitwise_and(img, img, mask=target_mask))
        log.debug(f'neg {neg_ref_value} pos {pos_ref_value}')
        draw_images(pos_filtered_result, pos_level_cnt, pos_img, show=False,
                    simple=True, filename=positive)
    else:
        neg_img = cv2.imread(negative)
        pos_img = cv2.imread(positive)
        neg_mask = get_cfm_masks(neg_img, quick)
        pos_mask = get_cfm_masks(pos_img, quick)
        neg_result, neg_no_yellow_mask = calculate(neg_img, neg_mask)
        neg_ref_value = neg_result[0]
        pos_result, pos_no_yellow_mask = calculate(pos_img, pos_mask,
                                                   neg_ref_value=neg_ref_value)
        pos_ref_value = pos_result[0]
        for target in targets:
            target_img = cv2.imread(target)
            target_mask = get_cfm_masks(target_img, quick)
            target_result, no_yellow_mask = calculate(
                target_img, target_mask, neg_ref_value, pos_ref_value)
            target_results.append(target_result)
            target_png = get_out_filename(target, '-masked')
            write_masks(target_img, target_mask, no_yellow_mask, target_png)
        neg_png = get_out_filename(negative, '-masked')
        pos_png = get_out_filename(positive, '-masked')
        write_masks(neg_img, neg_mask, neg_no_yellow_mask, neg_png)
        write_masks(pos_img, pos_mask, pos_no_yellow_mask, pos_png)
    return neg_result, pos_result, target_results


def mode_2(ref1: str, _: str, targets: list, auto_ref: bool, quick: bool) -> (
        list, list, list):
    """
    use negative to calibrate positive, and then measure each target
    assume positive in each image is same, ignore light change
    ignore small_external, inner_contours
    """
    negative_positive_ref = ref1
    target_results = []
    if auto_ref:
        ref_level_cnt, ref_img = get_contour(negative_positive_ref)
        ref_filtered_result = filter_contours(ref_img, ref_level_cnt, big=2)
        if ref_filtered_result is not None:
            neg_mask, pos_mask = get_left_right_mask(ref_filtered_result,
                                                     ref_level_cnt, ref_img)
            neg_result, _ = calculate(ref_img, neg_mask)
            neg_ref_value = neg_result[0]
            pos_result, _ = calculate(ref_img, pos_mask,
                                      neg_ref_value=neg_ref_value)
        else:
            neg_result, neg_mask = manual_ref(ref_img, NEG_TEXT)
            pos_result, pos_mask = manual_ref(ref_img, POS_TEXT)
        neg_ref_value = neg_result[0]
        pos_ref_value = pos_result[0]
        log.debug(f'neg {neg_ref_value} pos {pos_ref_value}')
        for target in targets:
            level_cnt, img = get_contour(target)
            filtered_result = filter_contours(img, level_cnt)
            if filtered_result is not None:
                target_mask, pos_mask_ = get_left_right_mask(filtered_result,
                                                             level_cnt,
                                                             img)
                target_result, _ = calculate(img, target_mask, neg_ref_value,
                                             pos_ref_value)
            else:
                log.info(GENERAL_TEXT)
                target_result, target_mask = manual_ref(
                    img, text=SHORT_TEXT, method='polygon')
            target_results.append(target_result)
            img_dict = draw_images(filtered_result, level_cnt, img, show=False,
                                   simple=True, filename=target)
    else:
        ref_img = cv2.imread(negative_positive_ref)
        ref_mask = get_cfm_masks(ref_img, quick)
        neg_mask, pos_mask = split_left_right_mask(ref_mask)
        neg_result, neg_no_yellow_mask = calculate(ref_img, neg_mask)
        neg_ref_value = neg_result[0]
        pos_result, pos_no_yellow_mask = calculate(ref_img, pos_mask,
                                                   neg_ref_value=neg_ref_value)
        pos_ref_value = pos_result[0]
        for target in targets:
            target_img = cv2.imread(target)
            mask = get_cfm_masks(target_img, quick)
            target_mask, pos_mask_ = split_left_right_mask(mask)
            target_result, no_yellow_mask = calculate(
                target_img, target_mask, neg_ref_value, pos_ref_value)
            target_results.append(target_result)
            target_png = get_out_filename(target, '-masked')
            write_masks(target_img, target_mask, no_yellow_mask, target_png)
        neg_png = get_out_filename(negative_positive_ref, '-negative-masked')
        pos_png = get_out_filename(negative_positive_ref, '-positive-masked')
        write_masks(ref_img, neg_mask, neg_no_yellow_mask, neg_png)
        write_masks(ref_img, pos_mask, pos_no_yellow_mask, pos_png)
    return neg_result, pos_result, target_results


def mode_3(ref1: str, ref2: str, targets: list, auto_ref: bool, quick: bool
           ) -> (list, list, list):
    # use color card to calibrate each image
    # first left: negative, first right: card
    # second left: positive, second right: card
    # third and next left: target, right: card
    negative = ref1
    positive = ref2
    ok_neg = color_calibrate(negative)
    ok_pos = color_calibrate(positive)
    ok_targets = [color_calibrate(i) for i in targets]
    target_results = []
    if auto_ref:
        neg_level_cnt, neg_img = get_contour(ok_neg)
        neg_filtered_result = filter_contours(neg_img, neg_level_cnt, big=2)
        if neg_filtered_result is not None:
            neg_left_mask, neg_right_mask = get_left_right_mask(
                neg_filtered_result, neg_level_cnt, neg_img)
            neg_result, _ = calculate(neg_img, neg_left_mask)
        else:
            neg_result, neg_mask = manual_ref(neg_img, NEG_TEXT)
        pos_level_cnt, pos_img = get_contour(ok_pos)
        pos_filtered_result = filter_contours(pos_img, pos_level_cnt, big=2)
        pos_left_mask, pos_right_mask = get_left_right_mask(
            pos_filtered_result, pos_level_cnt, pos_img)
        pos_result, _ = calculate(pos_img, pos_left_mask)
        neg_ref_value = neg_result[0]
        pos_ref_value = pos_result[0]
        log.debug(f'neg {neg_ref_value} pos {pos_ref_value}')
        for target in ok_targets:
            level_cnt, img = get_contour(target)
            filtered_result = filter_contours(img, level_cnt, big=2)
            if filtered_result is not None:
                left_mask, right_mask = get_left_right_mask(filtered_result,
                                                            level_cnt,
                                                            img)
                target_result, _ = calculate(img, left_mask, neg_ref_value,
                                             pos_ref_value)
            else:
                log.info(GENERAL_TEXT)
                target_result, target_mask = manual_ref(img, text=SHORT_TEXT,
                                                        method='polygon')
            target_results.append(target_result)
            img_dict = draw_images(filtered_result, level_cnt, img, show=False,
                                   simple=True, filename=target)
    else:
        # draw dots in color checker too when use cfm method
        neg_img = cv2.imread(ok_neg)
        pos_img = cv2.imread(ok_pos)
        neg_mask_raw = get_cfm_masks(neg_img, quick)
        pos_mask_raw = get_cfm_masks(pos_img, quick)
        neg_mask, _ = split_left_right_mask(neg_mask_raw)
        pos_mask, _ = split_left_right_mask(pos_mask_raw)
        neg_result, neg_no_yellow_mask = calculate(neg_img, neg_mask)
        neg_ref_value = neg_result[0]
        pos_result, pos_no_yellow_mask = calculate(pos_img, pos_mask,
                                                   neg_ref_value=neg_ref_value)
        pos_ref_value = pos_result[0]
        for target in ok_targets:
            target_img = cv2.imread(target)
            target_mask_raw = get_cfm_masks(target_img, quick)
            target_mask, _ = split_left_right_mask(target_mask_raw)
            target_result, no_yellow_mask = calculate(
                target_img, target_mask, neg_ref_value, pos_ref_value)
            target_results.append(target_result)
            target_png = get_out_filename(target, '-masked')
            write_masks(target_img, target_mask, no_yellow_mask, target_png)
        neg_png = get_out_filename(ok_neg, '-masked')
        pos_png = get_out_filename(ok_pos, '-masked')
        write_masks(neg_img, neg_mask, neg_no_yellow_mask, neg_png)
        write_masks(pos_img, pos_mask, pos_no_yellow_mask, pos_png)
    return neg_result, pos_result, target_results


def mode_4(_: str, __: str, targets: list, ___: bool, ____: bool) -> (
        list, list, list):
    """
    Select region manually
    """
    name_dict = {'neg': ('Negative reference', hex2bgr('#FF6B6B')),
                 'pos': ('Positive reference', hex2bgr('#FFd93D')),
                 'target': ('Target region', hex2bgr('#3BEBEB'))}
    neg_result = []
    pos_result = []
    target_results = []
    size = (1500, 1500)
    for target in targets:
        log.info(f'Analyzing {target}')
        img = cv2.imread(target)
        height, width = img.shape[:2]
        if img is None:
            show_error(f'Bad image file {target}')
        img_copy = img.copy()
        img_copy_small = resize(img_copy, *size)
        mask1, mask2, mask3 = None, None, None
        mask1 = select_polygon(img_copy_small, name_dict['neg'][0],
                               name_dict['neg'][1])
        if mask1 is not None:
            mask2 = select_polygon(img_copy_small, name_dict['pos'][0],
                                   name_dict['pos'][1])
            if mask2 is not None:
                mask3 = select_polygon(img_copy_small, name_dict['target'][0],
                                       name_dict['target'][1])
        for i in mask1, mask2, mask3:
            if i is None:
                return None, None, None
        mask1 = cv2.resize(mask1, (width, height),
                           interpolation=cv2.INTER_LANCZOS4)
        mask2 = cv2.resize(mask2, (width, height),
                           interpolation=cv2.INTER_LANCZOS4)
        mask3 = cv2.resize(mask3, (width, height),
                           interpolation=cv2.INTER_LANCZOS4)
        try:
            imshow('Press any key to continue', img_copy_small)
            cv2.waitKey()
            cv2.destroyAllWindows()
        except cv2.error:
            error_msg = 'Bad selection.'
            show_error(error_msg)
            return neg_result, pos_result, target_results
        neg_result, _ = calculate(img, mask1, neg_ref_value=0)
        neg_ref_value, neg_std, *_ = neg_result
        pos_result, _ = calculate(img, mask2, pos_ref_value=255)
        pos_ref_value, pos_std, *_ = pos_result
        neg_ref_value += neg_std
        pos_ref_value += pos_std
        log.debug(f'neg {neg_ref_value} pos {pos_ref_value}')
        result, _ = calculate(img, mask3, neg_ref_value, pos_ref_value)
        target_results.append(result)
        out_p = Path(target)
        out_filename = str(out_p.parent / out_p.with_name(
            f'{out_p.stem}_select.png'))
        img_copy2 = cv2.resize(img_copy_small, (width, height),
                               interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(out_filename, img_copy2)
        log.debug(f'Write image {out_filename}')
    return neg_result, pos_result, target_results


def fill_mask(shape: list, target: list, fake_inner: list,
              inner_background: list, level_cnt: dict) -> np.array:
    cnt = level_cnt[target]
    white = (255, 255, 255)
    black = (0, 0, 0)
    mask = np.zeros(shape, dtype='uint8')
    # related inner background
    cv2.fillPoly(mask, [cnt], white)
    for i in fake_inner:
        cnt_i = level_cnt[i]
        cv2.fillPoly(mask, [cnt_i], white)
    for j in inner_background:
        cnt_j = level_cnt[j]
        cv2.fillPoly(mask, [cnt_j], black)
    return mask


def get_single_mask(filtered_result: list, level_cnt: dict,
                    img: np.array) -> np.array:
    # big_external + fake_inner - inner_background
    (big_external_contours, small_external_contours, inner_contours,
     fake_inner, inner_background) = filtered_result
    target = big_external_contours[0]
    self_index = target[4]
    mask = fill_mask(img.shape[:2], target, fake_inner, inner_background,
                     level_cnt)
    return mask


def get_left_right(big_external_contours: list,
                   level_cnt: dict) -> (np.array, np.array):
    """
    Left is target, right is ref
    Split images to left and right according to bounding rectangle of
    external contours.
    Return None for errors.
    """
    left = None
    right = None
    if len(big_external_contours) == 0:
        show_error('Cannot find targets in the image.')
    elif len(big_external_contours) == 1:
        left = big_external_contours[0]
        log.info('Only detected one target in the image.')
    else:
        left, right = big_external_contours
        x1, y1, w1, h1 = cv2.boundingRect(level_cnt[left])
        x2, y2, w2, h2 = cv2.boundingRect(level_cnt[right])
        # opencv axis: 0->x, 0|vy
        if x1 > x2:
            left, right = right, left
            log.debug('Exchange left and right.')
            x1, y1, w1, h1, x2, y2, w2, h2 = x2, y2, w2, h2, x1, y1, w1, h1
        if x1 + w1 > x2:
            if y1 > y2:
                y1, h1, y2, h2 = y2, h2, y1, h1
            if y1 + h1 > y2:
                show_error('Target and reference are overlapped!')
                # left = right = None
    return left, right


def get_left_right_mask(filtered_result: list, level_cnt: dict,
                        img: np.array) -> (np.array, np.array):
    # by contour
    (big_external_contours, small_external_contours, inner_contours,
     fake_inner, inner_background) = filtered_result
    # target, ref
    left, right = get_left_right(big_external_contours, level_cnt)
    if left is None and right is None:
        show_error('Object not found.')
    left_right_mask = list()
    for target in left, right:
        try:
            self_index = target[4]
        except TypeError:
            show_error('Detect edge failed.')
            return
        log.debug(f'target {target}')
        log.debug(f'self {self_index}')
        related_fake_inner = [i for i in fake_inner if i[3] == self_index]
        # related inner background
        related_inner_bg = [i for i in inner_background if i[3] == target[4]]
        mask = fill_mask(img.shape[:2], target, related_fake_inner,
                         related_inner_bg, level_cnt)
        left_right_mask.append(mask)
    left_mask, right_mask = left_right_mask
    return left_mask, right_mask


def split_left_right_mask(mask: np.array) -> (np.array, np.array):
    # split mask with two big objects
    left = mask.copy()
    right = mask.copy()
    c, big = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(c, key=lambda x: cv2.contourArea(x), reverse=True)
    big_c = contours[:2]
    if len(big_c) != 2:
        show_error('Cannot find two targets in given images.')
    box = [cv2.boundingRect(i) for i in big_c]
    cnt_box = list(zip(big_c, box))
    # by box's x
    cnt_box.sort(key=lambda x: x[1][0])
    start = cnt_box[0][1][0] + cnt_box[0][1][2]
    end = cnt_box[1][1][0]
    middle = (start + end) // 2
    # if debug:
    # for i in box:
    #     cv2.rectangle(mask, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (255, 255, 255), 5)
    # cv2.line(mask, (middle, 0), (middle, mask.shape[0]), (255, 255, 0), 4)
    # cv2.imshow('mask', mask)
    # cv2.waitKey()
    left[:, middle:] = 0
    right[:, :middle] = 0
    return left, right


def auto_Canny(image: np.array, sigma=0.33) -> np.array:
    """
    Compute the median of the single channel pixel intensities
    """
    v = np.median(image)
    # use edited lower bound
    lower = int(max(0, (1.0 - sigma * 2) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edge = cv2.Canny(image, lower, upper)
    return edge


def make_clean(image: np.array) -> np.array:
    # blur->dilate->erode
    # ensure CV_8U
    image = cv2.convertScaleAbs(image, alpha=1, beta=0)
    # gaussian is better than median
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    dilate = cv2.dilate(blur, None)
    erode_edge = cv2.erode(dilate, None)
    return erode_edge


def get_scribbles(img: np.array):
    img_new = img.copy()
    draw_lines(img_new, 'draw lines on plants', 'fore')
    draw_lines(img_new, 'draw lines on background', 'back')
    return img_new


def get_edge(image: np.array) -> np.array:
    """
    Args:
        image: raw BGR image
    Returns:
        erode_edge: edge
    """
    # edge->blur->dilate->erode->contours
    # b, g, r = cv2.split(image)
    # combine = revert(g // 3 + r // 3 + b // 3)
    # grab(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    revert_gray = revert(gray)
    edge = auto_Canny(revert_gray)
    # blur edge, not original image
    erode_edge = make_clean(edge)
    if debug:
        imshow('revert_gray', revert_gray)
        imshow('gray', gray)
        imshow('erode_edge', erode_edge)
        imshow('edge', edge)
        cv2.waitKey()
    return erode_edge


def revert(img: np.array) -> np.array:
    return 255 - img


def get_arc_epsilon(max_contour: np.array, ratio=0.0001) -> float:
    log.debug(f'Max contour area: {cv2.contourArea(max_contour)}')
    arc_epsilon = cv2.arcLength(max_contour, True) * ratio
    log.debug(f'Set arc epsilon: {arc_epsilon}')
    return arc_epsilon


def get_contour_value(img: np.array, cnt: np.array,
                      with_std=True) -> (float, float):
    # fill contour with (255,255,255)
    mask = np.zeros(img.shape[:2], dtype='uint8')
    cv2.fillPoly(mask, [cnt], (255, 255, 255))
    if with_std:
        mean, std = cv2.meanStdDev(img, mask=mask)
        return mean[0][0], std[0][0]
    else:
        mean = cv2.mean(img, mask=mask)
        return mean[0], 0.0


def get_background_value(img: np.array, external_contours: list,
                         level_cnt: dict) -> (float, float):
    # assume background value is greater than negative reference value
    mask = np.ones(img.shape[:2], dtype='uint8')
    for external in external_contours:
        cnt = level_cnt[external]
        cv2.fillPoly(mask, [cnt], (0, 0, 0))
    mean, std = cv2.meanStdDev(img, mask=mask)
    # imshow('Background masked', masked)
    return mean[0][0], std[0][0]


def remove_fake_inner_cnt(img: np.array, level_cnt: dict,
                          big_external_contours: list,
                          external_contours: list,
                          inner_contours: list) -> (list, list):
    fake_inner = list()
    inner_background = list()
    b, g, r = cv2.split(img)
    revert_b = revert(b)
    # background blue mean
    bg_blue_mean, bg_blue_std = get_background_value(revert_b,
                                                     external_contours,
                                                     level_cnt)
    bg_size = img.size
    log.debug(f'Whole image: Area {img.size}\t '
              f'Whole blue mean {cv2.meanStdDev(revert_b)}')
    log.debug(
        f'Background masked: Area {bg_size}\t Blue mean {bg_blue_mean}'
        f'+-std{bg_blue_std}')
    for big in big_external_contours:
        # [next, previous, child, parent, self]
        big_cnt = level_cnt[big]
        big_area = cv2.contourArea(big_cnt)
        self_index = big[4]
        related_inner = [i for i in inner_contours if i[3] == self_index]
        big_blue_mean, big_blue_std = get_contour_value(revert_b, big_cnt)
        # log.debug(f'Big region: No.{big[-1]}\t '
        # f'Area: {big_area}\t Blue mean: {big_blue_mean}')
        for inner in related_inner:
            inner_cnt = level_cnt[inner]
            # inner_cnt_area = cv2.contourArea(inner_cnt)
            inner_blue_mean, _ = get_contour_value(revert_b, inner_cnt,
                                                   with_std=False)
            # inner_green_mean, _ = get_contour_value(revert_g, inner_cnt)
            # assert big_blue_mean > bg_blue_mean
            if inner_blue_mean < bg_blue_mean + bg_blue_std:
                pass
                # log.debug(f'Real background region: No.{inner[-1]}\t '
                #           f'Area: {inner_cnt_area}\t'
                #           f'Blue mean: {inner_blue_mean}')
                inner_background.append(inner)
            elif inner_blue_mean >= big_blue_mean:
                fake_inner.append(inner)
    return fake_inner, inner_background


def filter_contours(img: np.array, level_cnt: dict, big=2) -> (list, list, list,
                                                               list, list):
    """
    Args:
        img: original image
        level_cnt(dict):
        big: number of big external contours
    Returns:
        big:
        small:
        inner:
        fake_inner:
        inner_background:
    """
    external_contours = []
    inner_contours = []
    for level, cnt in level_cnt.items():
        next_, previous_, first_child, parent, self_ = level
        # -1 means no parent -> external
        if parent == -1:
            external_contours.append(level)
        else:
            inner_contours.append(level)
    external_area_dict = {}
    for i in external_contours:
        external_area_dict[i] = cv2.contourArea(level_cnt[i])
    external_contours.sort(key=lambda x: external_area_dict[x], reverse=True)
    # a picture only contains at most TWO target (sample and reference)
    big_external_contours = external_contours[:big]

    # list(dict.values()) does not work
    area_list_new = [external_area_dict[i] for i in external_contours]
    z_scores = get_zscore(area_list_new)
    external_zscore_dict = dict(zip(external_contours, z_scores))
    for i in big_external_contours:
        # less than 1 means not big enough
        if external_zscore_dict[i] < 1:
            log.info('Found abnormal contours.')
            return None
    try:
        small_external_contours = external_contours[2:]
    except IndexError:
        small_external_contours = list()
    fake_inner, inner_background = remove_fake_inner_cnt(
        img, level_cnt, big_external_contours, external_contours,
        inner_contours)
    return (big_external_contours, small_external_contours, inner_contours,
            fake_inner, inner_background)


def show_channel(img: np.array) -> None:
    # show rgb channel
    # opencv use BGR
    b, g, r = cv2.split(img)
    for title, value in zip(['b', 'g', 'r'], [b, g, r]):
        imshow(title + 'revert', 255 - value)
        imshow(title, value)


def get_yellow_mask(b: np.array, g: np.array, r: np.array) -> np.array:
    mask = np.zeros(b.shape[:2], dtype='uint8')
    mask[np.bitwise_and(r > b, g > b)] = 255
    return mask


def adjust_by_pos_ref(revert_b: np.array, pos_ref_value) -> np.array:
    new = revert_b.copy()
    if pos_ref_value == 255:
        return revert_b
    factor = pos_ref_value / 256
    new = cv2.convertScaleAbs(new, alpha=factor)
    return new


def get_real_blue(original_image: np.array, neg_ref_value: float,
                  pos_ref_value: float) -> (np.array, np.array):
    if neg_ref_value > pos_ref_value or pos_ref_value <= 0.0:
        show_error('Bad negative and positive reference values.')
    b, g, r = cv2.split(original_image)
    yellow_mask = get_yellow_mask(b, g, r)
    revert_b = revert(b)
    # revert_b_adjusted = adjust_by_pos_ref(revert_b, pos_ref_value)
    return revert_b, yellow_mask


def split_image(left_cnt: np.array, right_cnt: np.array,
                img: np.array) -> (np.array, np.array):
    """
    Target in left, reference in right
    Args:
        left_cnt:
        right_cnt:
        img: three-channels image
    Returns:
        target: left-part image
        right: right-part image
    """
    img_copy = img.copy()
    x1, y1, w1, h1 = cv2.boundingRect(left_cnt)
    x2, y2, w2, h2 = cv2.boundingRect(right_cnt)
    if not (x1 < x1 + w1 < x2 < x2 + w2):
        show_error('Split image failed.')
    middle = (x1 + w1 + x2) // 2
    target = img_copy[:, :middle]
    ref = img_copy[:, middle:]
    return target, ref


def manual_ref(img: np.array, text=None, method='box') -> (list, np.array):
    """
    Select reference region manually
    """
    log.info(text)
    img_copy = img.copy()
    if method == 'box':
        select = select_box
        select = select_polygon
    else:
        select = select_polygon
    if text is not None:
        # log.info(text)
        mask = select(img_copy, text)
    else:
        mask = select(img_copy)
    blue, yellow_mask = get_real_blue(img, 0, 255)
    mask_no_yellow = np.bitwise_and(mask, 255 - yellow_mask)
    ref_value, std = cv2.meanStdDev(blue, mask=mask)
    ref_value, std = ref_value[0][0], std[0][0]
    area = np.count_nonzero(mask)
    ref_value2, std2 = cv2.meanStdDev(blue, mask=mask_no_yellow)
    ref_value2, std2 = ref_value2[0][0], std2[0][0]
    area2 = np.count_nonzero(mask_no_yellow)
    ratio = area2 / area
    masked = cv2.bitwise_and(blue, blue, mask=mask_no_yellow)
    flatten = masked[masked > 0]
    if area > area2:
        log.info(f'Original expression area: {area}')
        log.info(f'After removing yellow region: {area2}')
    if len(flatten) == 0:
        flatten = np.array([0])
    fig_size = img.shape[0] * img.shape[1]
    result = [ref_value2, std2, area2, ref_value, std, area, ratio, fig_size,
              flatten]
    if debug:
        imshow('mask', mask)
        imshow('masked', cv2.bitwise_and(img, img, mask=mask))
        imshow('b-y', cv2.bitwise_and(blue, blue, mask=255 - yellow_mask))
    return result, mask


def get_contour_wrapper(
        img_file: str, neg_ref_value: float, pos_ref_value: float, big: int,
        text=GENERAL_TEXT) -> (tuple, np.array, list, dict, np.array):
    level_cnt, img = get_contour(img_file)
    filtered_result = filter_contours(img, level_cnt, big=big)
    if filtered_result is not None:
        mask = get_single_mask(filtered_result, level_cnt, img)
        result, _ = calculate(img, mask, neg_ref_value=neg_ref_value,
                              pos_ref_value=pos_ref_value)
    else:
        # log.info(text)
        result, mask = manual_ref(img, text=SHORT_TEXT, method='polygon')
    return result, mask, filtered_result, level_cnt, img


def get_contour(img_file: str) -> (dict, np.array):
    # .png .jpg .tiff .bmp
    log.info(f'Analyzing {img_file}')
    img = cv2.imread(img_file)
    edge = get_edge(img)
    # APPROX_NONE to avoid omitting dots
    contours, raw_hierarchy = cv2.findContours(edge, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
    # at least two objects
    if len(contours) < 3:
        show_error('Cannot find objects in given image.')
    hierarchy = []
    # raw hierarchy is [[[1,1,1,1]]]
    for index, value in enumerate(raw_hierarchy[0]):
        # [next, previous, child, parent, self]
        new_value = np.append(value, index)
        hierarchy.append(new_value)
    level_cnt = {}
    for key, value in zip(hierarchy, contours):
        level_cnt[tuple(key)] = value
    return level_cnt, img


def calculate(original_image: np.array, target_mask: np.array,
              neg_ref_value=0.0, pos_ref_value=255.0) -> (tuple, np.array):
    """
    Calculate given region's value.
    Args:
        original_image: original BGR image
        target_mask: mask of target region
        pos_ref_value: positive reference value for up threshold
        neg_ref_value: negative reference value for down threshold
    Returns:
        result = (express_value, express_std, express_area, total_value,
            total_std, total_area, express_ratio, express_flatten)
        express_mask_no_yellow
    """
    neg_ref_value = int(neg_ref_value)
    pos_ref_value = int(pos_ref_value)
    revert_b, yellow_mask = get_real_blue(original_image, neg_ref_value,
                                          pos_ref_value)
    express_mask = target_mask.copy()
    express_mask[revert_b <= neg_ref_value] = 0
    express_mask_no_yellow = np.bitwise_and(express_mask, 255 - yellow_mask)
    # cv2.contourArea return different value with np.count_nonzero
    total_area = np.count_nonzero(target_mask)
    if total_area == 0:
        show_error('Cannot detect expression region.')
    express_area = np.count_nonzero(express_mask_no_yellow)
    express_ratio = express_area / total_area
    # total_sum = np.sum(revert_b[target_mask>0])
    total_value, total_std = cv2.meanStdDev(revert_b, mask=target_mask)
    total_value, total_std = total_value[0][0], total_std[0][0]
    express_value, express_std = cv2.meanStdDev(revert_b,
                                                mask=express_mask_no_yellow)
    express_value, express_std = express_value[0][0], express_std[0][0]
    # violin plot also use yellow mask
    express_flatten_ = revert_b[express_mask_no_yellow > 0]
    express_flatten = express_flatten_[express_flatten_ > 0]
    fig_size = original_image.shape[0] * original_image.shape[1]
    result = (express_value, express_std, express_area, total_value, total_std,
              total_area, express_ratio, fig_size, express_flatten)
    log.debug('express_value, express_std, express_area, total_value, '
              'total_std, total_area, express_ratio, fig_size, express_flatten')
    log.debug(result)
    if debug:
        pass
        # imshow('original', original_image)
        # imshow('target', target_mask)
        # imshow('express', express_mask)
        # imshow('express no yellow', express_mask_no_yellow)
    return result, express_mask_no_yellow


def get_zscore(values: list) -> list:
    """
    Args:
        values: value list
    Returns:
        z_scores: list
    """
    z_scores = []
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return [1, ] * len(values)
    for i in values:
        z_score = (i - mean) / std
        z_scores.append(z_score)
    return z_scores


def draw_images(filtered_result: list, level_cnt: dict, img: np.array,
                simple=False, show=False, filename=None) -> dict:
    """
    Return None for empty
    """
    if filtered_result is None:
        return {}

    def drawing(levels, color):
        line_width = 2
        for level in levels:
            cnt = level_cnt[level]
            min_rect = cv2.minAreaRect(cnt)
            min_rect_points = np.int0(cv2.boxPoints(min_rect))
            x, y, w, h = cv2.boundingRect(cnt)
            approx = cv2.approxPolyDP(cnt, arc_epsilon, True)
            # b,g,r
            cv2.fillPoly(img_dict['fill'], [approx], color)
            if not simple:
                cv2.rectangle(img_dict['rectangle'], (x, y), (x + w, y + h),
                              color, line_width)
                cv2.drawContours(img_dict['min_area_rectangle'],
                                 [min_rect_points], 0, color, line_width)
                cv2.polylines(img_dict['polyline'], [approx], True, color,
                              line_width)

    (big_external_contours, small_external_contours, inner_contours,
     fake_inner, real_background) = filtered_result
    arc_epsilon = get_arc_epsilon(level_cnt[big_external_contours[0]])
    img_dict = dict()
    img_dict['fill'] = img.copy()
    if not simple:
        img_dict['raw'] = img
        img_dict['rectangle'] = img.copy()
        img_dict['min_area_rectangle'] = img.copy()
        img_dict['polyline'] = img.copy()
    color_blue = hex2bgr('#4d96ff')
    color_green = hex2bgr('#6bcb77')
    color_red = hex2bgr('#ff6b6b')
    color_yellow = hex2bgr('#ffd93d')
    drawing(big_external_contours, color_blue)
    drawing(small_external_contours, color_red)
    drawing(inner_contours, color_yellow)
    drawing(fake_inner, color_green)
    drawing(real_background, color_red)
    for title, image in img_dict.items():
        if show:
            imshow(title, image)
        if filename is not None:
            out_p = Path(filename)
            out_filename = str(out_p.parent / out_p.with_name(
                f'{out_p.stem}_{title}.png'))
            cv2.imwrite(out_filename, image)
            log.debug(f'Write image {out_filename}')
    return img_dict


def get_out_filename(image: str, stem2: str) -> Path:
    png = Path(image)
    png = png.with_name(png.stem + stem2 + '.png')
    return png


def write_masks(img_raw: np.array, target_mask: np.array,
                express_mask: np.array,
                output: Path) -> Path:
    alpha = np.zeros(img_raw.shape[:2], dtype='uint8')
    alpha[target_mask > 0] = 128
    alpha[express_mask > 0] = 255
    b, g, r = cv2.split(img_raw)
    merged = cv2.merge([b, g, r, alpha])
    cv2.imwrite(str(output), merged)
    return output


def write_image(results: list, labels: list, out: Path) -> Path:
    """
    violin outer and inner
    or violin outer and bar inner
    Args:
        results: calculate results
        labels: x-axis ticks
        out: output filename
    Returns:
        out: figure file
    """
    # result = (express_value, express_std, express_area, total_value,
    # total_std, total_area, express_ratio, express_flatten)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    width = 0.6
    short_labels = [Path(i).stem for i in labels]
    if len(labels) <= 5:
        figsize = (10, 6)
    else:
        figsize = (2 * len(labels), 8)
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(211)
    x = np.arange(1, len(labels) + 1)
    violin_data_raw = []
    for i in results:
        data = i[-1]
        if len(data) == 0:
            data = [0]
        violin_data_raw.append(data)
    violin_data = np.array(violin_data_raw, dtype='object')
    # violin_data = np.array([i[-1] for i in results], dtype='object')
    try:
        violin_parts = ax1.violinplot(violin_data, showmeans=True,
                                      showmedians=False, showextrema=False,
                                      widths=width)
    except ValueError:
        show_error('Failed to plot results due to bad values.')
        return Path()
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#0d56ff')
        pc.set_edgecolor('black')
    ax1.set_ylabel('Expression value')
    ax1.set_yticks(np.linspace(0, 256, 5))
    ax1.set_xticks(np.arange(1, len(labels) + 1), labels=short_labels)
    # ax2 = ax1.twinx()
    ax2 = plt.subplot(212)
    express_area = [i[2] for i in results]
    # modify negative reference area
    # express_area[-1] = 0
    all_area = [i[-2] for i in results]
    total_area = [i[5] for i in results]
    no_express_area = [t - e for t, e in zip(total_area, express_area)]
    express_area_percent = [round(i / j, 4) for i, j in zip(express_area,
                                                            all_area)]
    no_express_area_percent = [round(i / j, 4) for i, j in zip(no_express_area,
                                                               all_area)]
    rects1 = ax2.bar(x, express_area_percent, width=width, alpha=0.5,
                     color='green', label='Expression region')
    rects2 = ax2.bar(x, no_express_area_percent, width=width,
                     bottom=express_area_percent, alpha=0.4,
                     color='orange', label='No expression region')
    # rects1 = [i * 100 for i in rects1]
    # ax2.bar_label(rects1, label_type='center')
    # ax2.bar_label(rects2, label_type='center')
    ax2.set_xticks(np.arange(1, len(labels) + 1), labels=short_labels)
    ax2.legend()
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Area percent')
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=1))
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    log.info(f'Output figure file {out}')
    return out


def write_csv(all_result: list, targets: list, out: Path) -> Path:
    """
    Output csv
    """
    header = ('Name,Expression value,Expression std,Expression area,'
              'Total value,Total std,Total area,Expression ratio,Figure size,'
              'Z-score,Outlier,Estimate color temperature(K)')
    # use z score to find out outliers
    z_score_threshold = 3
    values = [i[0] for i in all_result]
    z_scores = get_zscore(values)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header.split(','))
        for name, result, z_score in zip(targets, all_result, z_scores):
            is_outlier = (np.abs(z_score) > z_score_threshold)
            estimate_CCT = get_CCT(cv2.imread(name))
            if is_outlier:
                log.warning(f'{name} has abnormal expression value.')
            numbers = [round(i, 4) for i in result[:-1]]
            z_score = round(z_score, 4)
            writer.writerow([name, *numbers, z_score, is_outlier, estimate_CCT])
    log.info(f'Output table file {out}')
    return out


def cli_main(arg_str=None) -> (Path, Path):
    if __name__ == '__main__':
        log.info('Welcome to pyGUS.')
    arg = parse_arg(arg_str)
    negative, positive, targets, auto_ref, quick, message = get_input(arg)
    pdf_file = None
    csv_file = None
    if message is not None:
        show_error(message)
    log.info(f'Running mode {arg.mode}...')
    log.info(f'Negative reference image: {negative}')
    log.info(f'Positive reference image: {positive}')
    log.info('Target images:')
    for i in targets:
        log.info(f'\t{i}')
    run_dict = {1: mode_1, 2: mode_2, 3: mode_3, 4: mode_4}
    run = run_dict[arg.mode]
    neg_result, pos_result, target_results = run(negative, positive, targets,
                                                 auto_ref, quick)
    neg_result = list(neg_result[:-1])
    neg_result.append([0])
    for i in neg_result, pos_result, target_results:
        if i is None:
            return pdf_file, csv_file
    # add ref results
    target_results.append(pos_result)
    target_results.append(neg_result)
    targets.extend(('Positive reference', 'Negative reference'))
    pdf_file = Path(targets[0]).parent / 'Result.pdf'
    csv_file = pdf_file.with_suffix('.csv')
    for f in pdf_file, csv_file:
        if f.exists():
            log.warning(f'{f} exists, overwrite.')
    write_image(target_results, targets, pdf_file)
    write_csv(target_results, targets, csv_file)
    # wait or poll
    cv2.pollKey()
    if debug:
        cv2.waitKey()
    cv2.destroyAllWindows()
    log.info('Done.')
    return pdf_file, csv_file


if __name__ == '__main__':
    cli_main()