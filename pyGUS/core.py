#!/usr/bin/python3.10
import argparse
import csv
from pathlib import Path

# for nuitka
from matplotlib.backends import backend_pdf
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')

from pyGUS.global_vars import log
from pyGUS.utils import select_polygon, color_calibrate, if_exist, show_error


# todo: color correction test
# todo mode 1 test: single object for each image, manually select positive,
#  negative, targets
# todo mode 2 test: two object for each image, left target, right positive
# todo mode 3 test: two object for each image, left target, right color card
# todo: mode 4 test: select area by mouse
# todo: calculate values, statistic analysis
# todo: GUI
# todo: manual
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
    arg.add_argument('-images', nargs='*', help='Input images', required=True)
    arg.add_argument('-mode', type=int, choices=(1, 2, 3, 4), required=True,
                     help=('1. single target in each image; '
                           '2. target and positive reference in each image; '
                           '3. target and colorchecker in each image; '
                           '4. manually select target regions with mouse'))
    if arg_str is None:
        return arg.parse_args()
    else:
        return arg.parse_args(arg_str.split(' '))


def get_input(arg):
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
    return negative, positive, targets, message


def mode_1(negative, positive, targets):
    # ignore light change
    # first: negative
    # second: positive
    # third and after: target
    neg_filtered_result, neg_level_cnt, neg_img = get_contour(negative)
    pos_filtered_result, pos_level_cnt, pos_img = get_contour(positive)
    neg_mask = get_single_mask(neg_filtered_result, neg_level_cnt, neg_img)
    pos_mask = get_single_mask(pos_filtered_result, pos_level_cnt, pos_img)
    neg_result = calculate(neg_img, neg_mask)
    neg_ref_value = neg_result[0]
    pos_result = calculate(pos_img, pos_mask, neg_ref_value=neg_ref_value)
    pos_ref_value = pos_result[0]
    log.debug(f'neg {neg_ref_value} pos {pos_ref_value}')
    target_results = []
    for target in targets:
        filtered_result, level_cnt, img = get_contour(target)
        target_mask = get_single_mask(filtered_result, level_cnt, img)
        target_result = calculate(img, target_mask, neg_ref_value=neg_ref_value,
                                  pos_ref_value=pos_ref_value)
        target_results.append(target_result)
        img_dict = draw_images(filtered_result, level_cnt, img, simple=True,
                               show=False, filename=target)
    neg_img_dict = draw_images(neg_filtered_result, neg_level_cnt, neg_img,
                               show=False, simple=True, filename=negative)
    pos_img_dict = draw_images(pos_filtered_result, pos_level_cnt, pos_img,
                               show=False, simple=True, filename=positive)
    return neg_result, pos_result, target_results


def mode_2(ref1, ref2, targets):
    # use negative to calibrate positive, and then measure each target
    # assume positive in each image is same, ignore light change
    # first left: negative, first right: positive
    # next left: object, next right: positive
    # ignore small_external, inner_contours,
    negative_positive_ref = ref1
    ref_filtered_result, ref_level_cnt, ref_img = get_contour(
        negative_positive_ref)
    neg_mask, pos_mask = get_left_right_mask(ref_filtered_result,
                                             ref_level_cnt, ref_img)
    neg_result = calculate(ref_img, neg_mask)
    neg_ref_value = neg_result[0]
    pos_result = calculate(ref_img, pos_mask, neg_ref_value=neg_ref_value)
    pos_ref_value = pos_result[0]
    log.debug(f'neg {neg_ref_value} pos {pos_ref_value}')
    target_results = []
    for target in targets:
        filtered_result, level_cnt, img = get_contour(target)
        target_mask, pos_mask_ = get_left_right_mask(filtered_result, level_cnt,
                                                     img)
        target_result = calculate(img, target_mask, neg_ref_value=neg_ref_value,
                                  pos_ref_value=pos_ref_value)
        target_results.append(target_result)
        img_dict = draw_images(filtered_result, level_cnt, img, show=False,
                               simple=True, filename=target)
    ref_img_dict = draw_images(ref_filtered_result, ref_level_cnt, ref_img,
                               show=False, simple=True,
                               filename=negative_positive_ref)
    masked_neg = cv2.bitwise_and(ref_img, ref_img, mask=neg_mask)
    # cv2.imshow('masked negative reference', 255 - masked_neg)
    masked_pos = cv2.bitwise_and(ref_img, ref_img, mask=pos_mask)
    # cv2.imshow('masked positive reference', 255 - masked_pos)
    return neg_result, pos_result, target_results


def mode_3(ref1, ref2, targets):
    # use color card to calibrate each image
    # first left: negative, first right: card
    # second left: positive, second right: card
    # third and next left: target, right: card
    negative = ref1
    positive = ref2
    ok_neg = color_calibrate(negative)
    ok_pos = color_calibrate(positive)
    ok_targets = [color_calibrate(i) for i in targets]
    ###
    neg_filtered_result, neg_level_cnt, neg_img = get_contour(ok_neg)
    neg_left_mask, neg_right_mask = get_left_right_mask(neg_filtered_result,
                                                        neg_level_cnt, neg_img)
    neg_result = calculate(neg_img, neg_left_mask)
    neg_ref_value = neg_result[0]
    # right_result = calculate(img, right_mask, neg_ref_value, pos_ref_value)
    pos_filtered_result, pos_level_cnt, pos_img = get_contour(ok_pos)
    pos_left_mask, pos_right_mask = get_left_right_mask(pos_filtered_result,
                                                        pos_level_cnt, pos_img)
    pos_result = calculate(pos_img, pos_left_mask)
    pos_ref_value = pos_result[0]
    log.debug(f'neg {neg_ref_value} pos {pos_ref_value}')
    target_results = []
    for target in ok_targets:
        filtered_result, level_cnt, img = get_contour(target)
        left_mask, right_mask = get_left_right_mask(filtered_result, level_cnt,
                                                    img)
        target_result = calculate(img, left_mask, neg_ref_value, pos_ref_value)
        target_results.append(target_result)
        img_dict = draw_images(filtered_result, level_cnt, img, show=False,
                               simple=True, filename=target)
    neg_img_dict = draw_images(neg_filtered_result, neg_level_cnt, neg_img,
                               show=False, simple=True, filename=negative)
    pos_img_dict = draw_images(pos_filtered_result, pos_level_cnt, pos_img,
                               show=False, simple=True, filename=positive)
    return neg_result, pos_result, target_results


def mode_4(ref1, ref2, targets):
    """
    Select region manually
    """
    #
    name_dict = {'neg': ('Negative reference', hex2bgr('#FF6B6B')),
                 'pos': ('Positive reference', hex2bgr('#FFd93D')),
                 'target': ('Target region', hex2bgr('#6BCB77'))}
    neg_result = []
    pos_result = []
    target_results = []
    for target in targets:
        log.info(f'Analyzing {target}')
        img = cv2.imread(target)
        if img is None:
            show_error(f'Bad image file {target}')
        img_copy = img.copy()
        cropped1, mask1 = select_polygon(img_copy, name_dict['neg'][0],
                                         name_dict['neg'][1])
        cropped2, mask2 = select_polygon(img_copy, name_dict['pos'][0],
                                         name_dict['pos'][1])
        cropped3, mask3 = select_polygon(img_copy, name_dict['target'][0],
                                         name_dict['target'][1])
        try:
            cv2.imshow('Press space bar to continue', img_copy)
            cv2.waitKey()
            cv2.destroyAllWindows()
        except cv2.error:
            error_msg = 'Bad selection.'
            show_error(error_msg)
            return neg_result, pos_result, target_results
        # todo: is it ok to directly use calculate to get ref value?
        neg_result = calculate(img, mask1, neg_ref_value=0)
        neg_ref_value, neg_std, *_ = neg_result
        pos_result = calculate(img, mask2, pos_ref_value=255)
        pos_ref_value, pos_std, *_ = pos_result
        # neg_ref_value += neg_std * 3
        neg_ref_value += neg_std
        pos_ref_value += pos_std
        log.debug(f'neg {neg_ref_value} pos {pos_ref_value}')
        result = calculate(img, mask3, neg_ref_value, pos_ref_value)
        target_results.append(result)
    return neg_result, pos_result, target_results


def fill_mask(shape, target, fake_inner, inner_background, level_cnt):
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


def get_single_mask(filtered_result, level_cnt, img):
    # big_external + fake_inner - inner_background
    (big_external_contours, small_external_contours, inner_contours,
     fake_inner, inner_background) = filtered_result
    target = big_external_contours[0]
    self_index = target[4]
    # log.debug(f'self {self_index}')
    # log.debug(f'contour area {cv2.contourArea(target)}')
    mask = fill_mask(img.shape[:2], target, fake_inner, inner_background,
                     level_cnt)
    return mask


def get_left_right_mask(filtered_result, level_cnt, img):
    (big_external_contours, small_external_contours, inner_contours,
     fake_inner, inner_background) = filtered_result
    # target, ref
    left, right = get_left_right(big_external_contours, level_cnt)
    if left is None and right is None:
        show_error('Object not found.')
    left_right_mask = list()
    for target in left, right:
        self_index = target[4]
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


def auto_Canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # use edited lower bound
    lower = int(max(0, (1.0 - sigma * 2) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edge = cv2.Canny(image, lower, upper)
    return edge


def threshold(img, show=False):
    # todo: find suitable edge to split target and edge
    r, g, b = cv2.split(img)
    r_g = r // 2 + g // 2
    r_g_reverse = 255 - r_g
    blur = cv2.GaussianBlur(r_g_reverse, (5, 5), 0)
    h, w = img.shape[:2]
    # mask = np.zeros([h + 2, w + 2], np.uint8)
    # ret1, th1 = cv2.threshold(img, 16, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    equalize = cv2.equalizeHist(blur)
    r, t = cv2.threshold(equalize, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge = auto_Canny(255 - th2)
    # cv2.floodFill(th2, mask=mask, seedPoint=(1,1), newVal=0, loDiff=3,
    # upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    log.debug(f'h * w: {h} {w}')
    # cv2.floodFill(th2, mask=mask, seedPoint=(1,h-1), newVal=0, loDiff=3,
    # upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    # cv2.floodFill(th2, mask=mask, seedPoint=(w-1,1), newVal=0, loDiff=3,
    # upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    # cv2.floodFill(th2, mask=mask, seedPoint=(w-1,h-1), newVal=0, loDiff=3,
    # upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    # cv2.THRESH_BINARY, 11, 4)
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    # cv2.THRESH_BINARY, 11, 2)
    th3 = auto_Canny(th2)
    if show:
        cv2.imshow('th2', th2)
        cv2.imshow('threshold', t)
        cv2.imshow('t_edge', edge)
    log.debug(f'ret2 {ret2}')
    return th3


def get_edge(image):
    # edge->blur->dilate->erode->contours
    # img_equalize = cv2.equalizeHist(image)
    # clahe = cv2.createCLAHE()
    # sharped = clahe.apply(image)
    # blur1 = cv2.GaussianBlur(img_equalize, (3, 3), 0)
    img_equalize = image
    # threshold(image)
    edge = auto_Canny(img_equalize)
    # blur edge, not original image
    blur = cv2.GaussianBlur(edge, (5, 5), 0)
    dilate = cv2.dilate(blur, None)
    erode_edge = cv2.erode(dilate, None)
    # cv2.imshow('edge', edge)
    return erode_edge


def revert(img):
    return 255 - img


def get_arc_epsilon(max_contour, ratio=0.0001):
    log.debug(f'Max contour area: {cv2.contourArea(max_contour)}')
    arc_epsilon = cv2.arcLength(max_contour, True) * ratio
    log.debug(f'Set arc epsilon: {arc_epsilon}')
    return arc_epsilon


def get_contour_value(img, cnt, with_std=True):
    # fill contour with (255,255,255)
    mask = np.zeros(img.shape[:2], dtype='uint8')
    cv2.fillPoly(mask, [cnt], (255, 255, 255))
    if with_std:
        mean, std = cv2.meanStdDev(img, mask=mask)
        return mean[0][0], std[0][0]
    else:
        mean = cv2.mean(img, mask=mask)
        return mean[0], 0


def get_background_value(img, external_contours, level_cnt):
    # assume background value is greater than negative reference value
    mask = np.ones(img.shape[:2], dtype='uint8')
    for external in external_contours:
        cnt = level_cnt[external]
        cv2.fillPoly(mask, [cnt], (0, 0, 0))
    mean, std = cv2.meanStdDev(img, mask=mask)
    # cv2.imshow('Background masked', masked)
    return mean[0][0], std[0][0]


def remove_fake_inner_cnt(img, level_cnt, big_external_contours,
                          external_contours, inner_contours):
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


def filter_contours(img, level_cnt: dict) -> (list, list, list):
    """
    Args:
        img: original image
        level_cnt(dict):
    Returns:
        big:
        small:
        inner:
        fake_inner:
        inner_background:
    """
    external_contours = list()
    inner_contours = list()
    for level, cnt in level_cnt.items():
        next_, previous_, first_child, parent, self_ = level
        # -1 means no parent -> external
        if parent == -1:
            external_contours.append(level)
        else:
            inner_contours.append(level)
    external_contours.sort(key=lambda key: cv2.contourArea(level_cnt[key]),
                           reverse=True)
    # a picture only contains at most TWO target (sample and reference)
    big_external_contours = external_contours[:2]
    try:
        small_external_contours = external_contours[2:]
    except IndexError:
        small_external_contours = list()
    fake_inner, inner_background = remove_fake_inner_cnt(
        img, level_cnt, big_external_contours, external_contours,
        inner_contours)
    return (big_external_contours, small_external_contours, inner_contours,
            fake_inner, inner_background)


def get_left_right(big_external_contours, level_cnt):
    """
    Left is target, right is ref
    Split images to left and right according to bounding rectangle of
    external contours.
    Args:
        big_external_contours:
        level_cnt:
    Returns:
        left:
        right:
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


def show_channel(img):
    # show rgb channel
    # opencv use BGR
    b, g, r = cv2.split(img)
    for title, value in zip(['b', 'g', 'r'], [b, g, r]):
        cv2.imshow(title + 'revert', 255 - value)
        cv2.imshow(title, value)


def hex2bgr(hex_str: str):
    """
    Args:
        hex_str: #FFFFFF
    Returns:
        255, 255, 255
    """
    hex2 = hex_str.removeprefix('#')
    r = int('0x' + hex2[0:2].lower(), base=16)
    g = int('0x' + hex2[2:4].lower(), base=16)
    b = int('0x' + hex2[4:6].lower(), base=16)
    return b, g, r


def draw_images(filtered_result, level_cnt, img, simple=False, show=False,
                filename=None):
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
            cv2.imshow(title, image)
        if filename is not None:
            out_p = Path(filename)
            out_filename = str(out_p.parent / out_p.with_name(
                f'{out_p.stem}_{title}.png'))
            cv2.imwrite(out_filename, image)
            log.debug(f'Write image {out_filename}')
    return img_dict


def get_real_blue(original_image, neg_ref_value, pos_ref_value):
    if neg_ref_value > pos_ref_value or pos_ref_value <= 0:
        show_error('Bad negative and positive reference values.')
    b, g, r = cv2.split(original_image)
    factor = 1
    revert_b = revert(b)
    amplified_neg_ref = int(factor * neg_ref_value)
    return revert_b, amplified_neg_ref


def get_real_blue_2(original_image, neg_ref_value, pos_ref_value):
    # todo, 255-b is not real blue part
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
    # todo: is it ok?
    revert_b = revert_b.astype('float')
    factor = 255 // pos_ref_value
    revert_b = (revert_b - neg_ref_value) * factor
    revert_b[revert_b > 255] = 255
    revert_b = revert_b.astype('uint8')
    amplified_neg_ref = int(factor * neg_ref_value)
    return revert_b, amplified_neg_ref


def calculate(original_image, target_mask, neg_ref_value=0, pos_ref_value=255):
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
    """
    # todo: remove green
    # blue express area
    revert_b, amplified_neg_ref = get_real_blue(original_image, neg_ref_value,
                                                pos_ref_value)
    # cv2.imshow('revert blue', revert_b)
    # cv2.waitKey()
    express_mask = target_mask.copy()
    express_mask[revert_b <= amplified_neg_ref] = 0
    # cv2.contourArea return different value with np.count_nonzero
    total_area = np.count_nonzero(target_mask)
    express_area = np.count_nonzero(express_mask)
    # todo: how to get correct ratio
    express_ratio = express_area / total_area
    # total_sum = np.sum(revert_b[target_mask>0])
    total_value, total_std = cv2.meanStdDev(revert_b, mask=target_mask)
    total_value, total_std = total_value[0][0], total_std[0][0]
    express_value, express_std = cv2.meanStdDev(revert_b, mask=express_mask)
    express_value, express_std = express_value[0][0], express_std[0][0]
    express_flatten_ = revert_b[express_mask > 0]
    express_flatten = express_flatten_[express_flatten_ > 0]
    result = (express_value, express_std, express_area, total_value, total_std,
              total_area, express_ratio, express_flatten)
    log.debug('express_value, express_std, express_area, total_value, '
              'total_std, total_area, express_ratio, express_flatten')
    log.debug(result)
    return result


def split_image(left_cnt, right_cnt, img):
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


def get_contour(img_file):
    # .png .jpg .tiff
    log.info(f'Analyzing {img_file}')
    img = cv2.imread(img_file)
    # show_channel(img)
    b, g, r = cv2.split(img)
    # reverse to get better edge
    # use green channel
    # todo: g or b
    revert_img = revert(g // 2 + r // 2)
    # revert_img = revert(g)
    edge = get_edge(revert_img)
    # APPROX_NONE to avoid omitting dots
    contours, raw_hierarchy = cv2.findContours(edge, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
    hierarchy = list()
    # raw hierarchy is [[[1,1,1,1]]]
    for index, value in enumerate(raw_hierarchy[0]):
        # [next, previous, child, parent, self]
        new_value = np.append(value, index)
        hierarchy.append(new_value)
    level_cnt = dict()
    for key, value in zip(hierarchy, contours):
        level_cnt[tuple(key)] = value
    filtered_result = filter_contours(img, level_cnt)
    return filtered_result, level_cnt, img


def write_image(results, labels, out):
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
    if len(labels) <= 5:
        figsize = (10, 6)
    else:
        figsize = (10*len(labels)/5, 6)
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(211)
    _ = results[0][-1]
    x = np.arange(1, len(labels) + 1)
    width = 0.2
    try:
        violin_parts = ax1.violinplot([i[-1] for i in results], showmeans=False,
                                      showmedians=False, showextrema=False,
                                      widths=0.4)
    except ValueError:
        show_error('Failed to plot results due to bad values.')
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#0d56ff')
        pc.set_edgecolor('black')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Expression value', color='b')
    ax1.set_yticks(np.linspace(0, 256, 9))
    short_labels = [Path(i).name for i in labels]
    ax1.set_xticks(np.arange(1, len(labels) + 1), labels=short_labels)
    # ax2 = ax1.twinx()
    ax2 = plt.subplot(212)
    rects1 = ax2.bar(x - width / 2, [i[2] for i in results], width=width,
                     alpha=0.4,
                     color='green', label='Express area')
    rects2 = ax2.bar(x + width / 2, [i[5] for i in results], width=width,
                     alpha=0.4,
                     color='orange', label='Total area')
    ax2.bar_label(rects1, padding=3)
    ax2.bar_label(rects2, padding=3)
    ax2.set_xticks(np.arange(1, len(labels) + 1), labels=short_labels)
    ax2.legend()
    ax2.set_ylabel('Area (pixels)')
    plt.tight_layout()
    plt.savefig(out)
    log.info(f'Output figure file {out}')
    return out


def write_csv(all_result, targets, out):
    """
    Output csv
    Args:
        all_result:
        targets:
        out:
    Returns:
        out:
    """
    header = ('Name,Express value, Express std,Express area, Total value,'
              'Total std,Total Area,Express ratio')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header.split(','))
        for name, result in zip(targets, all_result):
            writer.writerow([name, *result[:-1]])
    log.info(f'Output table file {out}')
    return out


def cli_main(arg_str=None):
    log.info('Welcome to pyGUS.')
    arg = parse_arg(arg_str)
    negative, positive, targets, message = get_input(arg)
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
    neg_result, pos_result, target_results = run(negative, positive, targets)
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
    cv2.destroyAllWindows()
    log.info('Done.')
    # todo: 30s per image, too slow
    return pdf_file, csv_file


if __name__ == '__main__':
    cli_main()