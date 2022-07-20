#!/usr/bin/python3.10
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from pyGUS.global_vars import log
from pyGUS.utils import select_polygon, color_calibrate, if_exist


# todo: color correction test
# todo mode 1 test: single object for each image, manually select positive, negative, targets
# todo mode 2 test: two object for each image, left target, right positive
# todo mode 3 test: two object for each image, left target, right color card
# todo: mode 4 test: select area by mouse
# todo: calculate values, statistic analysis
# todo: GUI
# todo: manual
# todo: manuscript


def parse_arg():
    arg = argparse.ArgumentParser(prog='pyGUS')
    arg.add_argument('-ref1', help='Negative expression reference image')
    arg.add_argument('-ref2', help='Positive expression reference image')
    arg.add_argument('-images', nargs='*', help='Input images', required=True)
    arg.add_argument('-mode', type=int, choices=(1, 2, 3, 4), required=True,
                     help=('1. single target in each image; '
                           '2. target and positive reference in each image; '
                           '3. target and colorchecker in each image; '
                           '4. manually select target regions with mouse'))
    return arg.parse_args()


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
        log.error(message)
        raise SystemExit(-1)
    targets = [Path(i).absolute() for i in arg.images]
    targets = [if_exist(i) for i in targets]
    if arg.mode != 4:
        negative = Path(arg.ref1).absolute()
        negative = if_exist(negative)
    if arg.mode == 1:
        positive = Path(arg.ref2).absolute()
        positive = if_exist(positive)
    return negative, positive, targets


def mode_1(negative, positive, targets):
    # ignore light change
    # first: negative
    # second: positive
    # third and after: target
    neg_filtered_result, neg_level_cnt, neg_img = get_contour(negative)
    pos_filtered_result, pos_level_cnt, pos_img = get_contour(positive)
    neg_value, neg_std = get_single_value(
        neg_filtered_result, neg_level_cnt, neg_img)
    pos_value, pos_std = get_single_value(
        pos_filtered_result, pos_level_cnt, pos_img)
    for target in targets:
        filtered_result, level_cnt, img = get_contour(target)
        target_value, target_std = get_single_value(
            filtered_result, level_cnt, img)
    print(neg_value, neg_std, pos_value, pos_std, target_value, target_std)


def mode_2(negative, positive, targets):
    # use negative to calibrate positive, and then measure each target
    # assume positive in each image is same, ignore light change
    # first left: negative, first right: positive
    # next left: object, next right: positive
    # ignore small_external, inner_contours,
    neg_filtered_result, neg_level_cnt, neg_img = get_contour(negative)
    neg_value, neg_std, pos_value, pos_std = get_left_right_value(
        neg_filtered_result, neg_level_cnt, neg_img)
    for target in targets:
        filtered_result, level_cnt, img = get_contour(target)
        target_value, target_std, pos_value_, pos_std_ = get_left_right_value(
            filtered_result, level_cnt, img)
    pass


def mode_3(negative, positive, targets):
    # use color card to calibrate each image
    # first left: negative, first right: card
    # second left: positive, second right: card
    # third and next left: target, right: card
    ok_neg = color_calibrate(negative)
    ok_pos = color_calibrate(positive)
    ok_targets = [color_calibrate(i) for i in targets]
    ###
    neg_filtered_result, neg_level_cnt, neg_img = get_contour(ok_neg)
    pos_filtered_result, pos_level_cnt, pos_img = get_contour(ok_pos)
    neg_value, neg_std, card_value, card_std = get_left_right_value(
        neg_filtered_result, neg_level_cnt, neg_img)
    pos_value, pos_std, card_value2, card_std2 = get_left_right_value(
        pos_filtered_result, pos_level_cnt, pos_img)
    for target in ok_targets:
        filtered_result, level_cnt, img = get_contour(target)
        target_value, target_std, card_value_, card_std_ = get_left_right_value(
            filtered_result, level_cnt, img)
    print(neg_value, neg_std, pos_value, pos_std, target_value, target_std)
    pass


def mode_4(negative, positive, targets):
    """
    Select region manually
    """
    name_dict = {'neg': ('Negative reference', (0, 0, 255)),
                 'pos': ('Positive reference', (0, 255, 0)),
                 'target': ('Target region', (255, 0, 0))}
    all_result = []
    for target in targets:
        img = cv2.imread(target)
        cropped1, mask1 = select_polygon(img, name_dict['neg'][0],
                                         name_dict['neg'][1])
        cropped2, mask2 = select_polygon(img, name_dict['pos'][0],
                                         name_dict['pos'][1])
        cropped3, mask3 = select_polygon(img, name_dict['target'][0],
                                         name_dict['target'][1])
        try:
            cv2.imshow('neg', cropped1)
            cv2.imshow('pos', cropped2)
            cv2.imshow('target', cropped3)
            cv2.waitKey()
            cv2.destroyAllWindows()
        except cv2.error:
            log.error('Bad selection.')
            raise SystemExit(-3)
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
        all_result.append(result)
    all_result.append(pos_result)
    all_result.append(neg_result)
    targets.extend(('Positive reference', 'Negative reference'))
    svg_file = Path(targets[0]).with_suffix('.svg')
    csv_file = svg_file.with_suffix('.csv')
    draw(all_result, targets, svg_file)
    write_csv(all_result, targets, csv_file)
    return svg_file, csv_file


def get_single_value(filtered_result, level_cnt, img):
    # big_external + fake_inner - inner_background
    (big_external_contours, small_external_contours, inner_contours,
     fake_inner, inner_background) = filtered_result
    target = big_external_contours[0]
    self_index = target[4]
    cnt = level_cnt[target]
    log.debug(f'self {self_index}')
    log.debug(f'contour area {cv2.contourArea(target)}')
    mask = np.zeros(img.shape[:2], dtype='uint8')
    # related inner background
    cv2.fillPoly(mask, [cnt], (255, 255, 255))
    for i in fake_inner:
        cnt_i = level_cnt[i]
        cv2.fillPoly(mask, [cnt_i], (255, 255, 255))
    for j in inner_background:
        cnt_j = level_cnt[j]
        cv2.fillPoly(mask, [cnt_j], (0, 0, 0))
    masked = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('mask', revert(mask))
    real_b = get_real_blue(img)
    value, std = cv2.meanStdDev(real_b, mask=mask)
    cv2.imshow('mask3', real_b)
    log.debug(f'{value} {std}')
    calculate(img, mask)
    return value[0][0], std[0][0]


def get_left_right_value(filtered_result, level_cnt, img):
    (big_external_contours, small_external_contours, inner_contours,
     fake_inner, inner_background) = filtered_result
    # target, ref
    left, right = get_left_right(big_external_contours, level_cnt)
    assert left is not None and right is not None, 'Object and reference not found.'
    left_cnt = level_cnt[left]
    right_cnt = level_cnt[right]
    left_right_mask = list()
    for target in left, right:
        self_index = target[4]
        cnt = level_cnt[target]
        log.debug(f'target {target}')
        log.debug(f'self {self_index}')
        mask = np.zeros(img.shape[:2], dtype='uint8')
        related_fake_inner = [i for i in fake_inner if i[3] == self_index]
        # related inner background
        related_inner_bg = [i for i in inner_background if i[3] == target[4]]
        cv2.fillPoly(mask, [cnt], (255, 255, 255))
        for i in related_fake_inner:
            cnt_i = level_cnt[i]
            cv2.fillPoly(mask, [cnt_i], (255, 255, 255))
        for j in related_inner_bg:
            cnt_j = level_cnt[j]
            cv2.fillPoly(mask, [cnt_j], (0, 0, 0))
        left_right_mask.append(mask)
    left_mask, right_mask = left_right_mask
    masked_left = cv2.bitwise_and(img, img, mask=left_mask)
    cv2.imshow('mask', 255 - masked_left)
    masked_right = cv2.bitwise_and(img, img, mask=right_mask)
    cv2.imshow('mask2', 255 - masked_right)
    b, g, r = cv2.split(img)
    # todo, 255-b is not real blue part
    left_value, left_std = cv2.meanStdDev(255 - b, mask=left_mask)
    cv2.imshow('mask3', 255 - b)
    right_value, right_std = cv2.meanStdDev(255 - b, mask=right_mask)
    calculate(img, left_mask)
    calculate(img, right_mask)
    return left_value[0][0], left_std[0][0], right_value[0][0], right_std[0][0]




def get_input_demo(input_file='example/ninanjie-ok-75-2.tif'):
    # input_path = 'example/example.png'
    # input_file = 'example/ninanjie-0-1.tif'
    input_file = 'example/ninanjie-75-2.tif'
    # input_file = 'example/ninanjie-50-1.tif'
    # input_file = 'example/ninanjie-ok-75-2.tif'
    # input_file = 'example/ninanjie-100-1.tif'
    # input_file = 'example/ninanjie-2h-3.tif'
    # input_file = 'example/ersuiduanbingcao-BD15-27.png'
    # input_file = 'example/ersuiduanbingcao-DH39-5.png'
    # input_file = 'example/ersuiduanbingcao-ok-BD7-4.png'
    img = cv2.imread(input_file)
    log.info(f'Image size: {img.shape}')
    return input_file


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
    mask = np.zeros([h + 2, w + 2], np.uint8)
    # ret1, th1 = cv2.threshold(img, 16, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    equalize = cv2.equalizeHist(blur)
    r, t = cv2.threshold(equalize, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge = auto_Canny(255 - th2)
    # cv2.floodFill(th2, mask=mask, seedPoint=(1,1), newVal=0, loDiff=3, upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    log.debug(f'h * w: {h} {w}')
    # cv2.floodFill(th2, mask=mask, seedPoint=(1,h-1), newVal=0, loDiff=3, upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    # cv2.floodFill(th2, mask=mask, seedPoint=(w-1,1), newVal=0, loDiff=3, upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    # cv2.floodFill(th2, mask=mask, seedPoint=(w-1,h-1), newVal=0, loDiff=3, upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = auto_Canny(th2)
    if show:
        cv2.imshow('th2', th2)
        cv2.imshow('threshold', t)
        cv2.imshow('t_edge', edge)
    log.debug(f'ret2 {ret2}')
    return


def get_edge(image):
    # edge->blur->dilate->erode->contours
    # img_equalize = cv2.equalizeHist(image)
    # clahe = cv2.createCLAHE()
    # sharped = clahe.apply(image)
    # blur1 = cv2.GaussianBlur(img_equalize, (3, 3), 0)
    # cv2.imshow('equalize', img_equalize)
    # cv2.imshow('blur1', blur1)
    # img_equalize = cv2.equalizeHist(bl)
    img_equalize = image
    # threshold(image)
    edge = auto_Canny(img_equalize)
    # blur edge, not original image
    blur = cv2.GaussianBlur(edge, (5, 5), 0)
    dilate = cv2.dilate(blur, None)
    erode_edge = cv2.erode(dilate, None)
    cv2.imshow('edge', edge)
    # plt.hist(img_equalize, 256)
    # plt.show()
    # cv2.imshow('dilate', dilate)
    return erode_edge


def revert(img):
    return 255 - img


def get_arc_epsilon(max_contour, ratio=0.0001):
    log.info(f'Max contour area: {cv2.contourArea(max_contour)}')
    arc_epsilon = cv2.arcLength(max_contour, True) * ratio
    log.info(f'Set arc epsilon: {arc_epsilon}')
    return arc_epsilon


def get_contour_value(img, cnt):
    # fill contour with (255,255,255)
    mask = np.zeros(img.shape[:2], dtype='uint8')
    cv2.fillPoly(mask, [cnt], (255, 255, 255))
    masked = cv2.bitwise_and(img, img, mask=mask)
    mean, std = cv2.meanStdDev(img, mask=mask)
    return mean[0][0], std[0][0]


def get_background_value(img, external_contours, level_cnt):
    # assume bacground value is greater than negative reference value
    mask = np.ones(img.shape[:2], dtype='uint8')
    for external in external_contours:
        cnt = level_cnt[external]
        cv2.fillPoly(mask, [cnt], (0, 0, 0))
    masked = cv2.bitwise_and(img, img, mask=mask)
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
    log.info(f'Whole image: Area {img.size}\t '
             f'Whole blue mean {cv2.meanStdDev(revert_b)}')
    log.info(
        f'Background masked: Area {bg_size}\t Blue mean {bg_blue_mean}+-std{bg_blue_std}')
    for big in big_external_contours:
        # [next, previous, child, parent, self]
        big_cnt = level_cnt[big]
        big_area = cv2.contourArea(big_cnt)
        self_index = big[4]
        related_inner = [i for i in inner_contours if i[3] == self_index]
        # cv2.imshow('Masked big', masked)
        big_blue_mean, big_blue_std = get_contour_value(revert_b, big_cnt)
        log.info(f'Big region: No.{big[-1]}\t '
                 f'Area: {big_area}\t Blue mean: {big_blue_mean}')
        for inner in related_inner:
            inner_cnt = level_cnt[inner]
            inner_cnt_area = cv2.contourArea(inner_cnt)
            inner_blue_mean, _ = get_contour_value(revert_b, inner_cnt)
            # inner_green_mean, _ = get_contour_value(revert_g, inner_cnt)
            # assert big_blue_mean > bg_blue_mean
            if inner_blue_mean < bg_blue_mean + bg_blue_std:
                log.debug(f'Real background region: No.{inner[-1]}\t '
                          f'Area: {inner_cnt_area}\t'
                          f'Blue mean: {inner_blue_mean}')
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
        log.error('Cannot find targets in the image.')
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
                # todo: show error in window
                log.error('Target and reference are overlapped!')
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


def draw_images(filtered_result, level_cnt, img):
    def drawing(levels, color):
        line_width = 2
        for level in levels:
            cnt = level_cnt[level]
            min_rect = cv2.minAreaRect(cnt)
            min_rect_points = np.int0(cv2.boxPoints(min_rect))
            x, y, w, h = cv2.boundingRect(cnt)
            approx = cv2.approxPolyDP(cnt, arc_epsilon, True)
            # b,g,r
            cv2.rectangle(img_dict['rectangle'], (x, y), (x + w, y + h),
                          color, line_width)
            cv2.drawContours(img_dict['min_area_rectangle'],
                             [min_rect_points], 0, color, line_width)
            cv2.polylines(img_dict['polyline'], [approx], True, color,
                          line_width)
            cv2.fillPoly(img_dict['fill'], [approx], color)

    (big_external_contours, small_external_contours, inner_contours,
     fake_inner, real_background) = filtered_result
    arc_epsilon = get_arc_epsilon(level_cnt[big_external_contours[0]])
    b, g, r = cv2.split(img)
    img_dict = dict()
    img_dict['raw'] = img
    img_dict['rectangle'] = img.copy()
    img_dict['min_area_rectangle'] = img.copy()
    img_dict['polyline'] = img.copy()
    img_dict['fill'] = img.copy()
    # img_dict['edge'] = edge
    # img_dict['revert_blue'] = 255 - b
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
        cv2.imshow(title, image)
    return img_dict


def get_real_blue(original_image, neg_ref_value, pos_ref_value):
    assert neg_ref_value <= pos_ref_value
    assert pos_ref_value > 0
    b, g, r = cv2.split(original_image)
    factor = 1
    revert_b = revert(b)
    amplified_neg_ref = int(factor * neg_ref_value)
    return revert_b, amplified_neg_ref


def get_real_blue_2(original_image, neg_ref_value, pos_ref_value):
    # todo, 255-b is not real blue part
    assert neg_ref_value <= pos_ref_value
    assert pos_ref_value > 0
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


def calculate(original_image, target_mask, neg_ref_value=32, pos_ref_value=255):
    """
    Calculate given region's value.
    Args:
        original_image: original BGR image
        target_mask: mask of target region
        pos_ref_value: positive reference value for up threshold
        neg_ref_value: negative reference value for down threshold
    Returns:
    """
    # todo: remove green
    # blue express area
    revert_b, amplified_neg_ref = get_real_blue(original_image, neg_ref_value, pos_ref_value)
    cv2.imshow('x', revert_b)
    cv2.waitKey()
    zero = np.zeros(original_image.shape[:2], dtype='uint8')

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
    assert x1 < x1 + w1 < x2 < x2 + w2
    middle = (x1 + w1 + x2) // 2
    target = img_copy[:, :middle]
    ref = img_copy[:, middle:]
    return target, ref


def get_contour(img_file):
    # .png .jpg .tiff
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


def draw(results, labels, out):
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
    fig, ax1 = plt.subplots(figsize=(10, 6))
    _ = results[0][-1]
    x = np.arange(1, len(labels) + 1)
    width = 0.2
    try:
        violin_parts = ax1.violinplot([i[-1] for i in results], showmeans=False,
                                      showmedians=False, showextrema=False,
                                      widths=0.4)
    except ValueError:
        log.error('Failed to plot results due to bad values.')
        raise SystemExit(-2)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#0d56ff')
        pc.set_edgecolor('black')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Expression value', color='b')
    ax1.set_yticks(np.linspace(0, 256, 9))
    short_labels = [Path(i).name for i in labels]
    ax1.set_xticks(np.arange(1, len(labels) + 1), labels=short_labels)
    ax2 = ax1.twinx()
    rects1 = ax2.bar(x - width / 2, [i[2] for i in results], width=width,
                     alpha=0.4,
                     color='green', label='Express area')
    rects2 = ax2.bar(x + width / 2, [i[5] for i in results], width=width,
                     alpha=0.4,
                     color='orange', label='Total area')
    ax2.bar_label(rects1, padding=3)
    ax2.bar_label(rects2, padding=3)
    ax2.legend()
    ax2.set_ylabel('Area')
    plt.tight_layout()
    # todo: not show in formal version
    plt.savefig(out)
    plt.show()
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


def main():
    arg = parse_arg()
    negative, positive, images = get_input(arg)
    log.info('Welcome to pyGUS.')
    log.info(f'Running mode {arg.mode}...')
    log.info(f'Negative reference image: {negative}')
    log.info(f'Positive reference image: {positive}')
    log.info('Target images:')
    for i in images:
        log.info(f'\t{i}')
    run_dict = {1: mode_1, 2: mode_2, 3: mode_3, 4: mode_4}
    run = run_dict[arg.mode]
    run(negative, positive, images)
    # todo: need rewrite
    return


def demo():
    input_file = get_input_demo()
    # .png .jpg .tiff
    filtered_result, level_cnt, img = get_contour(input_file)
    img_dict = draw_images(filtered_result, level_cnt, img)
    a, b, c, d = get_left_right_value(filtered_result, level_cnt, img)
    x = ['target', 'reference']
    print(a, b, c, d)
    # from matplotlib import pyplot as plt
    # plt.title(f'{input_file}:   {a / c:.2%}')
    # plt.bar(x, [a, c], width=0.5, color='#61a1cd')
    # plt.errorbar(x, [a, c], yerr=[b, d], fmt='+', ecolor='r', capsize=4)
    # plt.show()
    # use mask
    # target, ref = split_image(left_cnt, right_cnt, img)
    # show
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demo()