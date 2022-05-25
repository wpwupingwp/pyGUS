#!/usr/bin/python3.10
import coloredlogs
import cv2
import logging
import numpy as np
from pathlib import Path

# from matplotlib import pyplot as plt
# define logger
FMT = '%(asctime)s %(levelname)-8s %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
default_level = logging.DEBUG

coloredlogs.install(level=default_level, fmt=FMT, datefmt=DATEFMT)
log = logging.getLogger('pyGUS')


def get_input(input_file='example/0-1.tif'):
    # input_path = 'example/example.png'
    input_file = 'example/0-1.tif'
    # input_file = 'example/75-2.tif'
    img = cv2.imread(input_file)
    log.info(f'Image size: {img.shape}')
    print(img.shape)
    return input_file


def auto_Canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edge = cv2.Canny(image, lower, upper)
    return edge


def get_edge(image):
    # edge->blur->dilate->erode->contours
    edge = auto_Canny(image)
    blur = cv2.GaussianBlur(edge, (3, 3), 0)
    dilate = cv2.dilate(blur, None)
    erode_edge = cv2.erode(dilate, None)
    # cv2.imshow('edge', edge)
    # cv2.imshow('dilate', dilate)
    # cv2.imshow('blur', blur)
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
    mask = np.ones(img.shape[:2], dtype='uint8')
    for external in external_contours:
        cnt = level_cnt[external]
        cv2.fillPoly(mask, [cnt], (0, 0, 0))
    masked = cv2.bitwise_and(img, img, mask=mask)
    mean, std = cv2.meanStdDev(img, mask=mask)
    cv2.imshow('Background masked', masked)
    return mean[0][0], std[0][0]


def remove_fake_inner_cnt(img, level_cnt, big_external_contours,
                          external_contours, inner_contours):
    fake_inner = list()
    real_background = list()
    b, g, r = cv2.split(img)
    revert_b = revert(b)
    #  use green channel to detect real background
    revert_g = revert(g)
    # background blue mean
    bg_blue_mean, bg_blue_std = get_background_value(revert_b, external_contours, level_cnt)
    bg_green_mean, bg_green_std = get_background_value(revert_g, external_contours, level_cnt)
    bg_size = img.size
    log.info(f'Whole image: Area {img.size}\t '
             f'Whole blue mean {cv2.meanStdDev(revert_b)}')
    log.debug(f'Background masked: Area {bg_size}\t Blue mean {bg_blue_mean}+-std{bg_blue_std}\tGreen mean {bg_green_mean}+-std{bg_green_std}')
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
            inner_green_mean, _ = get_contour_value(revert_g, inner_cnt)
            if inner_blue_mean >= big_blue_mean:
                fake_inner.append(inner)
            if inner_blue_mean < bg_blue_mean+bg_blue_std:
                log.info(f'Inner region: No.{inner[-1]}\t Area: {inner_cnt_area}\tBlue mean: {inner_blue_mean}\tGreen mean: {inner_green_mean}')
                real_background.append(inner)
    return fake_inner, real_background


def filter_contours(img, level_cnt: dict) -> (list, list, list):
    """
    Args:
        img: original image
        level_cnt(dict):
    Returns:
        big:
        small:
        inner
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
    fake_inner, real_background = remove_fake_inner_cnt(
        img, level_cnt, big_external_contours, external_contours, inner_contours)
    return (big_external_contours, small_external_contours, inner_contours,
            fake_inner, real_background)


def get_left_right(big_external_contours):
    if len(big_external_contours) == 0:
        log.error('Cannot find targets in the image.')
        raise SystemExit(-1)
    left = None
    right = None
    return left, right


def show_channel(img):
    # show rgb channel
    # opencv use BGR
    b, g, r = cv2.split(img)
    for title, value in zip(['b', 'g', 'r'], [b, g, r]):
        cv2.imshow(title+'revert', 255 - value)
        cv2.imshow(title, value)


def hex2bgr(hex_str: str):
    """
    Args:
        hex_str: #FFFFFF
    Returns:
        255, 255, 255
    """
    hex2 = hex_str.removeprefix('#')
    r = int('0x'+hex2[0:2].lower(), base=16)
    g = int('0x'+hex2[2:4].lower(), base=16)
    b = int('0x'+hex2[4:6].lower(), base=16)
    return b, g, r


def drawing(levels, level_cnt, arc_epsilon, img_dict, color):
    line_width = 1
    for level in levels:
        cnt = level_cnt[level]
        min_rect = cv2.minAreaRect(cnt)
        min_rect_points = np.int0(cv2.boxPoints(min_rect))
        x, y, w, h = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(cnt, arc_epsilon, True)
        # b,g,r
        cv2.rectangle(img_dict['rectangle'], (x, y), (x+w, y+h), color, line_width)
        cv2.drawContours(img_dict['min_area_rectangle'], [min_rect_points], 0,
                         color, line_width)
        cv2.polylines(img_dict['polyline'], [approx], True, color, line_width)
        cv2.fillPoly(img_dict['fill'], [approx], color)
    return img_dict


def main():
    input_file = get_input()
    # .png .jpg .tiff
    img = cv2.imread(input_file)
    # split_channel(img)
    b, g, r = cv2.split(img)
    # reverse to get better edge
    # use green channel
    revert_img = revert(g)
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
    (big_external_contours, small_external_contours, inner_contours,
     fake_inner, real_background) = filter_contours(img, level_cnt)
    # use mask
    # todo: split image to left and right according to boundingrect of external contours
    left, right = get_left_right(big_external_contours)
    # todo: use histogram
    # todo:calculate blue values, then divide by blue region and total region
    # show
    arc_epsilon = get_arc_epsilon(level_cnt[big_external_contours[0]])
    img_dict = dict()
    img_dict['raw'] = img
    img_dict['rectangle'] = img.copy()
    img_dict['min_area_rectangle'] = img.copy()
    img_dict['polyline'] = img.copy()
    img_dict['fill'] = img.copy()
    # img_dict['edge'] = edge
    img_dict['revert_blue'] = 255 - b
    color_blue = hex2bgr('#4d96ff')
    color_green = hex2bgr('#6bcb77')
    color_red = hex2bgr('#ff6b6b')
    color_yellow = hex2bgr('#ffd93d')
    drawing(big_external_contours, level_cnt, arc_epsilon, img_dict, color_blue)
    drawing(small_external_contours, level_cnt, arc_epsilon, img_dict, color_red)
    drawing(inner_contours, level_cnt, arc_epsilon, img_dict, color_yellow)
    drawing(fake_inner, level_cnt, arc_epsilon, img_dict, color_green)
    drawing(real_background, level_cnt, arc_epsilon, img_dict, color_red)
    for title, image in img_dict.items():
        cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
