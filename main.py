#!/usr/bin/python3.10
import cv2
import logging
import numpy as np
from pathlib import Path
from itertools import chain

# from matplotlib import pyplot as plt
# define logger
FMT = '%(asctime)s %(levelname)-8s %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
default_level = logging.INFO
import coloredlogs

coloredlogs.install(level=default_level, fmt=FMT, datefmt=DATEFMT)
log = logging.getLogger('pyGUS')


def test_1():
    black = np.zeros((512, 512, 3), np.uint8)
    cv2.line(black, (0, 0), (511, 511), (0, 100, 255), 3)
    cv2.circle(black, (50, 50), 50, (50, 200, 50), 2)
    # negative for fill
    cv2.circle(black, (150, 50), 50, (50, 200, 50), -1)
    # (-1, 1, 2) => (4, 1, 2)
    polygon = np.array([[10, 10], [200, 30], [300, 30], [400, 10]], np.int32).reshape((-1, 1, 2))
    # if polygon instead of [polygon], only draw points
    cv2.polylines(black, [polygon], True, (255, 255, 255), 2)
    cv2.putText(black, 'OpenCV', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('link', black)
    cv2.waitKey(0)
    cv2.destroyWindow('link')


def test_2():
    def show():
        cv2.imshow('Raw', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    img_file = Path('example/example.png').absolute()
    img = cv2.imread(str(img_file))
    # set value of first (0) channel
    img.itemset((10, 10, 0), 50)
    # set area
    img[50:60, 50:60] = [128, 128, 128]
    b, g, r = cv2.split(img)
    r2 = img[:, :, 2]
    cv2.imshow('Raw', img)
    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_3():
    flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    # print(flags)
    img = cv2.imread('example/example.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # normal hsv [0,360], [0,100], [0,100]
    # opencv hsv [0,180), [0,255], [0,255]
    # https://blog.csdn.net/yu0046/article/details/112385007?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-112385007-blog-89305438.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-112385007-blog-89305438.pc_relevant_default&utm_relevant_index=2
    # https://docs.opencv2.org/4.5.5/df/d9d/tutorial_py_colorspaces.html
    blue_lower = np.array([110, 50, 50])
    blue_higher = np.array([130, 255, 255])
    h, s, v = cv2.split(hsv)
    cv2.imshow('Raw', img)
    cv2.imshow('h', h)
    cv2.imshow('s', s)
    cv2.imshow('v', v)
    # blue is green in tabaco (blue+yellow=green)
    mask = cv2.inRange(hsv, blue_lower, blue_higher)
    res = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.waitKey()
    # color recalibrate
    # https://github.com/18150167970/image_process_tool/blob/master/lighting_enhancement.py


def test_4():
    example = 'example/example.png'
    # example = 'example/0-1.tif'
    img = cv2.imread(example)
    black = cv2.inRange(img, (0, 0, 0), (0, 0, 0))
    res = cv2.bitwise_and(img, img, mask=black)
    img[res] = [255, 255, 255]

    def auto_Canny(image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edge = cv2.Canny(image, lower, upper)
        return edge

    # opencv use BGR
    b, g, r = cv2.split(img)
    cv2.imshow('raw', img)
    for title, value in zip(['b', 'g', 'r'], [b, g, r]):
        cv2.imshow(title, value)
    edge = auto_Canny(img)
    dilate = cv2.dilate(edge, None)
    erode = cv2.erode(dilate, None)
    cv2.imshow('edge', edge)
    cv2.imshow('dilate', dilate)
    cv2.imshow('erode', erode)
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img2 = img.copy()
    img3 = img.copy()
    img4 = img.copy()
    area_threshold = img.shape[0] * img.shape[1] * 0.0001
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        center, radius = cv2.minEnclosingCircle(cnt)
        area = cv2.contourArea(cnt)
        if area < area_threshold:
            # b,g,r
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 1)
        else:
            cv2.rectangle(img3, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.circle(img4, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 1)
    cv2.imshow('small area (<100)', img2)
    cv2.imshow('rectangle', img3)
    cv2.imshow('circle', img4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # color recalibrate
    # https://github.com/18150167970/image_process_tool/blob/master/lighting_enhancement.py


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


def split_channel(img):
    # show rgb channel
    # opencv use BGR
    b, g, r = cv2.split(img)
    for title, value in zip(['b', 'g', 'r'], [b, g, r]):
        cv2.imshow(title, 255 - value)


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
    if len(big_external_contours) == 0:
        log.error('Cannot detect objects in the image.')
        raise SystemExit(-1)
    try:
        small_external_contours = external_contours[2:]
    except IndexError:
        small_external_contours = list()
    fake_inner, real_background = remove_fake_inner_cnt(
        img, level_cnt, big_external_contours, external_contours, inner_contours)
    return (big_external_contours, small_external_contours, inner_contours,
            fake_inner, real_background)


def revert(img):
    return 255 - img


def get_arc_epsilon(max_contour, ratio=0.0001):
    log.info(f'Max contour area: {cv2.contourArea(max_contour)}')
    arc_epsilon = cv2.arcLength(max_contour, True) * ratio
    log.info(f'Set arc epsilon: {arc_epsilon}')
    return arc_epsilon


def drawing(levels, level_cnt, arc_epsilon, img_dict, color):
    line_width = 1
    for level in levels:
        cnt = level_cnt[level]
        rect = cv2.minAreaRect(cnt)
        rect_2 = np.int0(cv2.boxPoints(rect))
        approx = cv2.approxPolyDP(cnt, arc_epsilon, True)
        # b,g,r
        cv2.drawContours(img_dict['rectangle'], [rect_2], 0, color, line_width)
        cv2.polylines(img_dict['polyline'], [approx], True, color, line_width)
        cv2.fillPoly(img_dict['fill'], [approx], color)
    return img_dict


def split_region(img):
    left = None
    right = None
    return left, right


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
    print(mean)
    cv2.imshow('Background masked', masked)
    return mean[0][0], std[0][0]


def remove_fake_inner_cnt(img, level_cnt, big_external_contours,
                          external_contours, inner_contours):
    fake_inner = list()
    real_background = list()
    b, g, r = cv2.split(img)
    revert_b = revert(b)
    revert_g = revert(g)
    # todo: use green channle to detect real background
    cv2.imshow('Revert g', revert_g)
    # background blue mean
    bg_blue_mean, bg_blue_std = get_background_value(revert_b, external_contours, level_cnt)
    bg_green_mean, bg_green_std = get_background_value(revert_g, external_contours, level_cnt)
    bg_size = img.size
    log.info(f'Whole image: Area {img.size}\t '
             f'Whole blue mean {cv2.meanStdDev(revert_b)}')
    log.info(f'Background masked: Area {bg_size}\t Blue mean {bg_blue_mean}+-std{bg_blue_std}\tGreen mean {bg_green_mean}+-std{bg_green_std}')
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
            elif inner_blue_mean < bg_blue_mean+bg_blue_std:
                # todo: detect real background inner contour
                real_background.append(inner)
            elif inner_green_mean < bg_green_mean+bg_green_std:
                real_background.append(inner)
            if inner_cnt_area > 500:
                log.info(f'Inner region: No.{inner[-1]}\t Area: {inner_cnt_area}\tBlue mean: {inner_blue_mean}\tGreen mean: {inner_green_mean}')
                print(_)
    return fake_inner, real_background


def hex2bgr(hex: str):
    hex2 = hex.removeprefix('#')
    r = int('0x'+hex2[0:2].lower(), base=16)
    g = int('0x'+hex2[2:4].lower(), base=16)
    b = int('0x'+hex2[4:6].lower(), base=16)
    return b, g, r


def main():
    input_file = get_input()
    # .png .jpg .tiff
    img = cv2.imread(input_file)
    # split_channel(img)
    # todo: use blue revert or whole revert
    b, g, r = cv2.split(img)
    # reverse to get better edge
    # revert_img = revert(img)
    revert_img = revert(g)
    edge = get_edge(revert_img)
    # APPROX_NONE to avoid omitting dots
    contours, raw_hierarchy = cv2.findContours(edge, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
    hierarchy = list()
    # raw hierarchy is [[[1,1,1,1]]]
    for index, value in enumerate(raw_hierarchy[0]):
        # [next, previous, child, parent, self]
        new_value = np.array([*value, index], dtype=value.dtype)
        hierarchy.append(new_value)
    level_cnt = dict()
    for key, value in zip(hierarchy, contours):
        level_cnt[tuple(key)] = value
    (big_external_contours, small_external_contours, inner_contours,
     fake_inner, real_background) = filter_contours(img, level_cnt)
    # use mask
    # todo: split image to left and right according to boundingrect of external contours
    left, right = split_region(img)
    # todo: use histogram
    # todo:calculate blue values, then divide by blue region and total region
    # show
    arc_epsilon = get_arc_epsilon(level_cnt[big_external_contours[0]])
    img_dict = dict()
    img_dict['raw'] = img
    img_dict['rectangle'] = img.copy()
    img_dict['polyline'] = img.copy()
    img_dict['fill'] = img.copy()
    img_dict['edge'] = edge
    img_dict['revert_blue'] = 255 - b
    color_blue = hex2bgr('#4d96ff')
    color_green = hex2bgr('#6bcb77')
    color_red = hex2bgr('#ff6b6b')
    color_yellow = hex2bgr('#ffd93d')
    drawing(big_external_contours, level_cnt, arc_epsilon, img_dict, color_blue)
    drawing(small_external_contours, level_cnt, arc_epsilon, img_dict, color_red)
    drawing(real_background, level_cnt, arc_epsilon, img_dict, color_red)
    drawing(inner_contours, level_cnt, arc_epsilon, img_dict, color_yellow)
    drawing(fake_inner, level_cnt, arc_epsilon, img_dict, color_green)
    for title, image in img_dict.items():
        cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
