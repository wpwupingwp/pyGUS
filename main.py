#!/usr/bin/python3.10
import coloredlogs
import cv2
import logging
import numpy as np
from pathlib import Path
# todo: color correction
# todo mode 1: single object for each image, select positive, negative, targets
# todo mode 2: two object for each image, left target, right color card
# todo mode 3: two object for each image, left target, right positive/negative
# todo: select area by mouse
# todo: calculate values, statistic analysis
# todo: GUI
# todo: manual
# todo: manuscript

# from matplotlib import pyplot as plt
# define logger
FMT = '%(asctime)s %(levelname)-8s %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
default_level = logging.DEBUG

coloredlogs.install(level=default_level, fmt=FMT, datefmt=DATEFMT)
log = logging.getLogger('pyGUS')


def get_input(input_file='example/ninanjie-ok-75-2.tif'):
    # input_path = 'example/example.png'
    # input_file = 'example/ninanjie-0-1.tif'
    # input_file = 'example/ninanjie-75-2.tif'
    input_file = 'example/ninanjie-50-1.tif'
    # input_file = 'example/ninanjie-ok-75-2.tif'
    # input_file = 'example/ninanjie-100-1.tif'
    # input_file = 'example/ninanjie-2h-3.tif'
    # input_file = 'example/ersuiduanbingcao-BD15-27.png'
    # input_file = 'example/ersuiduanbingcao-DH39-5.png'
    # input_file = 'example/ersuiduanbingcao-ok-BD7-4.png'
    img = cv2.imread(input_file)
    log.info(f'Image size: {img.shape}')
    print(img.shape)
    return input_file


def auto_Canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # use edited lower bound
    lower = int(max(0, (1.0 - sigma*3) * v))
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
    mask = np.zeros([h+2, w+2], np.uint8)
    # ret1, th1 = cv2.threshold(img, 16, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    equalize = cv2.equalizeHist(blur)
    r, t = cv2.threshold(equalize, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    edge = auto_Canny(255-th2)
    # cv2.floodFill(th2, mask=mask, seedPoint=(1,1), newVal=0, loDiff=3, upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    print(h,w,sep='x')
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
    print('ret2', ret2)
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
    bg_blue_mean, bg_blue_std = get_background_value(revert_b, external_contours, level_cnt)
    bg_size = img.size
    log.info(f'Whole image: Area {img.size}\t '
             f'Whole blue mean {cv2.meanStdDev(revert_b)}')
    log.info(f'Background masked: Area {bg_size}\t Blue mean {bg_blue_mean}+-std{bg_blue_std}')
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
            if inner_blue_mean < bg_blue_mean+bg_blue_std:
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
    fake_inner, inner_background = remove_fake_inner_cnt(
        img, level_cnt, big_external_contours, external_contours, inner_contours)
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
                log.error('Target and reference are overlapped!')
                left = right = None
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


def calculate(inner_contours, fake_inner, inner_background, target, ref,
              level_cnt, img):
    # todo
    result = None
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
    assert x1 < x1+w1 < x2 < x2+w2
    middle = (x1+w1+x2) // 2
    target = img_copy[:, :middle]
    ref = img_copy[:, middle:]
    return target, ref


def main():
    input_file = get_input()
    # .png .jpg .tiff
    img = cv2.imread(input_file)
    # show_channel(img)
    b, g, r = cv2.split(img)
    # reverse to get better edge
    # use green channel
    # todo: g or b
    revert_img = revert(g//2+r//2)
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
    (big_external_contours, small_external_contours, inner_contours,
     fake_inner, inner_background) = filtered_result
    img_dict = draw_images(filtered_result, level_cnt, img)
    # use mask
    left, right = get_left_right(big_external_contours, level_cnt)
    left_cnt = level_cnt[left]
    right_cnt = level_cnt[right]
    target, ref = split_image(left_cnt, right_cnt, img)
    # threshold(target, show=True)
    # threshold(ref)
    result = calculate(inner_contours, fake_inner, inner_background, target, ref, level_cnt, img)
    # todo: use histogram
    # todo:calculate blue values, then divide by blue region and total region
    # show
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
