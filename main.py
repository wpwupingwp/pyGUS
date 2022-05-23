#!/usr/bin/python3.10
import cv2
import logging
import numpy as np
from pathlib import Path
# from matplotlib import pyplot as plt
# define logger
FMT = '%(asctime)s %(levelname)-8s %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
default_level = logging.INFO
import coloredlogs
coloredlogs.install(level=default_level, fmt=FMT, datefmt=DATEFMT)


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
    key = cv2.waitKey(0)
    cv2.destroyWindow('link')


def test_2():
    def show():
        cv2.imshow('Raw', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(Path().cwd())
    img_file = Path('example/example.png').absolute()
    img = cv2.imread(str(img_file))
    print(img.shape)
    print(img[0, 0])
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
    flags = [i for i in dir(cv) if i.startswith('COLOR_')]
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
    print(img.shape)
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
    # cv2.RETR_EXTERNAL for biggest?
    # todo: use polygon instead of retangle
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
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


def get_input(input_path='example/0-1.tif'):
    # example = 'example/example.png'
    example = 'example/0-1.tif'
    # example = 'example/75-2.tif'
    img = cv2.imread(example)
    print(img.shape)
    return input_path


def auto_Canny(image, sigma=0.23):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edge = cv2.Canny(image, lower, upper)
    return edge


def get_edge(image):
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
        cv2.imshow(title, value)


def filter_contours(contours, hierarchy):
    """
    Args:
        contours:
        hierarchy:
    Returns: big, small, inner
    """
    # hierarchy is [[[1,1,1,1]]]
    external_contours = list()
    inner_contours = list()
    for cnt, level in zip(contours, hierarchy[0]):
        area = cv2.contourArea(cnt)
        next_, previous_, first_child, parent = level
        # -1 means no parent -> external
        if parent == -1:
            external_contours.append((cnt, area, level))
        else:
            inner_contours.append((cnt, area, level))
    external_contours.sort(key=lambda x: x[1], reverse=True)
    # a picture only contains at most TWO target (sample and reference)
    big_external_contours = external_contours[:2]
    try:
        small_external_contours = external_contours[2:]
    except IndexError:
        small_external_contours = list()
    return big_external_contours, small_external_contours, inner_contours


def revert(img):
    return 255 - img


def get_arc_epsilon(max_contour, ratio=0.0001):
    arc_epsilon = cv2.arcLength(max_contour, True) * ratio
    return arc_epsilon


def main():
    input_file = get_input()
    # .png .jpg .tiff
    img = cv2.imread(input_file)
    # reverse to get better edge
    revert_img = revert(img)
    erode = get_edge(revert_img)
    # APPROX_NONE to avoid omitting dots
    contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    (big_external_contours, small_external_contours,
     inner_contours) = filter_contours(contours, hierarchy)
    img2 = img.copy()
    img3 = img.copy()
    img4 = img.copy()
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        area = cv2.contourArea(cnt)
        rect_2 = np.int0(cv2.boxPoints(rect))
        approx = cv2.approxPolyDP(cnt, arc_epsilon, True)
        if area < area_threshold:
            # b,g,r
            cv2.drawContours(img2, [rect_2], 0, (0, 0, 255), 1)
            cv2.polylines(img3, [approx], True, (0, 0, 255), 1)
        else:
            cv2.drawContours(img2, [rect_2], 0, (255, 255, 0), 1)
            cv2.polylines(img3, [approx], True, (255, 255, 0), 1)
            # cv2.fillPoly(img5, [approx], (255, 255, 0))
            # cv2.drawContours(img4, approx, -1, (255, 255, 0),)
    # todo: 第一层级只取最大的两个（假设只摆放两个物体）
    # 根据内外关系排除空洞
    # 参照选择EXTERNAL
    cv2.fillPoly(img4, contours, (255, 255, 0))
    cv2.imshow('raw', img)
    cv2.imshow('erode', erode)
    cv2.imshow('rectangle (red for small area)', img2)
    cv2.imshow('polyline', img3)
    cv2.imshow('fill', img4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()