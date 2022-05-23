#!/usr/bin/python3.10
import cv2 as cv
import numpy as np
from pathlib import Path


def test():
    black = np.zeros((512, 512, 3), np.uint8)
    cv.line(black, (0, 0), (511, 511), (0, 100, 255), 3)
    cv.circle(black, (50, 50), 50, (50, 200, 50), 2)
    # negative for fill
    cv.circle(black, (150, 50), 50, (50, 200, 50), -1)
    # (-1, 1, 2) => (4, 1, 2)
    polygon = np.array([[10, 10], [200, 30], [300, 30], [400, 10]], np.int32).reshape((-1, 1, 2))
    # if polygon instead of [polygon], only draw points
    cv.polylines(black, [polygon], True, (255, 255, 255), 2)
    cv.putText(black, 'OpenCV', (10, 500), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.imshow('link', black)
    key = cv.waitKey(0)
    cv.destroyWindow('link')



import cv2 as cv
import numpy as np
from pathlib import Path
def show():
    cv.imshow('Raw', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
print(Path().cwd())
img_file = Path('example/example.png').absolute()
img = cv.imread(str(img_file))
print(img.shape)
print(img[0, 0])
# set value of first (0) channel
img.itemset((10, 10, 0), 50)
# set area
img[50:60, 50:60] = [128, 128, 128]
b, g, r = cv.split(img)
r2 = img[:, :, 2]
cv.imshow('Raw', img)
cv.imshow('b', b)
cv.imshow('g', g)
cv.imshow('r', r)
cv.waitKey(0)
cv.destroyAllWindows()
# show()
import cv2 as cv
import numpy as np
flags = [i for i in dir(cv) if i.startswith('COLOR_')]
# print(flags)
img = cv.imread('example/example.png')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# normal hsv [0,360], [0,100], [0,100]
# opencv hsv [0,180), [0,255], [0,255]
#https://blog.csdn.net/yu0046/article/details/112385007?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-112385007-blog-89305438.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-112385007-blog-89305438.pc_relevant_default&utm_relevant_index=2
# https://docs.opencv.org/4.5.5/df/d9d/tutorial_py_colorspaces.html
blue_lower = np.array([110, 50, 50])
blue_higher = np.array([130, 255, 255])
h, s, v = cv.split(hsv)
cv.imshow('Raw', img)
cv.imshow('h', h)
cv.imshow('s', s)
cv.imshow('v', v)
# blue is green in tabaco (blue+yellow=green)
mask = cv.inRange(hsv, blue_lower, blue_higher)
res = cv.bitwise_and(img, img, mask=mask)
cv.imshow('mask', mask)
cv.imshow('res', res)
cv.waitKey()
# color recalibrate
# https://github.com/18150167970/image_process_tool/blob/master/lighting_enhancement.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
example = 'example/example.png'
# example = 'example/0-1.tif'
img = cv2.imread(example)
print(img.shape)
black = cv2.inRange(img, (0, 0, 0), (0, 0, 0))
res = cv2.bitwise_and(img, img, mask=black)
img[res]=[255,255,255]
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
        cv2.rectangle(img2, (x,y), (x+w, y+h), (0,0,255), 1)
    else:
        cv2.rectangle(img3, (x,y), (x+w, y+h), (255,0,0), 1)
        cv2.circle(img4, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 1)
cv2.imshow('small area (<100)', img2)
cv2.imshow('rectangle', img3)
cv2.imshow('circle', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()
# color recalibrate
# https://github.com/18150167970/image_process_tool/blob/master/lighting_enhancement.py
import cv2
import numpy as np
from matplotlib import pyplot as plt
# example = 'example/example.png'
example = 'example/0-1.tif'
# example = 'example/75-2.tif'
img = cv2.imread(example)
print(img.shape)
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
# opencv use BGR
b, g, r = cv2.split(img)
for title, value in zip(['b', 'g', 'r'], [b, g, r]):
    continue
    cv2.imshow(title, value)
b = 255 - b
# reverse to get better edge
#erode = get_edge(img)
erode = get_edge(255-img)
contours, hierarchy = cv2.findContours(erode, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
max_contour = contours[0]
arc_epsilon = cv2.arcLength(max_contour, True) * 0.001
area_threshold = img.shape[0] * img.shape[1] * 0.001
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