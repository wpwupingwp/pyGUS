#!/usr/bin/python3)

import cv2
import numpy as np

def draw_colorchecker(out='card.jpg'):
    # https://github.com/opencv/opencv_contrib/blob/4.x/modules/mcc/src/dictionary.hpp
    # default size : 2080x1400
    colors = (
        ('Dark Skin', (115, 82, 68)),
        ('Light Skin', (194, 150, 130)),
        ('Blue Sky', (98, 122, 157)),
        ('Foliage', (87, 108, 67)),
        ('Blue Flower', (133, 128, 177)),
        ('Bluish Green', (103, 189, 170)),
        ('Orange', (214, 126, 44)),
        ('Purplish Blue', (80, 91, 166)),
        ('Moderate Red', (193, 90, 99)),
        ('Purple', (94, 60, 108)),
        ('Yellow Green', (157, 188, 64)),
        ('Orange Yellow', (224, 163, 46)),
        ('Blue', (56, 61, 150)),
        ('Green', (70, 148, 73)),
        ('Red', (175, 54, 60)),
        ('Yellow', (231, 199, 31)),
        ('Magenta', (187, 86, 149)),
        ('Cyan', (8, 133, 161)),
        ('White (.05)*', (243, 243, 242)),
        ('Neutral 8 (.23) *', (200, 200, 200)),
        ('Neutral6.5 (.44) *', (160, 160, 160)),
        ('Neutral 5 (.70) *', (122, 122, 121)),
        ('Neutral3.5 (1.05) *', (85, 85, 85)),
        ('Black (1.5) *', (52, 52, 52)))
    rgb = np.array([i[1] for i in colors])
    factor = 2
    w = 150 * factor
    gap = 20 * factor
    w_and_gap = w + gap
    image = np.zeros((4*w+5*gap, 6*w+7*gap, 3))
    for i in range(4):
        for j in range(6):
            image[(gap+i*w_and_gap):(gap+i*w_and_gap+w), (gap+j*w_and_gap):(gap+j*w_and_gap+w), :] = rgb[(i*6+j), :]
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('a', image)
    cv2.imwrite(out, image)
    cv2.waitKey(0)
    print(image.shape)
    return out

def resize(img, new_height, new_width):
    # keep original w/h ratio
    height, width = img.shape[:2]
    if width / height >= new_width / new_height:
        img_new = cv2.resize(img, (new_width, int(height*new_width/width)))
    else:
        img_new = cv2.resize(img, (int(width*new_height/height), new_height))
    return img_new

a = draw_colorchecker()