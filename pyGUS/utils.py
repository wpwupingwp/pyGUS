#!/usr/bin/python3)

import cv2
import numpy as np

from pyGUS.global_vars import log


def get_crop(img, r):
    x, y, w, h = r
    cropped = img[int(y):(y+h), int(x):int(x+w)]
    return cropped


def select_box(img, text='Select the region, then press ENTER'):
    r = cv2.selectROI(text, img)
    cropped = get_crop(img, r)
    return cropped, r


def select_polygon(raw_img, title='', color=(255, 255, 255)):
    """
    Select polygon region.
    Args:
        raw_img:
        title:
        color:
    Returns:
        cropped:
        points_array:
    """
    # init
    name = f'{title} (Left click to add points, right click to finish, Esc to quit)'
    img = raw_img.copy()
    done = False
    current = (0, 0)
    points = list()
    mask = np.zeros(img.shape[:2], dtype='uint8')
    cropped = None
    box = None

    def on_mouse(event, x, y, buttons, user_param):
        # print(event, x, y)
        nonlocal done, current, points
        if done:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            done = True

    cv2.imshow(name, img)
    cv2.pollKey()
    cv2.setMouseCallback(name, on_mouse)
    while not done:
        if len(points) > 0:
            cv2.polylines(img, np.array([points]), False, color, 3)
            cv2.circle(img, points[-1], 2, color, 3)
        cv2.imshow(name, img)
        # Esc
        if cv2.waitKey(50) == 27:
            done = True
    points_array = np.array([points])
    if len(points) > 0:
        cv2.fillPoly(img, points_array, color)
        cv2.fillPoly(mask, points_array, (255, 255, 255))
        box = cv2.boundingRect(points_array)
        cropped = get_crop(img, box)
    else:
        pass
    cv2.imshow(name, img)
    cv2.pollKey()
    cv2.destroyWindow(name)
    return cropped, points_array


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
    # region, r = select_box(image)
    cropped, _ = select_polygon(image, 'test')
    cv2.imshow('selected', cropped)
    # cv2.imshow('selected', region)
    cv2.imwrite(out, image)
    cv2.waitKey(0)
    # print(image.shape)
    return out


def resize(img, new_height, new_width):
    # keep original w/h ratio
    height, width = img.shape[:2]
    if width / height >= new_width / new_height:
        img_new = cv2.resize(img, (new_width, int(height*new_width/width)))
    else:
        img_new = cv2.resize(img, (int(width*new_height/height), new_height))
    return img_new



def color_calibrate(img):
    """
    Use color card to calibrate colors
    Args:
        img:
    Returns:
        calibrated:
    """
    detector = cv2.mcc.CCheckerDetector_create()
    detector.process(img, cv2.mcc.MCC24)
    checker = detector.getBestColorChecker()
    # get ccm
    charts_rgb = checker.getChartsRGB()
    src = charts_rgb[:, 1].copy().reshape(24, 1, 3)
    src /= 255
    # generate model
    model = cv2.ccm_ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)
    # model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
    model.setCCM_TYPE(cv2.ccm.CCM_3x3)
    model.setDistance(cv2.ccm.DISTANCE_CIE2000)
    model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
    model.setLinearGamma(2.2)
    model.setLinearDegree(3)
    model.setSaturatedThreshold(0, 0.98)
    model.run()
    # ccm = model.getCCM()
    loss = model.getLoss()
    # print('ccm', ccm)
    log.debug(f'Color calibration loss {loss}')
    # calibrate
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = img2.astype(np.float64)
    img2 /= 255.0
    calibrated = model.infer(img2)
    out = calibrated * 255
    out[out < 0] = 0
    out[out > 255] = 255
    out = out.astype(np.uint8)
    out_img = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out_img


if __name__ == '__main__':
    draw_colorchecker()