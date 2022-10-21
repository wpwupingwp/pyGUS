#!/usr/bin/python3)
from pathlib import Path
import io
import os
import sys

import cv2
import numpy as np

from pyGUS import global_vars

log = global_vars.log


class Quit(SystemExit):
    def __init__(self, code):
        super().__init__(code)


def if_exist(filename: Path) -> str:
    """
    Check if given file exists.
    Check if given file is valid image file.
    """
    if not filename.exists():
        show_error(f'{filename} does not exist. Please check the input.')
    elif cv2.imread(str(filename)) is None:
        show_error(f'{filename} is not valid image file.')
    else:
        return str(filename)


def get_window_size() -> (int, int):
    from tkinter import Tk
    _ = Tk()
    _.withdraw()
    w = _.winfo_screenwidth()
    h = _.winfo_screenheight()
    _.destroy()
    return w, h


def get_ok_size_window(name: str, img_width: int, img_height: int) -> (
        str, bool):
    screen_width, screen_height = get_window_size()
    width, height = img_width, img_height
    resized = False
    if img_height > 0.9 * screen_height or img_width > 0.9 * screen_width:
        resized = True
        height = img_height * (screen_width / img_width)
        width = screen_width
        if height > screen_height:
            width = width * (screen_height / height)
            height = screen_height
        width, height = int(width * 0.9), int(height * 0.9)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width=width, height=height)
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    cv2.moveWindow(name, x, y)
    return name, resized


def imshow(name: str, img: np.array) -> bool:
    # show image with suitable window size
    # w,h vs h, w
    img_height, img_width = img.shape[:2]
    window, resized = get_ok_size_window(name, img_width, img_height)
    if resized:
        # width x height
        log.debug(f'Image is too big({img_width}*{img_height}), resize.')
    cv2.imshow(name, img)
    return resized


def show_error(msg: str) -> None:
    # show error message and quit
    if global_vars.is_gui:
        from tkinter import messagebox
        msg += '   Abort.'
        log.error(msg)
        messagebox.showerror(message=msg)
    else:
        log.error(msg)
    raise Quit(-10)


def hex2bgr(hex_str: str) -> (int, int, int):
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


def get_crop(img: np.array, r: list) -> np.array:
    # deprecated
    x, y, w, h = r
    cropped = img[int(y):int(y + h), int(x):int(x + w)]
    return cropped


def select_box(img: np.array, text='Drag to select, then press SPACE BAR',
               color=(255, 255, 255)) -> np.array:
    cv2.pollKey()
    hint = 'Drag to select, then press SPACE BAR'
    log.info(hint)
    img_height, img_width = img.shape[:2]
    window, resized = get_ok_size_window(text, img_width, img_height)
    # hide opencv stdout
    sys_stdout = sys.stdout.fileno()
    saved_sys_stdout = os.dup(sys_stdout)
    tmp = open(os.devnull, 'wb')
    sys.stdout.close()
    os.dup2(tmp.fileno(), sys_stdout)
    sys.stdout = io.TextIOWrapper(os.fdopen(tmp.fileno(), 'wb'))
    x, y, w, h = cv2.selectROI(text, img)
    sys.stdout.close()
    # stop hide
    sys.stdout = io.TextIOWrapper(os.fdopen(saved_sys_stdout, 'wb'))
    mask = np.zeros(img.shape[:2], dtype='uint8')
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    cv2.destroyWindow(text)
    return mask


def draw_dots(img: np.array) -> np.array:
    hint = 'Left click to add points, right click to finish, Esc to reset'
    name = hint
    log.info(hint)
    color = (255, 255, 255)
    width = int(img.shape[0] * 0.02)
    current = (0, 0)
    done = False
    img_raw = img.copy()
    points = list()

    def on_mouse(event, x, y, buttons, user_param):
        nonlocal done, current, points
        if done:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            done = True

    imshow(name, img)
    cv2.pollKey()
    cv2.setMouseCallback(name, on_mouse)
    while not done:
        if len(points) > 0:
            # cv2.polylines(img, np.array([points]), False, color, width)
            cv2.circle(img, points[-1], 2, color, width)
        cv2.imshow(name, img)
        # Esc
        if cv2.waitKey(5) == 27:
            points.clear()
            img = img_raw.copy()
    points_array = np.array([points])
    imshow(name, img)
    cv2.pollKey()
    cv2.destroyWindow(name)
    return img


def draw_box(img: np.array) -> np.array:
    """
    select background in cfm
    """
    # init
    # assert global_vars.is_gui
    img_raw = img.copy()
    color = (0, 0, 0)
    width = int(img.shape[0] * 0.02)
    hint = 'Left click to draw, right click to finish, Esc to reset'
    name = hint
    log.info(hint)
    done = False
    current = (0, 0)
    points = list()

    def on_mouse(event, x, y, buttons, user_param):
        nonlocal done, current, points, img
        if event == cv2.EVENT_MOUSEMOVE:
            current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            if points:
                points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            done = True
        if done:
            return

    imshow(name, img)
    cv2.pollKey()
    cv2.setMouseCallback(name, on_mouse)
    while not done:
        if len(points) == 1:
            img = img_raw.copy()
            cv2.rectangle(img, points[0], current, color=color, thickness=width)
        elif len(points) == 2:
            cv2.rectangle(img, points[0], points[1], color=color,
                          thickness=width)
        cv2.imshow(name, img)
        # Esc
        if cv2.waitKey(5) == 27:
            points.clear()
            img = img_raw.copy()
    imshow(name, img)
    cv2.pollKey()
    cv2.destroyWindow(name)
    return img


def select_polygon(img: np.array, title='', color=(255, 0, 255)) -> np.array:
    """
    Select polygon region.
    """
    # init
    # assert global_vars.is_gui
    name = title
    hint = 'Left click to add points, right click to finish, Esc to abort'
    log.info(hint)
    done = False
    current = (0, 0)
    points = list()
    mask = np.zeros(img.shape[:2], dtype='uint8')
    cropped = None
    box = None

    def on_mouse(event, x, y, buttons, user_param):
        nonlocal done, current, points
        if done:
            return
        if event == cv2.EVENT_MOUSEMOVE:
            current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            done = True

    imshow(name, img)
    cv2.pollKey()
    cv2.setMouseCallback(name, on_mouse)
    while not done:
        if len(points) > 0:
            cv2.polylines(img, np.array([points]), False, color, 3)
            cv2.circle(img, points[-1], 2, color, 3)
        imshow(name, img)
        # Esc
        if cv2.waitKey(50) == 27:
            cv2.destroyWindow(name)
            return None
    points_array = np.array([points])
    if len(points) > 0:
        cv2.fillPoly(img, points_array, color)
        cv2.fillPoly(mask, points_array, 255)
        # box = cv2.boundingRect(points_array)
    else:
        pass
    imshow(name, img)
    cv2.pollKey()
    cv2.destroyWindow(name)
    return mask


def draw_colorchecker(out='card.jpg') -> str:
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
    w_gap = w + gap
    image = np.zeros((4 * w + 5 * gap, 6 * w + 7 * gap, 3))
    for i in range(4):
        for j in range(6):
            x = i * 6 + j
            image[(gap + i * w_gap):(gap + i * w_gap + w),
            (gap + j * w_gap):(gap + j * w_gap + w), :] = rgb[x, :]
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out, image)
    return out


def resize(img: np.array, new_height: int, new_width: int) -> np.array:
    # keep original w/h ratio
    height, width = img.shape[:2]
    if width / height >= new_width / new_height:
        img_new = cv2.resize(img, (new_width, int(height * new_width / width)))
    else:
        img_new = cv2.resize(img,
                             (int(width * new_height / height), new_height))
    return img_new


def color_calibrate(img_file: str, draw_detected=False) -> str:
    """
    Use color card to calibrate colors
    Args:
        img_file: raw image filename
        draw_detected: draw detected colorchecker or not
    Returns:
        calibrated:
    """
    log.info(f'Calibrate {img_file}')
    img = cv2.imread(img_file)
    detector = cv2.mcc.CCheckerDetector_create()
    detector.process(img, cv2.mcc.MCC24)
    checker = detector.getBestColorChecker()
    if draw_detected:
        cdraw = cv2.mcc.CCheckerDraw_create(checker)
        img_draw = img.copy()
        cdraw.draw(img_draw)
        imshow('Detected colorchecker', img_draw)
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
    img_file_p = Path(img_file)
    # png is lossless compress
    out_img_file = img_file_p.parent / img_file_p.with_name(
        img_file_p.stem + '_calibrated.png')
    cv2.imwrite(str(out_img_file), out_img)
    log.debug(f'Calibrated image {out_img_file}')
    # cv2.imshow('original', img)
    # cv2.imshow('calibrated', out_img)
    return str(out_img_file)


def grab(image: np.array, mask: np.array) -> np.array:
    image = resize(image, 1000, 1000)
    rect = cv2.selectROI('', image)
    fg = np.zeros((1, 65), dtype="float")
    bg = np.zeros((1, 65), dtype="float")
    mask, bg, fg = cv2.grabCut(image, mask, rect, bg, fg, iterCount=2,
                               mode=cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    if global_vars.debug:
        imshow('grab', mask2)
        cv2.waitKey()
    return mask


if __name__ == '__main__':
    draw_colorchecker()