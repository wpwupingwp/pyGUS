#!/usr/bin/python3

from subprocess import run
from platform import system
from pyGUS import core, utils
import cv2
from pyGUS import core

if system() == 'Windows':
    python = 'python'
else:
    python = 'python3'


def demo_mode_3():
    cmd = (f'{python} -m pyGUS -mode 3 -images example/color/DSC_8081.JPG -ref1'
           f' example/color/DSC_8081.JPG -ref2 example/color/DSC_8081.JPG')
    run(cmd, shell=True)
    return


def demo_old():
    input_file = core.get_input_demo()
    filtered_result, level_cnt, img = core.get_contour(input_file)
    img_dict = core.draw_images(filtered_result, level_cnt, img, simple=False,
                                show=True)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def demo_mode_4():
    cmd = (f'{python} -m pyGUS -mode 4 -images example/ninanjie-100-1.tif '
           f'example/ninanjie-100-2.tif')
    run(cmd, shell=True)
    return


def demo_show_colorchecker():
    img_file = utils.draw_colorchecker()
    img = cv2.imread(img_file)
    cv2.imshow('colorchecker', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def demo_show_detected_checker():
    utils.color_calibrate('s2.png', draw_detected=True)
    cv2.waitKey()
    cv2.destroyAllWindows()
    utils.color_calibrate('card2.jpeg', draw_detected=True)
    cv2.waitKey()
    cv2.destroyAllWindows()


demo_show_detected_checker()
demo_mode_3()
demo_mode_4()
demo_old()
demo_show_colorchecker()