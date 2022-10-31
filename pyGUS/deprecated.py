import cv2
import numpy as np


from pyGUS.core import make_clean, imshow, show_error, revert
from pyGUS.core import auto_Canny
from pyGUS.global_vars import debug, log


def test(gray):
    # get edge, fail
    equal = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(equal, (15, 15), 0)
    threshold3, blur_bin = cv2.threshold(blur, 0, 255,
                                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bin_edge = cv2.Canny(blur_bin, 0, 255)
    bin_edge_ = make_clean(bin_edge)
    fill = fill_boundary(blur_bin)

    s_cnt, _ = cv2.findContours(bin_edge_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    s_img = np.zeros(gray.shape[:2])
    for i in s_cnt:
        cv2.drawContours(s_img, [i], 0, (255, 255, 255), 2)
    imshow('equal', equal)
    imshow('blur', blur)
    imshow('blur bin', blur_bin)
    imshow('blur bin_edge', bin_edge_)
    imshow('blur cnt', s_img)
    imshow('fill', fill)
    cv2.waitKey()
    return


def fill_boundary(img) -> np.array:
    img_copy = img.copy()
    # is it ok?
    ratio = 0.05
    height, width = img.shape[:2]
    _loc = int(height * ratio)
    log.debug(f'fill boundary, use width {_loc}')
    mask = np.zeros((height + 2, width + 2), dtype='uint8')
    w_middle = width // 2
    h_middle = height // 2
    mask[int(h_middle - height * ratio):int(h_middle + height * ratio),
    int(w_middle - width * ratio):int(w_middle + width * ratio)] = 1
    cv2.floodFill(img_copy, mask, (_loc, _loc), 255, 0, 0,
                  cv2.FLOODFILL_FIXED_RANGE)
    cv2.floodFill(img_copy, mask, (width - _loc, height - _loc), 255,
                  0, 0, cv2.FLOODFILL_FIXED_RANGE)
    cv2.floodFill(img_copy, mask, (_loc, height - _loc), (255, 255, 255), 0, 0,
                  cv2.FLOODFILL_FIXED_RANGE)
    cv2.floodFill(img_copy, mask, (width - _loc, _loc), (255, 255, 255), 0, 0,
                  cv2.FLOODFILL_FIXED_RANGE)
    return img_copy


def get_edge2(image: np.array) -> np.array:
    b, g, r = cv2.split(image)
    combine = revert(g // 2 + r // 2)
    sobel_add = get_sobel(image)
    s2_clean = make_clean(sobel_add)
    # s_equal = cv2.equalizeHist(s_erode)
    imshow('sobel2', sobel_add)
    imshow('s2_clean', s2_clean)
    scharr_x = cv2.Scharr(combine, cv2.CV_8U, 1, 0)
    scharr_y = cv2.Scharr(combine, cv2.CV_8U, 1, 0)
    scharr = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)
    sc_equal = cv2.equalizeHist(scharr)
    s_cnt, _ = cv2.findContours(scharr, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    s_img = np.zeros(image.shape[:2])
    for i in s_cnt:
        cv2.drawContours(s_img, [i], 0, (255, 255, 255), 2)
    imshow('scnt', s_img)
    if debug:
        imshow('sobel_add', sobel_add)
    laplacian = cv2.Laplacian(combine, cv2.CV_64F)
    imshow('laplacian', laplacian)
    return s2_clean


def old_get_yellow(b, g, r):
    # deprecated
    yellow_part = np.minimum(g, r)
    b2 = b.astype('int')
    b2 -= yellow_part
    b2[b2 < 0] = 255
    b2 = b2.astype('uint8')
    return b2


def get_real_blue2(original_image: np.array, neg_ref_value: float,
                   pos_ref_value: float) -> (np.array, int):
    """
    255-b is not real blue part
    Paused
    """
    if neg_ref_value > pos_ref_value or pos_ref_value <= 0:
        show_error('Bad negative and positive reference values.')
    factor = 1
    b, g, r = cv2.split(original_image)
    pos_ref_value = round(pos_ref_value)
    neg_ref_value = max(255, round(neg_ref_value))
    revert_b = revert(b)
    # amplify
    # revert_b = revert_b.astype('float')
    # log.info(f'Factor {factor}')
    # make sure express ratio <= 100%
    revert_b = revert_b.astype('float')
    factor = 255 // pos_ref_value
    revert_b = (revert_b - neg_ref_value) * factor
    revert_b[revert_b > 255] = 255
    revert_b = revert_b.astype('uint8')
    amplified_neg_ref = int(factor * neg_ref_value)
    return revert_b, amplified_neg_ref


def threshold(img: np.array, show=False) -> np.array:
    """
    Try to find suitable edge to split target and edge
    Paused.
    """
    r, g, b = cv2.split(img)
    r_g = r // 2 + g // 2
    r_g_reverse = 255 - r_g
    blur = cv2.GaussianBlur(r_g_reverse, (5, 5), 0)
    h, w = img.shape[:2]
    # mask = np.zeros([h + 2, w + 2], np.uint8)
    # ret1, th1 = cv2.threshold(img, 16, 255, cv2.THRESH_BINARY)
    equalize = cv2.equalizeHist(blur)
    r, t = cv2.threshold(equalize, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge = auto_Canny(255 - t)
    # cv2.floodFill(th2, mask=mask, seedPoint=(1,1), newVal=0, loDiff=3,
    # upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    log.debug(f'h * w: {h} {w}')
    # cv2.floodFill(th2, mask=mask, seedPoint=(1,h-1), newVal=0, loDiff=3,
    # upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    # cv2.floodFill(th2, mask=mask, seedPoint=(w-1,1), newVal=0, loDiff=3,
    # upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    # cv2.floodFill(th2, mask=mask, seedPoint=(w-1,h-1), newVal=0, loDiff=3,
    # upDiff=3, flags=cv2.FLOODFILL_FIXED_RANGE)
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    # cv2.THRESH_BINARY, 11, 4)
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    # cv2.THRESH_BINARY, 11, 2)
    th3 = auto_Canny(t)
    if show:
        imshow('th2', th2)
        imshow('threshold', t)
        imshow('edge', edge)
    log.debug(f'ret2 {ret2}')
    return th3


def get_sobel(img: np.array):
    xsobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    xsobel = cv2.convertScaleAbs(xsobel, alpha=1, beta=0)
    ysobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    ysobel = cv2.convertScaleAbs(ysobel, alpha=1, beta=0)
    sobel_add = cv2.addWeighted(xsobel, 0.5, ysobel, 0.5, 0)
    return sobel_add


def hist(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_max = np.max(hist)
    hist_max_idx = np.max(np.where(hist == hist_max))
    hist_edge = cv2.Canny(gray, 0, int(hist_max_idx * 0.9))
    imshow('histedge', hist_edge)
    return


def use_convex(convex, big_external_contours, level_cnt, external_area_dict,
               external_contours, big):
    if convex:
        # biggest contour is still biggest before and after process
        for old in big_external_contours:
            convexhull = cv2.convexHull(level_cnt[old], returnPoints=True)
            old_area = external_area_dict[old]
            new_area = cv2.contourArea(convexhull)
            log.info(f'Area before use convex hull: {old_area}')
            log.info(f'Area after use convex hull: {new_area}')
            level_cnt[old] = convexhull
            external_area_dict[old] = new_area
        external_contours.sort(key=lambda x: external_area_dict[x],
                               reverse=True)
        big_external_contours = external_contours[:big]


"""
## Options

### `auto_ref`

**This is a deprecated option.**

If use `-auto_ref`, the program will automatically detect the target regions
by finding the edge of the target. Such method may be inaccurate especially
when the image has extremely low contrast or the edge of the object is blur.

By default this option is closed and the program requires few mouse clicks
to get concise result.
"""