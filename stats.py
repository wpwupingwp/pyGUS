#!/usr/bin/python3
from sys import argv
import csv

from matplotlib import pyplot as plt
import cv2
import numpy as np

def get_sample_info(csv_file: str) -> dict:
    # sample: (filename, group)
    data = dict()
    with open(csv_file, 'r', newline='') as _:
        reader = csv.reader(_)
        # filename, sample, group
        for row in reader:
            filename, sample, group = row
            data[sample] = [filename, group]
    return data


def get_data(filename: str) -> np.array:
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Bad input file {filename}')
    b, g, r, a = cv2.split(img)
    # revert
    b_r = 255 - b
    # apply mask
    data = b_r[a==255]
    if len(data) == 0:
        data = np.array([0])
    return data


def main():
    info = get_sample_info(argv[1])
    data = dict()
    for sample in info:
        if sample in ('Negative', 'Positive'):
            continue
        data[sample] = get_data(info[sample][0])
    v = plt.violinplot(list(data.values()), showmeans=False, showmedians=False,
                       showextrema=False)
    plt.yticks(np.linspace(0, 256, 9))
    plt.show()
    # cv2.imshow('raw', img)
    # cv2.waitKey()
    return


main()