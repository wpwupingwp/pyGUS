#!/usr/bin/python3
from sys import argv
import csv
import cv2
from matplotlib import pyplot as plt

data = dict()
with open(argv[1], 'r', newline='') as _:
    reader = csv.reader(_)
    # filename, sample, group
    for row in reader:
        filename, sample, group = row
        data[sample] = [filename, group]
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f'Bad input file {filename}')
        b, g, r, a = cv2.split(img)
        #whole[whole>0] = 255
        # b = cv2.bitwise_and(b, whole)
        # g = cv2.bitwise_and(g, whole)
        # r = cv2.bitwise_and(r, whole)
        # merge = cv2.merge([b, g, r])
        b[a<255] = 255
        b_r = 255 - b
        v = plt.violinplot(b_r[a>0], showmeans=False, showmedians=False, showextrema=False)
        plt.show()
        cv2.imshow('raw', img)
        # cv2.imshow('2', b_r)
        cv2.waitKey()
        break
print(data.items(), sep='\n')