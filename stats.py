#!/usr/bin/python3
from sys import argv
import csv
from itertools import combinations

from matplotlib import pyplot as plt
import cv2
import numpy as np
from scipy import stats


def get_sample_info(csv_file: str) -> dict:
    # sample: (filename, group)
    info = dict()
    with open(csv_file, 'r', newline='') as _:
        reader = csv.reader(_)
        # filename, sample, group
        for row in reader:
            filename, sample, group = row
            info[sample] = [filename, group]
    return info


def get_data(filename: str) -> np.array:
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Bad input file {filename}')
    b, g, r, a = cv2.split(img)
    # revert
    b_r = 255 - b
    # apply mask
    data = b_r[a == 255]
    if len(data) == 0:
        data = np.array([0])
    return data


def pvalue_legend(pvalue: float) -> str:
    assert pvalue >= 0.0, f'Bad p value {pvalue}'
    if pvalue <= 1.0e-4:
        return '****'
    elif pvalue <= 1.0e-3:
        return '***'
    elif pvalue <= 1.0e-2:
        return '**'
    elif pvalue <= 5.0e-2:
        return '*'
    else:
        return 'n.s.'


def main():
    info = get_sample_info(argv[1])
    data = dict()
    for sample in info:
        if sample in ('Negative', 'Positive'):
            continue
        data[sample] = get_data(info[sample][0])
    group_data = dict()
    for sample in data:
        group = info[sample][1]
        if group not in group_data:
            group_data[group] = data[sample]
        else:
            group_data[group] = np.concatenate([group_data[group],
                                                data[sample]])
    group_list = list(group_data.keys())
    fig, ax = plt.subplots()
    v = ax.violinplot([group_data[i] for i in group_list], showmeans=False,
                      showmedians=False,
                      showextrema=False)
    # add p value
    group_pair_pvalue = {i: 0.0 for i in combinations(group_list, 2)}
    group_index = dict(zip(group_list, range(1, len(group_list) + 1)))
    offset = 4
    height = offset * 2
    for group_pair in group_pair_pvalue:
        a, b = group_pair
        t_stat, pvalue = stats.ttest_ind(group_data[a], group_data[b],
                                         equal_var=False)
        # print(group_data[a], group_data[b], pvalue)
        # print(f'{t_stat=}')
        group_pair_pvalue[group_pair] = pvalue_legend(pvalue)
        a_x = group_index[a]
        b_x = group_index[b]
        a_b_y = max(np.max(group_data[a]), np.max(group_data[b])) + offset
        plt.plot([a_x, a_x, b_x, b_x],
                 [a_b_y, a_b_y + height, a_b_y + height, a_b_y], lw=1,
                 color='black')
        plt.text((a_x + b_x) / 2, a_b_y+offset, group_pair_pvalue[group_pair],
                 ha='center', va='bottom', color='black')

    plt.yticks(np.linspace(0, 256, 9))
    plt.xticks(range(1, len(group_list) + 1), group_list)
    plt.xlabel('Groups')
    plt.ylabel('GUS signal intensity')
    plt.show()
    return


main()