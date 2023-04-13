#!/usr/bin/python3
from itertools import combinations
from pathlib import Path
from sys import argv
import csv

from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import stats
import cv2
import numpy as np


def get_sample_info(csv_file: Path) -> dict:
    # parse input sample csv for analyze_GUS_value
    # sample: (filename, group)
    info = dict()
    with open(csv_file, 'r', newline='') as _:
        reader = csv.reader(_)
        # filename, sample, group
        for row in reader:
            filename, sample, group = row
            if sample in {'Negative', 'Positive'}:
                continue
            name_p = Path(filename)
            # convert name format
            new_name = name_p.stem + '-masked' + name_p.suffix
            info[sample] = [new_name, group]
    return info


def get_sample_info2(csv_file: Path) -> dict:
    # parse pygus output csv
    ratio_info = dict()
    exclude = {'Name', 'Positive reference', 'Negative reference'}
    with open(csv_file, 'r', newline='') as _:
        reader = csv.reader(_)
        for row in reader:
            (name, exp_value, exp_std, exp_area, total_value, total_std,
             total_area, exp_ratio, fig_size, zscore, outlier) = row
            if name in exclude:
                continue
            name_p = Path(name)
            # convert name format
            new_name = name_p.stem + '-masked' + name_p.suffix
            ratio_info[new_name] = float(exp_ratio)
    return ratio_info


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
        data = np.zeros(10)
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


def add_p_value(pair: list, group_data: dict, group_index: dict,
                ax: plt.axes, offset=4, height=8):
    # return height for next offset
    a, b = pair
    t_stat, p_value = stats.ttest_ind(group_data[a], group_data[b])
                                      # equal_var=True)
    # print(group_data[a], group_data[b])
    # print(f'{t_stat=}, {p_value=}')
    p_value_str = pvalue_legend(p_value)
    # print(p_value_str)
    a_x = group_index[a]
    b_x = group_index[b]
    a_b_y = max(np.max(group_data[a]), np.max(group_data[b])) + offset
    ax.plot([a_x, a_x, b_x, b_x],
            [a_b_y, a_b_y + height, a_b_y + height, a_b_y], lw=1,
            color='black')
    ax.text((a_x + b_x) / 2, a_b_y + height, p_value_str,
            ha='center', va='bottom', color='black')
    return height


def analyze_GUS_value():
    csv_file = Path(argv[1])
    info = get_sample_info(csv_file)
    data = dict()
    for sample in info:
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
                      showmedians=True,
                      showextrema=False)
    for pc, c in zip(v['bodies'], colors.TABLEAU_COLORS):
        pc.set_facecolor(c)
        pc.set_edgecolor('black')
        pc.set_alpha(1)
        pc.set_linewidth(0.5)
    v['cmedians'].set_color('white')
    v['cmedians'].set_linewidth(1)
    # add p value
    group_pairs = list(combinations(group_list, 2))
    group_index = dict(zip(group_list, range(1, len(group_list) + 1)))
    offset = 4
    pad = 4
    for pair in group_pairs:
        height = add_p_value(pair, group_data, group_index, ax, offset)
        offset = offset + height + pad
    ax.set_yticks(np.linspace(0, 256, 5))
    ax.set_xticks(range(1, len(group_list) + 1), group_list)
    ax.set_xlabel('Groups')
    ax.set_ylabel('GUS signal intensity')
    out_file = csv_file.with_suffix('.2.pdf')
    plt.savefig(out_file)
    plt.close()
    # plt.show()
    return out_file


def analyze_GUS_ratio():
    csv_file1 = Path(argv[1])
    csv_file2 = Path(argv[2])
    sample_info= get_sample_info(csv_file1)
    ratio_info = get_sample_info2(csv_file2)
    group_data = dict()
    for sample in sample_info:
        filename, group = sample_info[sample]
        express_ratio = ratio_info[filename]
        if group not in group_data:
            group_data[group] = [express_ratio]
        else:
            group_data[group].append(express_ratio)
    group_list = list(group_data.keys())
    fig, ax = plt.subplots()
    b = ax.boxplot([group_data[i] for i in group_list],
                   patch_artist=True, labels=group_list)
    for m in b['medians']:
        m.set_color('white')
        m.set_linewidth(1)
    for pc, c in zip(b['boxes'], colors.TABLEAU_COLORS):
        pc.set_facecolor(c)
        pc.set_alpha(1)
        pc.set_linewidth(0.5)

    # ax.set_xticks(range(1, len(group_list)+1), group_list)
    group_pairs = list(combinations(group_list, 2))
    group_index = dict(zip(group_list, range(1, len(group_list) + 1)))
    offset = 0.05
    pad = 0.05
    for pair in group_pairs:
        height = add_p_value(pair, group_data, group_index, ax, offset,
                             height=0.08)
        offset = offset + height + pad
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel('Groups')
    ax.set_ylabel('Expression area ratio')
    out_file = csv_file1.with_suffix('.1.pdf')
    plt.savefig(out_file)
    return ratio_info


def main():
    assert len(argv) == 3, ('Usage: python3 stats.py [sample info file] '
                            '[result csv file]')
    analyze_GUS_ratio()
    analyze_GUS_value()
    return


main()