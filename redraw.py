#!/usr/bin/python3
from itertools import combinations
from pathlib import Path
from sys import argv
import csv

from matplotlib import pyplot as plt
from matplotlib import colors, rc
from scipy import stats
import cv2
import numpy as np


font = dict(size='22')
rc('font', **font)


def get_sample_info_3(csv_file: str, neg_img: str, pos_img: str) -> dict:
    """
    parse input sample csv for redraw pyGUS output image
    example csv file:
    ```
    A-H1.png,A-H1,high
    A-H2.png,A-H2,high
    A-H3.png,A-H3,high
    A-M1.png,A-M1,medium
    A-M2.png,A-M2,medium
    A-M3.png,A-M3,medium
    A-L1.png,A-L1,low
    A-L2.png,A-L2,low
    A-L3.png,A-L3,low
    ```
    """
    info = dict()
    labels = list()
    ref = ('Positive', 'Negative')
    with open(csv_file, 'r', newline='') as _:
        reader = csv.reader(_)
        # filename, sample, group
        for row in reader:
            filename, sample, group = row
            if sample in {'Negative', 'Positive'}:
                # do not read ref in here
                continue
            name_p = Path(filename)
            # convert name format
            new_name = name_p.stem + '-masked' + name_p.suffix
            info[sample] = [new_name, group]
            labels.append(sample)
    for img_file, ref_name in zip([neg_img, pos_img], ref):
        r_p = Path(img_file)
        new_name = r_p.stem + '-masked' + r_p.suffix
        info[ref_name] = [new_name, 'Ref']
        labels.append(ref_name)
    return info, labels


def read_results(info:dict) ->(list, list):
    pass



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
    fig, ax = plt.subplots(figsize=(10, 10))
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
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    # plt.show()
    return out_file


def write_image(results: tuple, labels: list, out: Path) -> Path:
    """
    violin outer and inner
    or violin outer and bar inner
    Args:
        results: calculate results
        labels: x-axis ticks
        out: output filename
    Returns:
        out: figure file
    """
    # result = (express_value, express_std, express_area, total_value,
    # total_std, total_area, express_ratio, express_flatten)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    short_labels = [Path(i).stem for i in labels]
    if len(labels) <= 5:
        figsize = (10, 6)
    else:
        figsize = (10 * len(labels) / 5, 6)
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot(211)
    x = np.arange(1, len(labels) + 1)
    width = 0.3
    violin_data_raw = []
    for i in results:
        data = i[-1]
        if len(data) == 0:
            data = [0]
        violin_data_raw.append(data)
    violin_data = np.array(violin_data_raw, dtype='object')
    # violin_data = np.array([i[-1] for i in results], dtype='object')
    try:
        violin_parts = ax1.violinplot(violin_data, showmeans=True,
                                      showmedians=False, showextrema=False,
                                      widths=0.4)
    except ValueError:
        show_error('Failed to plot results due to bad values.')
        return Path()
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#0d56ff')
        pc.set_edgecolor('black')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Expression value')
    ax1.set_yticks(np.linspace(0, 256, 9))
    ax1.set_xticks(np.arange(1, len(labels) + 1), labels=short_labels)
    # ax2 = ax1.twinx()
    ax2 = plt.subplot(212)
    express_area = [i[2] for i in results]
    # modify negative reference area
    # express_area[-1] = 0
    all_area = [i[-2] for i in results]
    total_area = [i[5] for i in results]
    no_express_area = [t - e for t, e in zip(total_area, express_area)]
    express_area_percent = [round(i / j, 4) for i, j in zip(express_area,
                                                            all_area)]
    no_express_area_percent = [round(i / j, 4) for i, j in zip(no_express_area,
                                                               all_area)]
    rects1 = ax2.bar(x, express_area_percent, width=width,
                     alpha=0.4,
                     color='green', label='Expression region')
    rects2 = ax2.bar(x, no_express_area_percent, width=width,
                     bottom=express_area_percent, alpha=0.4,
                     color='orange', label='No expression region')
    # rects1 = [i * 100 for i in rects1]
    # ax2.bar_label(rects1, label_type='center')
    # ax2.bar_label(rects2, label_type='center')
    ax2.set_xticks(np.arange(1, len(labels) + 1), labels=short_labels)
    ax2.legend()
    ax2.set_ylabel('Area percent')
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    log.info(f'Output figure file {out}')
    return out


def main():
    # python3 redraw.py stats.py list.csv A-mini.png A-35S.png
    assert len(argv) == 4, ('Usage: python3 stats.py [sample info file] '
                            '[positive origin image] [negative origin image]')
    sample_info, labels = get_sample_info_3(*argv[1:])
    print(sample_info, labels)
    return


main()