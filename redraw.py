#!/usr/bin/python3
from pathlib import Path
from sys import argv
import csv

from matplotlib import rc
import cv2

from pyGUS.core import write_image, calculate

font = dict(size='14')
rc('font', **font)


def get_sample_info_3(csv_file: str, neg_img: str, pos_img: str) -> (dict, list):
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


def get_results(info: dict, labels: list) -> list:
    target_results = list()
    for sample in labels:
        img_file, _ = info[sample]
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        b, g, r, alpha = cv2.split(img)
        target_mask = alpha.copy()
        target_mask[target_mask>0] = 255
        original_image = cv2.merge([b, g, r])
        result, mask = calculate(original_image, target_mask)
        target_results.append(result)
    return target_results


def main():
    # python3 redraw.py stats.py list.csv A-mini.png A-35S.png
    assert len(argv) == 4, ('Usage: python3 stats.py [sample info file] '
                            '[positive origin image] [negative origin image]')
    sample_info, labels = get_sample_info_3(*argv[1:])
    target_results = get_results(sample_info, labels)
    out = Path('Results-new.pdf')
    write_image(target_results, labels, out)
    # print(sample_info, labels, out)
    return


main()