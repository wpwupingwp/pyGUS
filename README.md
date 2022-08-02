# pyGUS
[![Build Status](https://travis-ci.org/wpwupingwp/novowrap.svg?branch=master)](https://travis-ci.org/wpwupingwp/novowrap)
[![PyPI version](https://badge.fury.io/py/novowrap.svg)](https://badge.fury.io/py/novowrap)
[![Anaconda version](https://anaconda.org/wpwupingwp/novowrap/badges/version.svg)](https://anaconda.org/wpwupingwp/novowrap)
# Quick start
Download [the package](https://github.com/wpwupingwp/pyGUS/releases),
unzip, and then double-click `pyGUS.exe` or `pyGUS`.

__OR__

Make sure you have [Python](https://www.python.org/) (3.8 or higher) installed.

Open terminal, run
   ```shell
   # Install, using pip (recommended)
   pip install pyGUS --user
   # Or, use conda
   conda install -c wpwupingwp pyGUS
   # Run
   # Windows
   python -m pyGUS
   # Linux and MacOS
   python3 -m pyGUS
   ```

# Table of Contents
* [Quick start](#quickstart)
* [Feature](#feature)
* [Prerequisite](#prerequisite)
    * [Hardware](#hardware)
    * [Software](#software)
* [Installation](#installation)
    * [Portable](#portable)
    * [Install with pip](#Installwithpip)
* [Usage](#usage)
    * [Graphical user interface](#graphicalinterface)
    * [Command line](#commandline)
* [Input](#input)
* [Output](#output)
* [Performance](#performance)
* [Citation](#citation)
* [Flowchart](#flowchart)
# Feature
:heavy_check_mark: Quantify GUS stain images automatically.

:heavy_check_mark: Use negative and positive reference to calibrate expression value.

:heavy_check_mark: Support Macbeth color checker to calibrate color bias.
# Prerequisite
## Hardware
The could run in normal computers and have no extra requirements for memory, CPU, et al.

Currently, macOS, Linux and Microsoft Windows systems were supported.
## Software
For the portable version, nothing needs to be installed manually.

For installing from pip, [Python](https://www.python.org/downloads/) is
required. Notice that the python version should be **3.8** or higher.

:white_check_mark: All third-party dependencies will be automatically
installed with the Internet, including `opencv-contrib-python`, `numpy`, `matplotlib`,
`coloredlogs` (python packages).

# Installation
## Portable
Download from the [link](https://github.com/wpwupingwp/pyGUS/releases),
unpack and run.
## Install with pip
1. Install [Python](https://www.python.org/downloads/). *3.8 or newer* is
   required.

2. Open the command line, run
```shell
pip3 install novowrap --user
```
# Usage
## Photo
体视镜，相机
分辨率
大小
边缘
背景
间隔
## Graphical interface
If installed with pip,
```shell
# Windows
python -m novowrap
# Linux and MacOS
python3 -m novowrap
```
If use the portable version, just double-click the `pyGUS.exe` or `pyGUS` in
the folder. Then click the button to choose which mode to run. 
## Command line
:exclamation: In Linux and macOS, Python2 is `python2` and Python3 is `python3`.  However,
in Windows, Python3 is called `python`, too. Please notice the difference.

* Show help information
 ```shell
 # Windows
 python -m pyGUS -h 
 # Linux and MacOS
 python3 -m pyGUS -h 
 ```
* Run
 ```shell
 # Windows
 #   mode 1
 python -m pyGUS -mode 1 -ref1 [file1] -ref2 [file2] -images [files3] [file4] ...
 #   mode 2
 python -m pyGUS -mode 2 -ref1 [file1] -images [file2] [file3] ...
 #   mode 3
 python -m pyGUS -mode 3 -ref1 [file1] -ref2 [file2] -images [files3] [file4] ...
 #   mode 4
 python -m pyGUS -mode 4 -images [file1] [file2] ...
 # Linux and macOS
 #   mode 1
 python3 -m pyGUS -mode 1 -ref1 [file1] -ref2 [file2] -images [files3] [file4] ...
 #   mode 2
 python3 -m pyGUS -mode 2 -ref1 [file1] -images [file2] [file3] ...
 #   mode 3
 python3 -m pyGUS -mode 3 -ref1 [file1] -ref2 [file2] -images [files3] [file4] ...
 #   mode 4
 python3 -m pyGUS -mode 4 -images [file1] [file2] ...
 ```
# Input
## Negative reference
One plant (whole or partial) with low expression or non expression.
## Positive reference
One plant (whole or partial) with high expression.
## Mode 1
The `mode 1` requires _negative reference_ (`-ref1`),  _positive reference_ (`-ref2`)
and target images (`-images`, one or more).
## Mode 2
The `mode 2` requires one refernce image (`-ref1`) with negative reference on left
and positive reference on right. In each target images (`-images`, one or more), 
target plants are on left, positive references are on right separately.
## Mode 3
The `mode 3` requires _negative reference_ (`-ref1`),  _positive reference_ (`-ref2`)
and target images (`-images`, one or more). In each image, the plant is on left
and a Macbetch color checker is on right.
## Mode 4
The `mode 4` requires one or more target images (`-images`). Users select reference and 
target regions by mouse.
# Output
`.csv` files: **csv** format table, with expression information of each image.

`-fill.png` files: Images filled with differenct colors. Blue means target, red means
background, green means darker regions inner target.

`-calibrate.png` files: Images calibrated with color checker.
# Performance
It depends on hardware. Normally one image cost seconds. The program does not require
too much memory or disk space.
# Citation
Unpublished
# License
The software itself is licensed under
[AGPL-3.0](https://github.com/wpwupingwp/pyGUS/blob/master/LICENSE) (**not include third-party
software**).

# Q&A
Please submit your questions in the
[Issue](https://github.com/wpwupingwp/pyGUS/issues) page :smiley:

* Q: I can't see the full UI, some part was missing.

  A: Please try to drag the corner of the window to enlarge it. We got reports
  that some users in macOS have this issue.

* Q: I got the error message that I don't have `tkinter `module installed.

  A: If you want to run GUI on Linux or macOS, this error may happen because the
  Python you used did not include tkinter as default package (kind of weird). Run
  ```
  # Debian and Ubuntu
  sudo apt install python3-tk
  # CentOS
  sudo yum install python3-tk
  ```
  may help.

  For macOS users or linux users without root privilege, please try to install the
  newest version of Python or to use conda, see [conda](https://docs.conda.io/en/latest/miniconda.html)
  and [Python](https://www.python.org/download/mac/tcltk/) for details.

* Q: It says my input is invalid, but I'm sure it's OK!

  A: Please check your files' path. The `space` character in the folder name
  or filename may cause this error.

## Flowchart
```mermaid
flowchart TB
    subgraph main
        m1((Mode1))
        m2((Mode2))
        m3((Mode3))
        m4((Mode4))
        targets[Target images]
        c[Macbeth Color checker]
        ref1[Positive reference]
        ref2[Negative reference]
        ref3[Negative and Positive reference]
        targets
        ref1 & ref2 & targets --> m1
        ref3 & targets --> m2
        m3 --> c --> ref1 & ref2 & targets
        m4 -- Mouse --> ref1 & ref2 & targets
        style c fill:#399
        style m1 fill:#393
        style m2 fill:#393
        style m3 fill:#393
        style m4 fill:#393
    end
    
```
```mermaid
flowchart LR
    subgraph run
        g1[Read image]
        g1.5[[Color calibration]]
        g2[Split to R, G, B channels]
        g3[Revert g//2+r//2]
        g4[Get edge]
        g5[Filter contours]
        g6[Get target mask]
        g7[Calculate]
        g8[Output table and figure]
        g1 --> g1.5 --> g2 --> g3 --> g4 --> g5 --> g6 --> g7 --> g8 
        style g1.5 fill:#59f
        style g6 fill:#59f
        style g7 fill:#557788
        style g8 fill:#59f
    end
    subgraph g1.5[Color calibration]
        direction TB
        c1[Detect Macbeth color checker]
        c2[Generate CCM matrix]
        c3[Apply matrix to original whole image]
        c4[Generate calibrated image]
        c1 --> c2 --> c3 --> c4
    end
    subgraph g4[Get contour]
        direction LR
        g41[Canny] --> g42[Gaussian Blur] --> g43[Dilate] --> g44[Erode]
        g44 --> g45[Find contours]
    end
    subgraph g5[Filter contours]
        direction LR
        g51[External] --> Big & Small 
        g52[Inner]
        g53[Fake inner]
        g54[Inner background]
        g53 & g54 --- g52
    end
    subgraph g6[Get target mask]
        direction LR
        one
        two --> s([Split by binding box]) --> Left & Right
        one & Left & Right --> f[Fill mask]
    end
    subgraph g7[Calculate]
        direction LR
        neg & pos --> target --- Mean & Area & Ratio
    end

```
## build
```
python3 -m build --wheel
```

