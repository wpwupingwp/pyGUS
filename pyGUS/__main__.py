#!/usr/bin/python3

from pyGUS.core import cli_main
from pyGUS.ui import ui_main

if __name__ == '__main__':
    try:
        ui_main()
    except Exception:
        cli_main()