#!/usr/bin/python3

from sys import argv
from pyGUS.core import cli_main
from pyGUS.ui import ui_main

if __name__ == '__main__':
    if argv[-1] in ('-h', '--help') or len(argv) > 1:
        cli_main()
    else:
        try:
            ui_main()
        except Exception:
            cli_main()