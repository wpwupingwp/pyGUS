#!/usr/bin/python3

import logging
import coloredlogs

# define logger
FMT = '%(asctime)s %(levelname)-8s %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
# todo
debug = False
if debug:
    default_level = logging.DEBUG
else:
    default_level = logging.INFO
coloredlogs.install(level=default_level, fmt=FMT, datefmt=DATEFMT)
log = logging.getLogger('pyGUS')
is_gui = False

