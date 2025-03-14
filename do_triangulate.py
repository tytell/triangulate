import sys, os
import logging
import argparse

from qtpy.QtCore import Qt, QTimer
from qtpy import QtWidgets
from pyqtgraph.parametertree import ParameterTree, Parameter

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from datetime import datetime

import subprocess


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Calibrate and triangulate points to 3D')

    parser.add_argument('config', nargs="?", default=None)

    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help='Display increasingly verbose output')

    parser.add_argument('--debug', type=bool, default=False,
                        help="Save debug images for the calibration")

    parser.add_argument('-o', '--output_file', 
                        help="Name of the output CSV file with the triangulated points")

    args = parser.parse_args()    

    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    if args.config is None:
        configfile, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Choose YAML config file", 
                                                                    filter="*.yml")
    else:
        configfile = args.config

    logging.debug(f"{configfile=}")

    yaml=YAML()   # default, if not specfied, is 'rt' (round-trip)

    with open(configfile, 'r') as f:
        cfg = yaml.load(f)

    out = subprocess.run(f"quarto render calibrate_charuco.qmd --execute-param parameterfile:{configfile} --to pdf",
                   shell=True, # capture_output=True,
                   stderr=subprocess.STDOUT)
    print(out.stdout)

    out = subprocess.run(f"quarto render triangulate_axes.qmd --execute-param parameterfile:{configfile} --to pdf",
                   shell=True, # capture_output=True,
                   stderr=subprocess.STDOUT)
    print(out.stdout)

    out = subprocess.run(f"quarto render triangulate_axes.qmd --execute-param parameterfile:{configfile} --to pdf",
                   shell=True, # capture_output=True,
                   stderr=subprocess.STDOUT)
    print(out.stdout)
    




if __name__ == "__main__":
    main()
