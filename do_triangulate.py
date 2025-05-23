import sys, os
import logging
import argparse
from shutil import copyfile

from qtpy.QtCore import Qt, QTimer
from qtpy import QtWidgets
from pyqtgraph.parametertree import ParameterTree, Parameter

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from datetime import datetime

from label3d.calibrate_charuco import get_triangulation_file_name

import subprocess

def render_notebook_and_rename(configfile, cfg, nbname0):
    out = subprocess.run(f"quarto render {nbname0} --execute-param parameterfile:{configfile} --to pdf",
                   shell=True, # capture_output=True,
                   stderr=subprocess.STDOUT)
    print(out.stdout)
    if out.returncode != 0:
        logging.error(f"Notebook {nbname0} had an error!")

    basename, ext = os.path.splitext(nbname0)
    nb_pdf = basename + '.pdf'
    if os.path.exists(nb_pdf):
        nb_pdf_final = get_triangulation_file_name(cfg, nb_pdf)
        copyfile(nb_pdf, nb_pdf_final)
        

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

    if configfile is None or len(configfile) == 0:
        logging.info('No config file. Ending')
        return

    logging.debug(f"{configfile=}")

    if not os.path.exists(configfile):
        logging.error(f"Config file {configfile} not found.")
        return
    
    yaml=YAML()   # default, if not specfied, is 'rt' (round-trip)

    with open(configfile, 'r') as f:
        cfg = yaml.load(f)

    # calibration notebook    
    render_notebook_and_rename(configfile, cfg, cfg['calibration_notebook'])

    # axes notebook
    render_notebook_and_rename(configfile, cfg, cfg['axes_notebook'])

    # triangulation notebook
    render_notebook_and_rename(configfile, cfg, cfg['triangulation_notebook'])
    




if __name__ == "__main__":
    main()
