import sys, os
import logging
import argparse
from shutil import copyfile

from qtpy.QtCore import Qt, QTimer
from qtpy import QtWidgets
from pyqtgraph.parametertree import ParameterTree, Parameter

from ruamel.yaml import YAML
# from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.parser import ParserError
from datetime import datetime

import pandas as pd

import hashlib
BLOCKSIZE = 1048576

from label3d.calibrate_charuco import get_triangulation_file_name

import subprocess

dependencies = {
    'calibration_notebook': ['nsquaresx', 'nsquaresy', 'square_length',
                             'marker_length', 'marker_bits', 'dict_size',
                             'calibration_videos'],
    'axes_notebook': ['calibration_file', 'axes_files'],
    'triangulation_notebook': ['calibration_file', 'axes_output']
}

def check_config_file(configfile):
    yaml=YAML()   # default, if not specfied, is 'rt' (round-trip)

    try:
        with open(configfile, 'r') as f:
            cfg = yaml.load(f)
    except ParserError as p:
        logging.error(f"Error parsing config file {configfile}!")
        logging.error(p)
        return None
    
    # check base path
    if not os.path.exists(cfg['base_path']):
        logging.error(f"Could not access 'base_path' {cfg['base_path']}")
        return None
    
    # check lists of files
    for flist in ['calibration_videos', 'axes_files']:
        for i, fn in enumerate(cfg[flist]):
            fnall = os.path.join(cfg['base_path'], fn)
            if not os.path.exists(fnall):
                logging.error(f"Could not access file {i+1} in {flist}.\n  Full path: {fnall}")
                return None
    
    # check the points file
    pfl = os.path.join(cfg['base_path'], cfg['points_files_list'])
    if not os.path.exists(pfl):
        logging.error(f"Could not access the 'points_files_list':\n  Full path: {fpl}")
        return None

    points_files = pd.read_csv(os.path.join(cfg['base_path'], cfg["points_files_list"]))
    for idx, row in points_files.iterrows():
        for c, f in enumerate(row.iloc[1:]):
            fn = os.path.join(cfg['base_path'], cfg['points_files_path'], f)
            if not os.path.exists(fn):
                logging.error(f"Could not access the points file from row {idx+1}, column {c+1}:\n  Full path: {fn}")
                return None

    return cfg

def update_hash_from_file(h, fn):
    try:
        with open(fn, 'rb') as f:
            buf = f.read(BLOCKSIZE)
            while len(buf) > 0:
                h.update(buf)
                buf = f.read(BLOCKSIZE)
    except FileNotFoundError as err:
        h.update(b"")
        
    return h

def hash_dependencies(cfg):
    deps = dict()

    # calibration
    caldata = [cfg['nsquaresx'], cfg['nsquaresy'], cfg['square_length'],
               cfg['marker_length'], cfg['marker_bits'], cfg['dict_size']] + \
               cfg['camera_names']

    calhash = hashlib.new('sha256')
    calhash.update(repr(caldata).encode('utf-8'))

    for calfile in cfg['calibration_videos']:
        fn = os.path.join(cfg['base_path'], calfile)
        calhash = update_hash_from_file(calhash, fn)

    deps['calibration'] = calhash

    calib_file = get_triangulation_file_name(cfg, cfg['calibration_file'], withdate=False)

    # axes
    axeshash = hashlib.new('sha256')
    axeshash = update_hash_from_file(axeshash, calib_file)
    for axesfile in cfg['axes_files']:
        fn = os.path.join(cfg['base_path'], axesfile)
        axeshash = update_hash_from_file(axeshash, fn)

    deps['axes'] = axeshash


    # points
    basis_file = get_triangulation_file_name(cfg, cfg['axes_output'], withdate=False)

    pointshash = hashlib.new('sha256')
    pointshash = update_hash_from_file(pointshash, calib_file)
    pointshash = update_hash_from_file(pointshash, basis_file)
    
    points_files = pd.read_csv(os.path.join(cfg['base_path'], cfg["points_files_list"]))
    for idx, row in points_files.iterrows():
        for c, f in enumerate(row.iloc[1:]):
            fn = os.path.join(cfg['base_path'], cfg['points_files_path'], f)
            pointshash = update_hash_from_file(pointshash, fn)

    deps['points'] = pointshash

    return deps

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
    
    cfg = check_config_file(configfile)
    if cfg is None:
        return

    deps = hash_dependencies(cfg)
        
    # calibration notebook    
    render_notebook_and_rename(configfile, cfg, cfg['calibration_notebook'])

    # axes notebook
    render_notebook_and_rename(configfile, cfg, cfg['axes_notebook'])

    # triangulation notebook
    render_notebook_and_rename(configfile, cfg, cfg['triangulation_notebook'])
    




if __name__ == "__main__":
    main()
