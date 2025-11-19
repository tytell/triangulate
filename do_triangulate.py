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

from label3d.triangulate import get_base_path

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
        return None, None
    
    base_path = get_base_path(cfg)

    # check base path
    if not os.path.exists(base_path):
        logging.error(f"Could not access 'base_path' {base_path}")
        return None, None

    def mtime(fn, bp = base_path, is_input=True):
        fullfn = os.path.join(bp, fn)
        if not os.path.exists(fullfn):
            if is_input:
                return datetime.max       # largest available date
            else:
                return datetime.min     # smallest date if output file doesn't exist
        else:
            return datetime.fromtimestamp(os.path.getmtime(fullfn))

    # check lists of files
    for flist in ['calibration_videos', 'axes_files']:
        for i, fn in enumerate(cfg[flist]):
            fnall = os.path.join(base_path, fn)
            if not os.path.exists(fnall):
                logging.error(f"Could not access file {i+1} in {flist}.\n  Full path: {fnall}")
                return None, None
    
    # check the points file
    pfl = os.path.join(base_path, cfg['points_files_list'])
    if not os.path.exists(pfl):
        logging.error(f"Could not access the 'points_files_list':\n  Full path: {pfl}")
        return None, None

    calibration_file = get_triangulation_file_name(cfg, cfg['calibration_file'], withdate=False)
    logging.debug(f"{calibration_file=}")
    
    # get file modification dates
    filedates = {'configfile': [mtime(configfile, bp='')],
                 # same calibration file, but sometimes it's an output, sometimes input
                'calibration_file_output': [mtime(calibration_file, is_input=False)],
                'calibration_file_input': [mtime(calibration_file, is_input=True)],
                'points_files_list': [mtime(cfg['calibration_file'])],
                'calibration_videos': [mtime(f) for f in cfg['calibration_videos']],
                'axes_files': [mtime(f) for f in cfg['axes_files']],
                'axes_output': [mtime(cfg['axes_output'], is_input=False)],
                'output_file': [mtime(cfg['output_file'], is_input=False)],
                'calibration_notebook': [mtime(cfg['calibration_notebook'], bp='')],
                'axes_notebook': [mtime(cfg['axes_notebook'], bp='')],
                'triangulation_notebook': [mtime(cfg['triangulation_notebook'], bp='')]
                }

    points_files = pd.read_csv(os.path.join(base_path, cfg["points_files_list"]))
    points_infiles_dates = []
    points_outfiles_dates = []
    for idx, row in points_files.iterrows():
        # get the middle columns (not the first one and not the last)
        for c, f in enumerate(row.iloc[1:-1]):
            fn = os.path.join(base_path, cfg['points_files_path'], f)
            if not os.path.exists(fn):
                logging.error(f"Could not access the points file from row {idx+1}, column {c+1}:\n  Full path: {fn}")
                return None, None
            points_infiles_dates.append(mtime(fn, bp=''))

        fn = os.path.join(base_path, cfg['points_files_path'], row['Output'])
        points_outfiles_dates.append(mtime(fn, bp=''))
    
    filedates['points_files_in'] = points_infiles_dates
    filedates['points_files_out'] = points_outfiles_dates

    return cfg, filedates


def out_of_date(filedates, infiles, outfiles):
    outfiledates = []
    for outfile1 in outfiles:
        logging.debug(f"{outfile1=} {filedates[outfile1]=}")
        outfiledates = outfiledates + filedates[outfile1]
    
    # if any of the output files don't exist, we're out of date
    if None in outfiledates:
        logging.debug("Out of date: At least one output file does not exist.")
        return True
    
    outfilemin = min(outfiledates)
    logging.debug(f"Min output file date: {outfilemin}")

    infiledates = []
    for infile1 in infiles:
        infiledates = infiledates + filedates[infile1]
    infilemax = max(infiledates)
    logging.debug(f"Max input file date: {infilemax}")

    if infilemax > outfilemin:
        logging.debug("Out of date: At least one input file is newer than an output file.")
        return True
    
    return False

def render_notebook_and_rename(configfile, cfg, nbname0):
    out = subprocess.run(f"quarto render {nbname0} --execute-param parameterfile:{configfile} --to pdf",
                   shell=True, # capture_output=True,
                   stderr=subprocess.STDOUT)
    print(out.stdout)
    if out.returncode != 0:
        logging.error(f"Notebook {nbname0} had an error!")
        return False
    
    basename, ext = os.path.splitext(nbname0)
    nb_pdf = basename + '.pdf'
    if os.path.exists(nb_pdf):
        nb_pdf_final = get_triangulation_file_name(cfg, nb_pdf)
        copyfile(nb_pdf, nb_pdf_final)

    return True    

def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Calibrate and triangulate points to 3D')

    parser.add_argument('config', nargs="?", default=None)

    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help='Display increasingly verbose output')

    parser.add_argument('-f', '--force', action="store_true", default=False,
                        help="Force recalibration and retriangulation")
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
    
    cfg, filedates = check_config_file(configfile)
    if cfg is None:
        return
        
    # calibration notebook
    if args.force or out_of_date(filedates,
                   infiles=['configfile', 'calibration_videos', 'calibration_notebook'],
                   outfiles=['calibration_file_output']):
        logging.debug(f"{cfg['calibration_notebook']=}")
        if not render_notebook_and_rename(configfile, cfg, cfg['calibration_notebook']):
            logging.error("Calibration notebook rendering failed. Stopping")
            return
    else:
        logging.info("Calibration file is up to date. Skipping...")

    # axes notebook
    if args.force or out_of_date(filedates,
                   infiles=['configfile', 'calibration_file_input', 'axes_files'],
                   outfiles=['axes_output']):
        if not render_notebook_and_rename(configfile, cfg, cfg['axes_notebook']):
            logging.error("Axes notebook rendering failed. Stopping")
            return
    else:
        logging.info("Axes output file is up to date. Skipping...")

    # triangulation notebook
    if args.force or out_of_date(filedates,
                   infiles=['configfile', 'calibration_file_input', 'points_files_in'],
                   outfiles=['output_file', 'points_files_out']):
        if not render_notebook_and_rename(configfile, cfg, cfg['triangulation_notebook']):
            logging.error("Triangulate points notebook rendering failed. Stopping")
            return
    else:
        logging.info("Triangulation points files are up to date. No action taken.")

    




if __name__ == "__main__":
    main()
