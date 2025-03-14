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

parameters = [
    {'name': 'Config YAML file', 'type': 'file', 'value': '',
        'winTitle': 'Select config YAML file',
        'nameFilter': '*.yml',
        'fileMode': 'AnyFile'},
    {'name': 'Save to new config file...', 'type': 'action'},

    {'name': 'base_path', 'type': 'str', 'value': ''},

    {'name': 'Charuco board parameters', 'type': 'group', 'children': [
        {'name': 'nsquaresx', 'type': 'int', 'value': 6, 'limits': [2, 100]},
        {'name': 'nsquaresy', 'type': 'int', 'value': 6, 'limits': [2, 100]},
        {'name': 'square_length', 'type': 'float', 'value': 24.33, 'suffix': 'mm',
            'siPrefx': True},
        {'name': 'marker_length', 'type': 'float', 'value': 24.33, 'suffix': 'mm',
            'siPrefx': True},
        {'name': 'marker_bits', 'type': 'int', 'value': 5},
        {'name': 'dict_size', 'type': 'int', 'value': 50}
    ]},
    {'name': 'Calibration setup', 'type': 'group', 'children': [
        {'name': 'calibration_videos', 'type': 'group', 'children': [],
            'addText': 'Add video'},
        {'name': 'camera_names', 'type': 'group', 'children': [],
            'addText': 'Add camera'},
        {'name': 'debug', 'type': 'bool', 'value': False},
        {'name': 'board_points_file', 'type': 'file', 'value': '',
                 'winTitle': 'Select debug output file for the board points',
                'nameFilter': '*.csv',
                'fileMode': 'AnyFile'},
        {'name': 'calibration_file', 'type': 'file', 'value': 'calibration.toml',
                 'winTitle': 'Select output file for the calibration',
                'nameFilter': '*.toml',
                'fileMode': 'AnyFile'}
    ]},

    {'name': 'Axes setup', 'type': 'group', 'children': [
        {'name': 'axes_files_path', 'type': 'file', 'value': '',
            'winTitle': 'Select folder with axes files',
            'nameFilter': '*.*',
            'options': 'ShowDirsOnly', 
            'fileMode': 'Directory'},
        {'name': 'axes_files', 'type': 'group', 'children': [],
            'addText': 'Add axes points file'},
        {'name': 'axes_output', 'type': 'file', 'value': 'axes.csv',
            'winTitle': 'Select output file for axes data',
            'nameFilter': '*.csv',
            'fileMode': 'AnyFile'}  
    ]},

    {'name': 'Video trials for triangulation', 'type': 'group', 'children': [
        {'name': 'points_files_path', 'type': 'file', 'value': '',
            'winTitle': 'Select folder with points files',
            'nameFilter': '*.*',
            'options': 'ShowDirsOnly', 
            'fileMode': 'Directory'},
        {'name': 'points_files_list', 'type': 'file', 'value': '',
            'winTitle': 'Select file containing matching trials by camera',
            'nameFilter': '*.csv',
            'fileMode': 'ExistingFile'},  
        {'name': 'output_file', 'type': 'file', 'value': '',
            'winTitle': 'Select file to save the 3D points',
            'nameFilter': '*.csv',
            'fileMode': 'AnyFile'}
    ]}

]

class SetupDialog(QtWidgets.QDialog):
    def __init__(self, parent = None):

        super(SetupDialog, self).__init__()

        self.params = Parameter.create(name='Parameters', type='group',
                                       children = parameters)
        self.paramtree = ParameterTree()
        self.paramtree.setParameters(self.params, showTop=False)

        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                                    QtWidgets.QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.paramtree)
        layout.addWidget(self.buttonBox)

        self.getFlatParam('Config YAML file').sigValueChanged.connect(self._setSelectConfigFile)
        self.getFlatParam('Save to new config file...').sigActivated.connect(self.chooseNewConfigFile)
        self.getFlatParam('calibration_videos').sigAddNew.connect(self.addCalibrationVideos)

        # self.params.child('Select video file...').sigActivated.connect(self.getVideoFile)

        self.setLayout(layout)

    def showEvent(self, arg__1):
        if len(self.params['Config YAML file']) == 0 or \
            not os.path.exists(self.params['Config YAML file']):
            self.chooseConfigFile()

        return super().showEvent(arg__1)

    def parseYMLtoParameters(self, yml):
        for k, v in yml.items():
            p = self.getFlatParam(k)
            if p is None:
                logging.warning(f"Parameter {k} is not recognized")
                next

            if isinstance(v, list):
                self.setParameterToList(p, v)
            else:
                p.setValue(v)

    def setParameterToList(self, p, l):
        p.clearChildren()
        
        for i, c in enumerate(l):
            p.addChild(dict(name=f"{i+1}", type='str', 
                                value=c, removable=True, renamable=True))

    def _findParam(self, params, name):
        for p in params.childs:
            if p.name() == name:
                return p
            elif p.type() == 'group':
                found = self._findParam(p, name)
                if found is not None:
                    return found
        
        return None
    
    def getFlatParam(self, name):
        return self._findParam(self.params, name)

    def accept(self):
        self.saveToYML()
        super(SetupDialog, self).accept()

    def chooseNewConfigFile(self):
        configfile, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Choose YAML config file", 
                                                            filter="*.yml")
        self.setConfigFile(configfile)

    def chooseConfigFile(self):
        configfile, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose YAML config file", 
                                                            filter="*.yml")
        self.setConfigFile(configfile)

    def setConfigFile(self, configfile):
        self.params['Config YAML file'] = configfile
        self._setSelectConfigFile()

    def _setSelectConfigFile(self):
        configFile = self.params['Config YAML file']

        if len(self.params['base_path']) == 0 or \
                not os.path.exists(self.params['base_path']):
            pn, _ = os.path.split(configFile)
            self.params['base_path'] = pn
        
        if os.path.exists(configFile):
            yaml = YAML(typ='rt')
            with open(configFile, 'r') as f:
                cfg = yaml.load(f)

            self.parseYMLtoParameters(cfg)

    def getConfigFile(self):
        return self.params['Config YAML file']
    
    def saveToYML(self):
        yaml = YAML()
        configFile = self.params['Config YAML file']
        if os.path.exists(configFile):
            with open(configFile, 'r') as f:
                cfg = yaml.load(f)
            isnew = False
        else:
            isnew = True
        
        nowstr = datetime.now().isoformat(timespec='minutes')
        cfg.yaml_set_start_comment(f"""\
Triangulation setup
{nowstr}
""")

        self.recurseToYML(self.params.childs[2:], cfg, isnew)

        with open(configFile, 'w') as file:
            yaml.dump(cfg, file)

    def recurseToYML(self, params, cfg, isnew):
        for p1 in params:
            if p1.type() == 'group':
                if p1.childs[0].name() == '1':
                    vals = [sub1.value() for sub1 in p1.childs]

                    cfg[p1.name()] = vals
                else:
                    self.recurseToYML(p1.childs, cfg, isnew)
                    if isnew:
                        cfg.yaml_set_comment_before_after_key(p1.childs[0].name(), before=p1.name())
            else:
                cfg[p1.name()] = p1.value()

    def addCalibrationVideos(self):
        param = self.getFlatParam('calibration_videos')
        self.addFiles(param)

    def addFiles(self, param):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Choose file(s)", directory=self.params['base_path'],
                                                            filter="*.*")

        if not files:
            return
        
        for f1 in files:
            fn1 = os.path.relpath(f1, self.params['base_path'])

            param.addChild(dict(name=f"{len(param.childs)+1}", type='file', 
                                value=fn1, removable=True, renamable=True,
                                relativeTo=self.params['base_path']))
                
    def getVideoFile(self):
        videoFile = self.params['Video file directory']
        if not videoFile:
            videoFile = ""
        videoFile, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choose video file or directory", directory=videoFile,
                                                            filter="*.*")

        if videoFile:
            if os.path.isfile(videoFile):
                videoDir, videoBaseName = os.path.split(videoFile)
                self.params['Video file directory'] = videoDir
                self.params['Video file base name'] = videoBaseName
            elif os.path.isdir(videoFile):
                self.params['Video file directory'] = videoFile

class RunDialog(QtWidgets.QDialog):
    def __init__(self, configfile, parent = None):
        super(RunDialog, self).__init__()

        self.configFile = configfile

        formlayout = QtWidgets.QFormLayout()
        self.runCalibration = QtWidgets.QPushButton("Go")
        formlayout.addRow("1. Run calibration", self.runCalibration)

        self.runAxes = QtWidgets.QPushButton("Go")
        formlayout.addRow("2. Set up axes", self.runAxes)

        self.runTriangulation = QtWidgets.QPushButton("Go")
        formlayout.addRow("3. Do triangulation", self.runTriangulation)

        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.buttonBox.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(formlayout)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

    def runCalibration(self):
        subprocess.run("quarto ")
        
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

    if args.config is not None:
        yaml=YAML(typ='rt')   # default, if not specfied, is 'rt' (round-trip)

        with open(args.config, 'r') as f:
            cfg = yaml.load(f)
            parser.set_defaults(**cfg)
        
        args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    setupdlg = SetupDialog(parameters)
    if args.config is not None:
        setupdlg.setConfigFile(args.config)
        
    if setupdlg.exec_() == QtWidgets.QDialog.Accepted:
        rundlg = RunDialog(setupdlg.getConfigFile())
        rundlg.exec_()
    else:
        logging.debug('No')




if __name__ == "__main__":
    main()
