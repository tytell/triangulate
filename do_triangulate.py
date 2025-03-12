import sys
import logging
import argparse

from qtpy.QtCore import Qt, QTimer
from qtpy import QtWidgets
from pyqtgraph.parametertree import ParameterTree, Parameter

parameters = [
    {'name': 'Output file', 'type': 'str', 'value': ''},
    {'name': 'Select output file...', 'type': 'action'},

    {'name': 'Video file directory', 'type': 'str', 'value': ''},
    {'name': 'Video file base name', 'type': 'str', 'value': ''},
    {'name': 'Select video file...', 'type': 'action'},

    {'name': 'Debug calibration', 'type': 'bool', 'value': False}
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

        self.params.child('Select output file...').sigActivated.connect(self.getOutputFile)
        self.params.child('Select video file...').sigActivated.connect(self.getVideoFile)

        self.setLayout(layout)

    def accept(self):
        self.writeSettings()
        self.loadCalibration()

        super(SetupDialog, self).accept()

    def getOutputFile(self):
        outputFile = self.params['Output file']
        if not outputFile:
            outputFile = ""
        outputFile, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Choose output file", directory=outputFile,
                                                            filter="*.csv")
        if outputFile:
            self.params['Output file'] = outputFile

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



def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Calibrate and triangulate points to 3D')

    parser.add_argument('config', nargs="*")

    parser.add_argument('--base_path', default='.',
                        help='Base path for data files')

    parser.add_argument('-nx', '--nsquaresx', type=int, 
                        help='Number of grid squares horizontally',
                        default=6)
    parser.add_argument('-ny', '--nsquaresy', type=int, 
                        help='Number of grid squares vertically',
                        default=6)
    parser.add_argument('-sz', '--square_length', type=float,
                        help = 'Size of square in mm',
                        default = 24.33)
    parser.add_argument('-mlen', '--marker_length', type=float,
                        help='Size of the Aruco marker in mm',
                        default=17)
    parser.add_argument('-mbits', '--marker_bits', type=int,
                        help='Information bits in the markers',
                        default=5)
    parser.add_argument('-dict','--dict_size', type=int,
                        help='Number of markers in the dictionary',
                        default=50)

    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help='Display increasingly verbose output')

    parser.add_argument('--calibration_videos', action='extend', nargs='+',
                        help="Video files containing synchronized images of the Charuco board")

    parser.add_argument('--camera_names', action='extend', nargs='+',
                        help="Names for eaech of the cameras, in the same order as the videos")

    parser.add_argument('--force_calibration', default=False, action="store_true",
                        help="Run the calibration even if the calibration TOML file is present")

    parser.add_argument('--debug', type=bool, default=False,
                        help="Save debug images for the calibration")
    parser.add_argument('--ndebugimages', type=int, default=10,
                        help="Save debug images for the calibration")

    parser.add_argument('--showreprojection', type=bool, default=False,
                        help="Show debug images to test reprojection error")

    parser.add_argument('--calibration_file', help='TOML File to store the calibration')

    parser.add_argument('--video_table', default=None,
                        help="CSV file containing a table of matching video files")
    
    parser.add_argument('-o', '--output_file', 
                        help="Name of the output CSV file with the triangulated points")

    args = parser.parse_args()    

    if args.config is not None:
        for config1 in args.config:
            with open(config1, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)
        
        args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    dlg = SetupDialog(parameters)
    if dlg.exec_() == QtWidgets.QDialog.Accepted:
        logging.debug('Yes!')
    else:
        logging.debug('No')




if __name__ == "__main__":
    main()
