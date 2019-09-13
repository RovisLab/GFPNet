import os
from pathlib import Path
basepath = os.path.dirname(__file__)
DIRECTORY_PATH = os.path.abspath(os.path.join(basepath))

class PATHS(object):
    '''
        Constant variables representing the paths to the folders used in the code.
    '''

    class KITTI(object):
        '''
        All these paths, are related to the kitti dataset folder, downloaded from ki
            @object_dir - path to kitti dataset directory.
            @IMG_ROOT - path to .png or .img files that are found in the image_2 folder.
            @PC_ROOT - path to velodyne pointclouds
            @CALIB_ROOT - path to calibration data, found in the folder calib.
            '''
        basepath = os.path.dirname(__file__)
        DIRECTORY_PATH = os.path.abspath(os.path.join(basepath, "Dataset" "\\kitti"))
        IMG_ROOT = os.path.join(DIRECTORY_PATH, "image_2\\")
        PC_ROOT = os.path.join(DIRECTORY_PATH, "velodyne\\")
        CALIB_ROOT = os.path.join(DIRECTORY_PATH, "calib\\")

        CAR_CLASS = 'Car'
        PEDESTRIAN_CLASS = 'Pedestrian'
        CYCLIST_CLASS = 'Cyclist'
        TRUCK_CLASS = 'Truck'


    class NETWORK(object):
        root = os.path.join(DIRECTORY_PATH, "dataset", "\\network")
        TRAIN_X = os.path.join(root, "train_x\\")
        TRAIN_Y = os.path.join(root, "train_y\\")


    class PATH_TO_PRIMITIVES(object):
        root = os.path.join(DIRECTORY_PATH, "dataset", "\\primitives")
        BOTTLE = os.path.join(root, "bottle\\")
        GLASSES = os.path.join(root, "glasses\\")
        HAT = os.path.join(root, "hat\\")
        KNIFE = os.path.join(root, "knife\\")
        MUG = os.path.join(root, "mug\\")
        OTHER = os.path.join(root, "other\\")
        SOFA = os.path.join(root, "sofa\\")
        HUMAN = os.path.join(root, "human\\")

        class CAR(object):
            root = os.path.join(DIRECTORY_PATH, "dataset", "primitives", "car\\")

