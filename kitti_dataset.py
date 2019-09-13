from os import path


class KITTIPaths(object):
    '''
    All these paths, are related to the kitti dataset folder, downloaded from ki
        @object_dir - path to kitti dataset directory.
        @IMG_ROOT - path to .png or .img files that are found in the image_2 folder.
        @PC_ROOT - path to velodyne pointclouds
        @CALIB_ROOT - path to calibration data, found in the folder calib.
        '''
    basepath = path.dirname(__file__)
    #TODO: set a generic path for the users to download all the data in
    DIRECTORY_PATH = path.abspath(path.join(basepath, "dataset", "kitti"))
    IMG_ROOT = DIRECTORY_PATH + "/" + "image_2/"
    PC_ROOT = DIRECTORY_PATH + "/" + "velodyne/"
    CALIB_ROOT = DIRECTORY_PATH + "/" + "calib/"

    CAR_CLASS = 'Car'
    PEDESTRIAN_CLASS = 'Pedestrian'
    CYCLIST_CLASS = 'Cyclist'
    TRUCK_CLASS = 'Truck'


