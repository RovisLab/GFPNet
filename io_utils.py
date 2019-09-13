import h5py

import numpy as np
from open3d.open3d import io, geometry

from path_utils import PATHS


def read_off_file(filename):
    """
        Method used for reading off files, holding points representing the point cloud.
    :param filename: [in] Path to .off file.
    :param read_normals [in] Boolean flag to read normals or not
    :return: [out] N x 3 array
    """
    f = open(filename)
    f.readline()  # ignore the 'OFF' at the first line
    f.readline()  # ignore the second line
    points = []
    normals = []
    while True:
        new_line = f.readline()
        x = new_line.split(' ')
        if len(x) == 6:
            if x[0] != '3' and new_line != '' and len(x) == 6:
                P = np.array(x[0:3], dtype='float32')
                N = np.array(x[3:6], dtype='float32')
                points.append(P)
                normals.append(N)
            else:
                f.close()
                break
        else:
            if x != '\n' and x[0] != '3' and new_line != '':
                A = np.array(x[0:3], dtype='float32')
                points.append(A)
            else:
                f.close()
                break
    return points, normals


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def createH5_from(data_path, labels_path, files_format = 'off', h5_filename='train_fileh5'):
    from pathlib import Path

    FILE_FORMAT = files_format
    # script to read all clouds and write in h5
    hf = h5py.File(PATHS.NETWORK.root +'{0}.h5'.format(h5_filename), 'w')
    train_x_clouds_path = Path(data_path).glob('*.{0}'.format(FILE_FORMAT))
    train_x_labels_path = Path(labels_path).glob('*.{0}'.format(FILE_FORMAT))
    point_cloud_points = []
    labels = []

    for cloud_path in train_x_clouds_path:
        cloud_path = str(cloud_path)
        cloud_points_with_normals = read_off_file(cloud_path)
        point_cloud_points.append(cloud_points_with_normals)

    for label_path in train_x_labels_path:
        label_path = str(label_path)
        label_points_with_normals = read_off_file(label_path)
        labels.append(label_points_with_normals)

    hf.create_dataset('data', data=point_cloud_points)
    hf.create_dataset('label', data=labels)

    hf.close()
    print('{0} CREATED SUCCESSFULLY - can be found at : {1}.'.format(h5_filename, PATHS.NETWORK.root + h5_filename))


def createH5_GPF(train_X, train_Y, h5_filename='train_fileh5'):
    hf = h5py.File(PATHS.NETWORK.root +'{0}.h5'.format(h5_filename), 'w')
    hf.create_dataset('data', data=train_X)
    hf.create_dataset('label', data=train_Y)

    hf.close()
    print('{0} CREATED SUCCESSFULLY - can be found at : {1}.'.format(h5_filename,PATHS.NETWORK.root + h5_filename))