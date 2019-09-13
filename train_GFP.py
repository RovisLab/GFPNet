import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras import optimizers

from io_observation import IoObservation
from io_primitive import IoPrimitive
from io_utils import load_h5, read_off_file, createH5_from
from model import GFPNet
from parameters import ModellingParameters
from path_utils import PATHS
from transformations_utils import make_samples, upsample_point_set


def euclidean_distance_loss(y_true, y_pred):
    # Euclidian loss func 2D
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


class GFPNetParams:
    NUM_POINTS = 1024
    MAX_EPOCH = 1000
    BATCH_SIZE = 512
    OPTIMIZER = optimizers.Adam(lr=0.001, decay=0.9)
    GPU_INDEX = 0
    PATH_TO_WEIGHTS = ''
    FILE_EXTENSION = ''
    TRAIN_PATH = ''
    TRAIN_FILENAME = ''
    TEST_PATH = ''
    TEST_FILENAME = ''

    def __init__(self, num_points, max_epoch, batch_size):
        self.NUM_POINTS = num_points
        self.MAX_EPOCH = max_epoch
        self.BATCH_SIZE = batch_size

    def set_paths_to_data(self, path_to_trainh5, path_to_testh5):
        self.FILE_EXTENSION = '.' + str(path_to_trainh5).rsplit("\\")[-1:][0].split(".")[1]
        self.TRAIN_PATH = str(path_to_trainh5)
        self.TEST_PATH = str(path_to_testh5)
        self.TRAIN_FILENAME = str(path_to_trainh5).rsplit("\\")[-1:][0].split(".")[0]
        self.TEST_FILENAME = str(path_to_testh5).rsplit("\\")[-1:][0].split(".")[0]

    def load_data(self, loadfor='train'):
        if loadfor.lower() == 'train':
            points, labels = load_h5(self.TRAIN_PATH)
        else:
            points, labels = load_h5(self.TEST_PATH)

        return points, labels

# TODO: replace with generated files
train_h5_path = PATHS.NETWORK.root + ''
test_h5_path = PATHS.NETWORK.root + ''

net_params = GFPNetParams(num_points=50, max_epoch=100, batch_size=256)
net_params.set_paths_to_data(train_h5_path, test_h5_path)
print(net_params.PATH_TO_WEIGHTS)


def test_GFP():
    from open3d import visualization, geometry, utility
    test_points, test_labels = load_h5(test_h5_path)

    nr_points = ModellingParameters.NUM_POINTS_UPSAMPLE
    model = GFPNet(nr_points)

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mse', 'accuracy'])

    # print the model summary
    model.load_weights(net_params.PATH_TO_WEIGHTS)
    print(model.summary())
    primitive_path = PATHS.PATH_TO_PRIMITIVES.CAR.root + "CarPrimitive_15_500.off"
    io_primitive = IoPrimitive(primitive_path)
    io_primitive.load_primitive(normalize=False)

    pathlist = Path(PATHS.NETWORK.root).glob('*.{0}'.format('off'))
    for path in pathlist:
        lidar_cloud_path = str(path)
        file_name = lidar_cloud_path.split("\\")[-1]
        label_path = PATHS.NETWORK.TEST_MODELLED + file_name
        cloud_path = PATHS.NETWORK.TEST_CLOUD + file_name

        observation_points = read_off_file(cloud_path)
        io_observation_cloud = IoObservation(observation_points)
        io_primitive.scale_relative_to(io_observation_cloud)
        io_primitive.align_primitive_on_z_axis(io_observation_cloud)
        io_primitive.compute_normals()

        modelled_primitive_points = read_off_file(label_path)
        io_modelled_primitive = IoObservation(modelled_primitive_points)

        eval_X, boolIndexes = make_samples(io_observation_cloud, io_primitive, io_modelled_primitive, usePrimitivePoints=False, generate_for='test')
        eval_X = upsample_point_set(sample_collection=eval_X, points_count=ModellingParameters.NUM_POINTS_UPSAMPLE)

        cloud_bef = geometry.PointCloud()
        cloud_bef.points = utility.Vector3dVector(np.asarray(io_primitive.point_cloud.points))
        cloud_bef.normals = utility.Vector3dVector(np.asarray(io_primitive.point_cloud.normals))
        cloud_bef.paint_uniform_color([255, 0, 0])

        cloud_lidar = geometry.PointCloud()
        cloud_lidar.points = utility.Vector3dVector(np.asarray(io_observation_cloud.point_cloud))
        cloud_lidar.paint_uniform_color([0, 0, 0])

        cloud_modelled = geometry.PointCloud()
        cloud_modelled.points = utility.Vector3dVector(np.asarray(io_modelled_primitive.point_cloud.points))
        cloud_modelled.paint_uniform_color([255, 255, 0])

        visualization.draw_geometries([cloud_bef, cloud_lidar, cloud_modelled])
        final_pts = []
        idx = 0
        for i in range(0, len(eval_X)):
            pred = eval_X[i].reshape(-1, nr_points, 3)
            points = model.predict(pred)
            final_pts.append(points)
            idx = i

        final_pts = np.reshape(final_pts, newshape=(len(final_pts), 3))
        print('Final pts len : ', len(final_pts))
        final_pts = [point * ModellingParameters.CAR.SCALE for point in final_pts]
        true_indexes = [i for i, val in enumerate(boolIndexes) if val]
        for i in true_indexes:
            cloud_bef.colors[i] = [0, 255, 0]
        import pclpy.pcl.point_types as ptype
        aux = ptype.PointXYZ()
        new_points = []
        for i in range(0, len(final_pts)):
            val = io_primitive.point_cloud.points[true_indexes[i]] + (final_pts[i] - ModellingParameters.NORMALIZATION_CENTER)
            aux.x = val[0]
            aux.y = val[1]
            aux.z = val[2]
            new_points.append(val)
            # transform_neighbor_points_for(i, aux, srcPrimitive, None)
        cloud_aft = geometry.PointCloud()
        cloud_aft.points = utility.Vector3dVector(new_points)
        cloud_aft.paint_uniform_color([0, 0, 255])
        # cloud.normals = utility.Vector3dVector(cloud_points[:,3:6])
        visualization.draw_geometries([cloud_bef, cloud_aft, cloud_lidar, cloud_modelled])


def train_gfp():
    train_points, train_labels = net_params.load_data('train')
    test_points, test_labels = net_params.load_data('test')
    model = GFPNet(net_params.NUM_POINTS)
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    print(model.summary())
    print('Shape Train', np.shape(train_points))
    # Fit model on training data
    for i in range(1, 2):
        model.fit(train_points, train_labels, batch_size=net_params.BATCH_SIZE, epochs=net_params.MAX_EPOCH,
                  shuffle=True, verbose=1)
        s = "Current epoch is:" + str(i)
        print(s)
        if i % 10 == 0:
            score = model.evaluate(test_points, test_labels, verbose=1)
            print('Test loss: ', score[0])
            print('Test accuracy: ', score[1])
    model.save_weights(net_params.PATH_TO_WEIGHTS)


if __name__ == '__main__':

    # tf version
    print(tf.__version__)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
    createH5_from(PATHS.NETWORK.TRAIN_X, PATHS.NETWORK.TRAIN_Y, 'off', 'training_set')
    # train_gfp()
    # test_GFP()