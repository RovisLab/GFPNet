import time

import numpy as np
from open3d.open3d import registration
from open3d.open3d.geometry import voxel_down_sample, estimate_normals, KDTreeSearchParamHybrid, KDTreeSearchParamKNN, PointCloud, KDTreeFlann
from open3d.open3d.registration import registration_ransac_based_on_feature_matching, \
    TransformationEstimationPointToPoint, CorrespondenceCheckerBasedOnEdgeLength, CorrespondenceCheckerBasedOnDistance, \
    RANSACConvergenceCriteria, compute_fpfh_feature, registration_fast_based_on_feature_matching, \
    FastGlobalRegistrationOption, registration_icp, TransformationEstimationPointToPlane, ICPConvergenceCriteria

from io_observation import IoObservation
from parameters import ModellingParameters


def compute_centroid(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]

    length = len(points)

    return [sum(x)/length, sum(y)/length, sum(z)/length]


def delaunay_triangulation(points_array):
    import numpy as np
    from scipy.spatial import Delaunay
    from open3d.open3d import geometry, utility

    u = [point[0] for point in points_array]
    v = [point[1] for point in points_array]

    tri = Delaunay(np.array([u,v]).T)
    mesh = geometry.TriangleMesh()
    mesh.vertices = utility.Vector3dVector(points_array)
    mesh.triangles = utility.Vector3iVector(tri.simplices)

    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    return mesh


def downsample_cloud_random(pointcloud, number_of_points):
    '''
        Method used for cloud downsampling by selecting random points.
    :param point_cloud_in: [in] Input point cloud.
    :param number_of_points: [in] Desired number of points.
    :return: [out] Downsampled point cloud.
    '''
    cloud_size = len(pointcloud.points)
    index = np.random.choice(cloud_size, number_of_points, replace=False)
    for i in range(len(index)):
        cloud_point = [pointcloud.points[index[i]][0],
                       pointcloud.points[index[i]][1],
                       pointcloud.points[index[i]][2]]
        pointcloud.points.push_back(cloud_point)
    return pointcloud

def preprocess_point_cloud(pcd, voxel_size):
    '''
        Method used for computing point cloud features.
        @ The FPFH feature is a 33-dimensional vector that describes the local geometric property of a point.
    :param pcd: [in] Open3D point cloud
    :param voxel_size: [in] Treshold for the voxel size.
    :return: [out] Downsampled cloud along with its FPFH features.
    '''
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd#voxel_down_sample(pcd, voxel_size)
    radius_normal = voxel_size * 1.5
    print(":: Estimate normal with search radius %.3f." % radius_normal)

    estimate_normals(pcd_down, KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = compute_fpfh_feature(pcd_down,
                                    KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def downsample_points(points, number_of_points):
    '''
        Method used for cloud downsampling by selecting random points.
    :param point_cloud_in: [in] Input point cloud.
    :param number_of_points: [in] Desired number of points.
    :return: [out] Downsampled point cloud.
    '''
    downsampled_points = []
    if number_of_points < len(points):
        index = np.random.choice(len(points), number_of_points, replace=False)
    else:
        index = np.random.choice(len(points), number_of_points, replace=True)

    for i in range(len(index)):
        point = [points[index[i]][0],
                 points[index[i]][0],
                 points[index[i]][0]]
        downsampled_points.append(point)
    return downsampled_points


def execute_global_registration(source, target, normals_radius, treshold):
    """
        Method used for executing global registration of two clouds
    :param source: source pointcloud
    :param target: target pointcloud
    :param normals_radius: radius for fast point feature histograms estimation
    :param treshold: treshold value for ransac registration
    :return: result - representing a registration result containing transformation matrix, inlier_rmse
    """
    source_fpfh = compute_fpfh_feature(source, KDTreeSearchParamHybrid(radius=normals_radius, max_nn=100))
    target_fpfh = compute_fpfh_feature(target, KDTreeSearchParamHybrid(radius=normals_radius, max_nn=100))
    print(":: RANSAC registration on downsampled point clouds.")
    result = registration_ransac_based_on_feature_matching(
        source, target,
        source_fpfh, target_fpfh,
        treshold,
        TransformationEstimationPointToPoint(False), 4,
        [CorrespondenceCheckerBasedOnEdgeLength(0.9),
         CorrespondenceCheckerBasedOnDistance(treshold)],
        RANSACConvergenceCriteria(4000000, 1000))
    return result


def execute_fast_global_registration(source_down, target_down,
                                     source_fpfh, target_fpfh, treshold):
    '''
        Method used for fast point clouds registration.
    :param source_down: [in] Source downsampled point cloud.
    :param target_down: [in] Target downsampled point cluoud.
    :param source_fpfh: [in] Source fpfh features
    :param target_fpfh: [in] Target fpfh features
    :param treshold:  [in] treshold value for executing fast feature match registration representing max correspondance distance
    :return: [out] Registration result
    '''

    print(":: Apply fast global registration with distance threshold %.3f" % treshold)
    result = registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        FastGlobalRegistrationOption(maximum_correspondence_distance=treshold))
    return result


def refine_registration(source, target, treshold, transformation):
    '''
        Method used for ICP registration, after global registration has been completed.
    :param source [in] source pointcloud
    :param target [in] target pointcloud
    :param treshold [in] The value by wich the search radius will be multiplied.
    :param transformation [in] Initial transformation matrix.
    :return [out] reg_p2p = ICP registration object, containing fitness, error, transformation matrix
    '''
    estimate_normals(source, KDTreeSearchParamKNN())
    estimate_normals(target, KDTreeSearchParamKNN())
    reg_p2p = registration_icp(source, target, treshold, transformation,
                               TransformationEstimationPointToPlane(),
                               ICPConvergenceCriteria(max_iteration=1))

    return reg_p2p


def get_diagonal(pointcloud):
    '''
        Method used for computing coordinate points for the Axis Aligned Bounding Box AABB
    :param cloud: [in] cloud for wich to calculate AABB
    :return: [out] length of the diagonal
    '''
    min_point_AABB = pointcloud.get_min_bound()
    max_point_AABB = pointcloud.get_max_bound()

    return np.sqrt(np.sum((min_point_AABB[0] - max_point_AABB[0]) ** 2 +
                          (min_point_AABB[1] - max_point_AABB[1]) ** 2 +
                          (min_point_AABB[2] - max_point_AABB[2]) ** 2))


def upsample_cloud_kd_tree(pointcloud, number_of_points):
    '''
        Method used for cloud upsampling following the principle of adding a point in the middle of 2 closest neigbors.
    :param point_cloud_in: [in] Input point cloud.
    :param number_of_points: [in] Desired number of points of the point cloud.
    :return: [out] Upsampled point cloud.
    '''
    cloud_size = len(pointcloud.points)
    points_to_be_added_count = number_of_points - cloud_size

    kdtree = KDTreeFlann()
    kdtree.set_geometry(pointcloud)
    closest_neighbor_index = 0
    K = 5

    # iterate through random selected indexes
    for i in range(0, points_to_be_added_count):
        index = np.random.choice(cloud_size, 1, replace=False)
        distance = 1000
        cloud_point = pointcloud.points[index[0]]

        # find nearest neighbors
        (count, pointIdxNKNSearch, pointNKNSquareDistance) = kdtree.search_knn_vector_3d(cloud_point, K)
        if kdtree.nearestKSearch(cloud_point, K, pointIdxNKNSearch, pointNKNSquareDistance) > 0:
            for j in range(len(pointIdxNKNSearch)):
                if pointNKNSquareDistance[j] < distance and pointNKNSquareDistance[j] != 0:
                    distance = pointNKNSquareDistance[j]
                    closest_neighbor_index = j
            # calculate distance between selected point and closest neighbor and add the point in between
            neighbor_point = pointcloud.points[pointIdxNKNSearch[closest_neighbor_index]]
            new_point = [(cloud_point[0] + (cloud_point[0] - neighbor_point[0])/2),
                         (cloud_point[0] + (cloud_point[0] - neighbor_point[0])/2),
                         (cloud_point[0] + (cloud_point[0] - neighbor_point[0])/2)]

            pointcloud.points.push_back(new_point)

    return pointcloud


def normalize_cloud(pointcloud):

    min_point_AABB = pointcloud.get_min_bound()
    pointcloud_points = translate_point_cloud(pointcloud.points, min_point_AABB, action='translate')
    maxv = np.amax(pointcloud_points, axis=0)
    translation_point = maxv
    pointcloud_points = translate_point_cloud(pointcloud_points, translation_point, action='normalize')

    return pointcloud_points


def translate_point_cloud(cloud_points_array, translationPoint_as_array, action):
    '''
        Method used for translating the point cloud by a translation point.
    :param cloud: [in] Point cloud points to be translated
    :param translationPoint: [in] Reference translation point
    :return: [out] Translated point cloud
    '''
    cloud_size = len(cloud_points_array)
    if action == 'translate':
        for i in range(0, cloud_size):
            cloud_points_array[i][0] -= translationPoint_as_array[0]
            cloud_points_array[i][1] -= translationPoint_as_array[1]
            cloud_points_array[i][2] -= translationPoint_as_array[2]
        return cloud_points_array
    elif action == 'normalize':
        for i in range(0, cloud_size):
            cloud_points_array[i][0] /= translationPoint_as_array[0]
            cloud_points_array[i][1] /= translationPoint_as_array[1]
            cloud_points_array[i][2] /= translationPoint_as_array[2]
        return cloud_points_array


def scale_pointcloud(pointcloud, scale):
    '''
        Method used for scaling the primitive point cloud relative to the scene cloud.
    :param primitive: [in] primitive point cloud
    :param object: [in] selected object point cloud from the scene
    :return: [out] referenced pointcloud from the primitive object
    '''

    cloud_size = len(pointcloud.points)
    for i in range(0, cloud_size):
        pointcloud.points[i][0] *= scale
        pointcloud.points[i][1] *= scale
        pointcloud.points[i][2] *= scale
    return pointcloud


def calculate_distance_from(points, origin):
    '''
        Calculates the distance between point and system origin.
    :param point: [in] point
    :return: [out] distance
    '''
    dist_vector = [(points[0] - origin[0]),
                   (points[1] - origin[1]),
                   (points[2] - origin[2])]
    return dist_vector


def normalize_points(points, norm_point):
    ''' Normalization of points related to a specific point given as parameter
      :param cloud: [in] Point cloud points to be normalized
      :param norm_point: [in] Reference normalization  point
      :return: [out] normalized points
      '''
    cloud_size = len(points)
    for i in range(0, cloud_size):
        points[i][0] /= norm_point[0]
        points[i][1] /= norm_point[1]
        points[i][2] /= norm_point[2]
    return points


def translate_points(points, translationPoint_as_array):
    '''
        Method used for translating the point cloud by a translation point.
    :param cloud: [in] Point cloud points to be translated
    :param translationPoint: [in] Reference translation point
    :return: [out] Translated point cloud
    '''
    cloud_size = len(points)
    for i in range(0, cloud_size):
        points[i][0] -= translationPoint_as_array[0]
        points[i][1] -= translationPoint_as_array[1]
        points[i][2] -= translationPoint_as_array[2]
    return points


def scale_points(points, scale):
    '''
        Method used for scaling a point cloud by a treshold scale value.
    :param cloud: [in] Point cloud
    :param scale: [in] Scale by wich to transform the point cloud.
    :return: [out] Scaled point cloud
    '''
    for i in range(0, len(points)):
        points[i][0] *= scale
        points[i][1] *= scale
        points[i][2] *= scale
    return points


def icp_align_clouds(source, target, threshold, show_on_visualizer=False, max_iterations=50):
    from open3d.open3d import registration, visualization

    result = execute_global_registration(source.point_cloud, target.point_cloud, normals_radius=threshold*10, treshold=threshold)
    estimate_normals(cloud=source.point_cloud, search_param=KDTreeSearchParamHybrid(threshold, 30))
    estimate_normals(cloud=target.point_cloud, search_param=KDTreeSearchParamHybrid(threshold, 30))

    if show_on_visualizer:
        vis = visualization.Visualizer()
        vis.create_window("ICP ALIGNMENT", 800, 600)
        vis.add_geometry(source.point_cloud)
        vis.add_geometry(target.point_cloud)
        source.point_cloud.transform(result.transformation)

        for i in range(max_iterations):
            reg_p2l = registration_icp(source.point_cloud, target.point_cloud, threshold,
                                       np.identity(4), TransformationEstimationPointToPoint(),
                                       ICPConvergenceCriteria(max_iteration=1))
            trans_matrix_z = [[reg_p2l.transformation[0][0], reg_p2l.transformation[0][1], 0.0, reg_p2l.transformation[0][3]],
                              [reg_p2l.transformation[1][0], reg_p2l.transformation[1][1], 0.0, reg_p2l.transformation[1][3]],
                              [0.0, 0.0, 1, reg_p2l.transformation[2][3]],
                              [0.0, 0.0, 0.0, 1]]
            source.point_cloud.transform(trans_matrix_z)
            vis.update_geometry()
            vis.poll_events()
            vis.update_renderer()
        vis.run()
        vis.destroy_window()
    else:
        source.point_cloud.transform(result.transformation)
        for i in range(max_iterations):
            reg_p2l = registration_icp(source.point_cloud, target.point_cloud, threshold,
                                       np.identity(4), TransformationEstimationPointToPoint(),
                                       ICPConvergenceCriteria(max_iteration=1))

            source.point_cloud.transform(reg_p2l.transformation)
    return source.point_cloud, target.point_cloud

def compute_point_cloud_control_points(srcPrimitive):
    '''
        Method used for computing the point cloud control points.
        It rotates the pointcloud and searches for points at sharp edges and corners.
    :param srcPrimitive: [in] Object holding the primitive point cloud.
    :return: [out] List of indexes of the control points inside the point cloud.
    '''
    indexes_of_control_points = []
    for x in range(0, 360, 120):
        for y in range(0, 360, 120):
            for z in range(0, 360, 120):
                rotation_point = [x, y, z]
                srcPrimitive.point_cloud.rotate(rotation_point)
                hPointIndex = 0
                lPointIndex = 0
                for e in range(0, len(srcPrimitive.point_cloud.points)):
                    if srcPrimitive.point_cloud.points[e].y > srcPrimitive.point_cloud.points[hPointIndex].y:
                        hPointIndex = e
                        indexes_of_control_points.append(hPointIndex)

                    elif srcPrimitive.point_cloud.points[e].y < srcPrimitive.point_cloud.points[lPointIndex].y:
                        lPointIndex = e
                        indexes_of_control_points.append(lPointIndex)
    indexes_of_control_points = list(set(indexes_of_control_points))
    return indexes_of_control_points


def get_sample_for_direction(prim_point_idx, object_cloud, primitive_object, visualization=None, usePrimitiveNNPoints = False):
    from common.data_processing.kitti_IO_utils import MODELLING_PARAMS
    from common.utils.transformations.geometric_transformations import calculate_distance_from

    primitive_point = primitive_object.point_cloud.xyz[prim_point_idx]
    sample_X = []
    step_counter = 0
    prim_aux = np.copy(primitive_point)
    sample_X_nn_obj = []
    sample_X_nn_prim = []
    radius_search = MODELLING_PARAMS.CAR.RADIUS_SEARCH
    while (step_counter < MODELLING_PARAMS.CAR.STEPS):
        prim_aux += primitive_object.normals.normals[prim_point_idx] * MODELLING_PARAMS.CAR.STEP_SIZE
        # Get current primitive points lidar NN
        nn_indexes_object_cloud = get_nn_indexes(prim_aux,
                                                 object_cloud,
                                                 radius_search)
        # Get current primitive points primitive NN
        nn_indexes_primitive_cloud = get_nn_indexes(prim_aux,
                                                    primitive_object,
                                                    radius_search)
        for index in nn_indexes_object_cloud:
            point = object_cloud.point_cloud.xyz[index]
            normal = object_cloud.normals.normals[index]
            sample_X_nn_obj.append((point, normal))
        cloud_points = [pair[0] for pair in sample_X_nn_obj]
        cloud_normals = [pair[1] for pair in sample_X_nn_obj]
        # ***
        #visualize_samples(cloud_points, cloud_normals, normalize=True, r=0, g=255, b=0)
        if usePrimitiveNNPoints:
            for index in nn_indexes_primitive_cloud:
                point = primitive_object.point_cloud.xyz[index]
                normal = primitive_object.normals.normals[index]
                sample_X_nn_prim.append((point, normal))
        step_counter+=1
        cloud_points = [pair[0] for pair in sample_X_nn_prim]
        cloud_normals = [pair[1] for pair in sample_X_nn_prim]
        # ***
        #visualize_samples(cloud_points, cloud_normals, normalize=True, r=255, g=0, b=0)
        sample = sample_X_nn_prim + sample_X_nn_obj
        from common.visualization.visualizer_parameters import visualize_samples
        if len(cloud_points) > 0:
            concat = np.concatenate((cloud_points, cloud_normals), axis=1)
            no_duplicates_points = unique_rows(concat)
            sample_X.append(no_duplicates_points)

    step_counter = 0
    prim_aux = np.copy(primitive_point)
    sample_X_nn_obj = []
    sample_X_nn_prim = []
    while (step_counter < MODELLING_PARAMS.CAR.STEPS):
        prim_aux -= primitive_object.normals.normals[prim_point_idx] * MODELLING_PARAMS.CAR.STEP_SIZE
        # Get current primitive points lidar NN
        nn_indexes_object_cloud = get_nn_indexes(prim_aux,
                                                 object_cloud,
                                                 radius_search)
        # Get current primitive points primitive NN
        nn_indexes_primitive_cloud = get_nn_indexes(prim_aux,
                                                    primitive_object,
                                                    radius_search)
        for index in nn_indexes_object_cloud:
            point = object_cloud.point_cloud.xyz[index]
            normal = object_cloud.normals.normals[index]
            sample_X_nn_obj.append((point, normal))
        cloud_points = [pair[0] for pair in sample_X_nn_obj]
        cloud_normals = [pair[1] for pair in sample_X_nn_obj]
        # ***
        #visualize_samples(cloud_points, cloud_normals, normalize=True, r=0, g=255, b=0)
        if usePrimitiveNNPoints:
            for index in nn_indexes_primitive_cloud:
                point = primitive_object.point_cloud.xyz[index]
                normal = primitive_object.normals.normals[index]
                sample_X_nn_prim.append((point, normal))
        step_counter += 1
    cloud_points = [pair[0] for pair in sample_X_nn_prim]
    cloud_normals = [pair[1] for pair in sample_X_nn_prim]
    # ***
    #visualize_samples(cloud_points, cloud_normals, normalize=True, r=255, g=0, b=0)
    sample = sample_X_nn_prim + sample_X_nn_obj
    cloud_points = [pair[0] for pair in sample]
    cloud_normals = [pair[1] for pair in sample]
    if len(cloud_points) > 0:
        concat = np.concatenate((cloud_points, cloud_normals), axis=1)
        no_duplicates_points = unique_rows(concat)
        sample_X.append(no_duplicates_points)
    return sample_X

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def make_samples(object_cloud, primitive_object, primitive_modelled_object, usePrimitivePoints=False, generate_for = 'train'):
    sample_X = []
    sample_Y = []
    primitive_norm_point = []
    primitive_norm_normal = []
    wasVisited = [False for i in range(0, len(primitive_object.point_cloud.points))]
    for i in range(0, primitive_object.cloud_size):
        primitive_point_before = primitive_object.point_cloud.xyz[i]
        primitive_point_after = primitive_modelled_object.point_cloud.xyz[i]
        primitive_norm_normal = primitive_object.normals.normals[i]
        translation_point = np.asarray(calculate_distance_from(primitive_point_before, ModellingParameters.NORMALIZATION_CENTER))


        primitive_point_after *= ModellingParameters.CAR.SCALE
        primitive_norm_point = np.array([primitive_point_after[0] - translation_point[0],
                                        primitive_point_after[1] - translation_point[1],
                                        primitive_point_after[2] - translation_point[2]])


        object_cloud_currentNNs = get_sample_for_direction(i, object_cloud, primitive_object, visualization=None, usePrimitiveNNPoints=usePrimitivePoints)
        object_cloud_currentNNs = [point * ModellingParameters.CAR.SCALE for point in object_cloud_currentNNs]
        primitive_point_beforex = np.array([primitive_point_before[0] - translation_point[0],
                                            primitive_point_before[1] - translation_point[1],
                                            primitive_point_before[2] - translation_point[2]])

        if len(object_cloud_currentNNs) > 0:
            object_cloud_currentNNs = object_cloud_currentNNs[0]
            object_cloud_currentNNS_translated = translate_point_cloud(object_cloud_currentNNs,
                                                                       translation_point,
                                                                       action="translate")
            wasVisited[i] = True
            sample_X.append(object_cloud_currentNNS_translated)
            sample_Y.append([primitive_norm_point[0], primitive_norm_point[1], primitive_norm_point[2]])


    if generate_for.lower() == 'train':
        return sample_X, sample_Y, primitive_point_beforex, primitive_norm_point
    elif generate_for.lower() == 'test':
        return sample_X, wasVisited

def upsample_point_set(sample_collection, points_count):
    """
        Used for upsampling the samples from the primitive to a certain number of points
    :param sample_collection: list of points
    :param points_count: number of points to upsample to
    :return:
    """
    train_X_upsampled = []
    for points_sample in sample_collection:
        io_obs_temp = IoObservation(points_sample)
        io_obs_temp.upsample_cloud_to(points_count)
        train_X_upsampled.append(np.asarray(io_obs_temp.point_cloud.points))

    return train_X_upsampled
