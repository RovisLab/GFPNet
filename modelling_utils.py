from open3d.open3d import geometry
import numpy as np

def find_primitive_points_dependencies(primitive_cloud):
    """
        Method used for computing the point dependencies from inside a point cloud.
        It takes 1 point and finds it's closest 2 neighbors, and keeps them in a structure.
    :param primitive_points: [in] points representing the primitive cloud.
    :return: [out] Sets the dependencies inside [allPrimitivePointsNeighboursDependinces] list.
    """
    from io_primitive import NEXT_PREV_POINT_DEP
    from open3d.open3d import geometry

    dependencies_list = []
    K = 2
    tree = geometry.KDTreeFlann()
    tree.set_geometry(primitive_cloud)
    points = np.asarray(primitive_cloud.points)
    for i in range(0, len(points)):
        search_point = points[i]
        (count, pointIdxNKNsearch, pointNKNSquaredDistances) = tree.search_knn_vector_3d(search_point, K)
        if count > 0:
            temp = NEXT_PREV_POINT_DEP(
                primitive_cloud.points[pointIdxNKNsearch[0]],
                primitive_cloud.points[pointIdxNKNsearch[1]],
                pointIdxNKNsearch[0],
                pointIdxNKNsearch[1])
            dependencies_list.append(temp)

    return dependencies_list


def compute_econt(point1, point2, point_cloud):
    """
        Computes the contour energy between two points inside a point cloud.
    :param point1: [in] First point
    :param point2: [in] Second point
    :param point_cloud: [in] Source point cloud
    :return: [out] E_cont value
    """
    # compute the contour energy
    temp = (point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2
    econt = (euclidian_distance(point_cloud) - np.sqrt(temp)) ** 2
    return econt


def compute_ecurv(point1, point2, point3):
    """
        Computes the curvature energy of three points.
    :param point1: [in] First point
    :param point2: [in] Middle point
    :param point3: [in] Last point
    :return: [out] Curvature energy value
    """
    ecurv = ((point1[0] - (2 * point2[0]) + point3[0]) ** 2) + \
            ((point1[1] - (2 * point2[1]) + point3[1]) ** 2) + \
            ((point1[2] - (2 * point2[2]) + point3[2]) ** 2)
    return ecurv

def euclidian_distance_2points(pointA, pointB):
    """Euclidian distance 2 points"""
    return np.linalg.norm(pointA - pointB)


def euclidian_distance(point_cloud):
    """ Computes euclidian distance between points in pointcloud
    :param point_cloud:
    :return:
    """
    distance = 0
    max_len = len(point_cloud.points)
    assert max_len > 0  # cloud has points
    for i in range(0, len(point_cloud.points)):
        if i == len(point_cloud.points) - 1:
            distance += np.sqrt(
                (point_cloud.points[max_len - 1][0] - point_cloud.points[0][0]) ** 2 +
                (point_cloud.points[max_len - 1][1] - point_cloud.points[0][1]) ** 2 +
                (point_cloud.points[max_len - 1][2] - point_cloud.points[0][2]) ** 2)
        else:
            distance += np.sqrt(
                (point_cloud.points[i][0] - point_cloud.points[i + 1][0]) ** 2 +
                (point_cloud.points[i][1] - point_cloud.points[i + 1][1]) ** 2 +
                (point_cloud.points[i][2] - point_cloud.points[i + 1][2]) ** 2)

    distance = distance / max_len
    return distance


def number_of_neighbors(ptCheckedPoint, observation_cloud, radius, returnWhat="count"):
    """
        Computes the number of neighbors for a certain point in the point cloud, given a radius.
    :param ptCheckedPoint: [in] Checked point
    :param objectROI: [in] Point Cloud where to search
    :param radius: [in] Radius to search
    :return: [out] Depends on parameter
    """
    from open3d.open3d import geometry

    tree = geometry.KDTreeFlann()
    tree.set_geometry(observation_cloud)
    (count, pointIdxRadiusSearch, pointSquareDistances) = tree.search_radius_vector_3d(ptCheckedPoint, radius)
    if returnWhat == 'count':
        return count
    elif returnWhat == 'indexes':
        return pointIdxRadiusSearch


def dist2point(point1, point2):
    """
        Method used for calculating the euclidian distance between two points.
    :param point1: [in] First point
    :param point2: [in] Second point
    :return: [out] Distance between points
    """
    dist = np.sqrt(
        ((point2[0] - point1[0]) ** 2) +
        ((point2[1] - point1[1]) ** 2) +
        ((point2[2] - point1[2]) ** 2)
    )
    return dist


def active_contour_modelling(srcPrimitive, objectROI, search_radius, steps, step_dist, visualizer):
    """
        Method used for modelling the point cloud acording to the Active Contours principle.
    :param srcPrimitive: [in] Object containing the primitive point cloud.
    :param objectROI: [in] Object containing the target, extracted object from scene point cloud.
    :param search_radius: [in] search raadius
    :param step: [in] Max step value by wich the points inside the primitive cloud can be moved along their normal direction.
    :param step_dir: [in] Step added to the point coordinates, by each iteration
    :param visualizer: [in] Visualizer for viewing live modellation
    :return:
    """
    import time
    from io_primitive import PPoint
    from open3d.open3d import visualization

    TIME_START = time.time()
    primitive_points_list = [PPoint(idx=i,
                                    isModified=False,
                                    x=srcPrimitive.point_cloud.points[i][0],
                                    y=srcPrimitive.point_cloud.points[i][1],
                                    z=srcPrimitive.point_cloud.points[i][2],
                                    f_eng=1000,
                                    neighborsCount=0,
                                    isControlPoint=False) for i in range(0, len(srcPrimitive.point_cloud.points))]

    total_energy_temp = 0
    treshold = 0.3
    number_of_iterations = 1
    srcPrimitive.allPrimitivePointsNeighboursDependinces = find_primitive_points_dependencies(srcPrimitive.point_cloud)
    for iteration in range(number_of_iterations):
        count = 0
        for i in range(0, srcPrimitive.cloud_size):
            visualizer.update_geometry()
            visualizer.poll_events()
            visualizer.update_renderer()
            if not srcPrimitive.primitiveModelledVertices[i].isModified and srcPrimitive.primitiveModelledVertices[i].isControlPoint:
                total_energy_temp = 0
                e_curv_temp = compute_ecurv(
                    srcPrimitive.point_cloud.points[srcPrimitive.allPrimitivePointsNeighboursDependinces[i].prevPointID],
                    srcPrimitive.point_cloud.points[i],
                    srcPrimitive.point_cloud.points[srcPrimitive.allPrimitivePointsNeighboursDependinces[i].nextPointID])

                th = 1 * 10**(-10)
                if e_curv_temp > th:
                    alpha = 0.1
                    beta = 0.3
                    gama = 0.8
                else:
                    alpha = 0.1
                    beta = 0.3
                    gama = 0.6

                functional_energy = alpha * compute_econt(srcPrimitive.point_cloud.points[i],
                                                          srcPrimitive.point_cloud.points[
                                                              srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                  i].nextPointID],
                                                          srcPrimitive.point_cloud) + \
                                    beta * compute_ecurv(srcPrimitive.point_cloud.points[i],
                                                         srcPrimitive.point_cloud.points[
                                                             srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                 i].nextPointID],
                                                         srcPrimitive.point_cloud.points[
                                                             srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                 srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                     i].nextPointID].nextPointID]) - \
                                    gama * number_of_neighbors(srcPrimitive.point_cloud.points[i],
                                                               objectROI.point_cloud, search_radius)

                point_NEG_dir = [srcPrimitive.point_cloud.points[i][0],
                                 srcPrimitive.point_cloud.points[i][1],
                                 srcPrimitive.point_cloud.points[i][2]]

                point_POS_dir = [srcPrimitive.point_cloud.points[i][0],
                                 srcPrimitive.point_cloud.points[i][1],
                                 srcPrimitive.point_cloud.points[i][2]]

                temp_pos_dir = point_POS_dir
                temp_neg_dir = point_NEG_dir

                for iteration_step in np.arange(step_dist, steps, step_dist):
                    point_POS_dir[0] = point_POS_dir[0] + srcPrimitive.point_cloud.normals[i][0] * iteration_step
                    point_POS_dir[1] = point_POS_dir[1] + srcPrimitive.point_cloud.normals[i][1] * iteration_step
                    point_POS_dir[2] = point_POS_dir[2] + srcPrimitive.point_cloud.normals[i][2] * iteration_step

                    point_NEG_dir[0] = point_NEG_dir[0] - srcPrimitive.point_cloud.normals[i][0] * iteration_step
                    point_NEG_dir[1] = point_NEG_dir[1] - srcPrimitive.point_cloud.normals[i][1] * iteration_step
                    point_NEG_dir[2] = point_NEG_dir[2] - srcPrimitive.point_cloud.normals[i][2] * iteration_step


                    functional_energy_POS = alpha * compute_econt(point_POS_dir,
                                                                   srcPrimitive.point_cloud.points[
                                                                       srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                           i].nextPointID],
                                                                   srcPrimitive.point_cloud) + \
                                             beta * compute_ecurv(point_POS_dir,
                                                                  srcPrimitive.point_cloud.points[
                                                                      srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                          i].nextPointID],
                                                                  srcPrimitive.point_cloud.points[
                                                                      srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                          srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                              i].nextPointID].nextPointID]) - \
                                             gama * number_of_neighbors(point_POS_dir,
                                                                        objectROI.point_cloud, search_radius)

                    functional_energy_NEG = alpha * compute_econt(point_NEG_dir,
                                                                  srcPrimitive.point_cloud.points[
                                                                      srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                          i].nextPointID],
                                                                  srcPrimitive.point_cloud) + \
                                            beta * compute_ecurv(point_NEG_dir,
                                                                 srcPrimitive.point_cloud.points[
                                                                     srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                         i].nextPointID],
                                                                 srcPrimitive.point_cloud.points[
                                                                     srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                         srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                             i].nextPointID].nextPointID]) - \
                                            gama * number_of_neighbors(point_NEG_dir,
                                                                       objectROI.point_cloud, search_radius)

                    POS_neigh = number_of_neighbors(point_POS_dir, objectROI.point_cloud, search_radius)
                    NEG_neigh = number_of_neighbors(point_NEG_dir, objectROI.point_cloud, search_radius)

                    POS_dist_to_init_pos = dist2point(point_POS_dir, srcPrimitive.point_cloud.points[i])
                    NEG_dist_to_init_pos = dist2point(point_NEG_dir, srcPrimitive.point_cloud.points[i])

                    if functional_energy_POS < primitive_points_list[i].functional_energy and \
                            POS_dist_to_init_pos < treshold and \
                            POS_neigh > primitive_points_list[i].nOfNeighbors:
                        functional_energy = functional_energy_POS
                        primitive_points_list[i].functional_energy = functional_energy_POS
                        primitive_points_list[i].index = i
                        primitive_points_list[i].x = point_POS_dir[0]
                        primitive_points_list[i].y = point_POS_dir[1]
                        primitive_points_list[i].z = point_POS_dir[2]
                        primitive_points_list[i].nOfNeighbors = POS_neigh
                        primitive_points_list[i].isModified = True

                        temp_pos_dir[0] += srcPrimitive.point_cloud.normals[i][0] * iteration_step
                        temp_pos_dir[1] += srcPrimitive.point_cloud.normals[i][1] * iteration_step
                        temp_pos_dir[2] += srcPrimitive.point_cloud.normals[i][2] * iteration_step

                        if number_of_neighbors(temp_pos_dir, objectROI.point_cloud, search_radius) > POS_neigh:
                            functional_energy_POS = alpha * compute_econt(temp_pos_dir,
                                                                          srcPrimitive.point_cloud.points[
                                                                              srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                                  i].nextPointID],
                                                                          srcPrimitive.point_cloud) + \
                                                    beta * compute_ecurv(temp_pos_dir,
                                                                         srcPrimitive.point_cloud.points[
                                                                             srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                                 i].nextPointID],
                                                                         srcPrimitive.point_cloud.points[
                                                                             srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                                 srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                                     i].nextPointID].nextPointID]) - \
                                                    gama * number_of_neighbors(temp_pos_dir,
                                                                               objectROI.point_cloud, search_radius)
                            primitive_points_list[i].functional_energy = functional_energy_POS
                            primitive_points_list[i].x = temp_pos_dir[0]
                            primitive_points_list[i].y = temp_pos_dir[1]
                            primitive_points_list[i].z = temp_pos_dir[2]
                            primitive_points_list[i].nOfNeighbors = number_of_neighbors(temp_pos_dir, objectROI.point_cloud, search_radius)
                            break

                    elif functional_energy_NEG < primitive_points_list[i].functional_energy and \
                            NEG_dist_to_init_pos < treshold and \
                            NEG_neigh > primitive_points_list[i].nOfNeighbors:

                        functional_energy = functional_energy_NEG
                        primitive_points_list[i].functional_energy = functional_energy_NEG
                        primitive_points_list[i].index = i
                        primitive_points_list[i].x = point_NEG_dir[0]
                        primitive_points_list[i].y = point_NEG_dir[1]
                        primitive_points_list[i].z = point_NEG_dir[2]
                        primitive_points_list[i].nOfNeighbors = NEG_neigh
                        primitive_points_list[i].isModified = True

                        temp_neg_dir[0] -= srcPrimitive.point_cloud.normals[i][0] * iteration_step
                        temp_neg_dir[1] -= srcPrimitive.point_cloud.normals[i][1] * iteration_step
                        temp_neg_dir[2] -= srcPrimitive.point_cloud.normals[i][2] * iteration_step

                        if number_of_neighbors(temp_neg_dir, objectROI.point_cloud, search_radius) > NEG_neigh:
                            functional_energy_NEG = alpha * compute_econt(temp_neg_dir,
                                                                          srcPrimitive.point_cloud.points[
                                                                              srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                                  i].nextPointID],
                                                                          srcPrimitive.point_cloud) + \
                                                    beta * compute_ecurv(temp_neg_dir,
                                                                         srcPrimitive.point_cloud.points[
                                                                             srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                                 i].nextPointID],
                                                                         srcPrimitive.point_cloud.points[
                                                                             srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                                 srcPrimitive.allPrimitivePointsNeighboursDependinces[
                                                                                     i].nextPointID].nextPointID]) - \
                                                    gama * number_of_neighbors(temp_neg_dir,
                                                                               objectROI.point_cloud, search_radius)
                            primitive_points_list[i].functional_energy = functional_energy_NEG
                            primitive_points_list[i].index = i
                            primitive_points_list[i].x = temp_neg_dir[0]
                            primitive_points_list[i].y = temp_neg_dir[1]
                            primitive_points_list[i].z = temp_neg_dir[2]
                            primitive_points_list[i].nOfNeighbors = number_of_neighbors(temp_neg_dir, objectROI.point_cloud, search_radius)
                            primitive_points_list[i].isModified = True
                            break

                total_energy_temp += functional_energy
                srcPrimitive.primitiveModelledVertices[i].isModified = True
                if primitive_points_list[i].isModified:
                    count += 1
                    transform_neighbor_points_for(i,
                                                  [primitive_points_list[i].x,
                                                   primitive_points_list[i].y,
                                                   primitive_points_list[i].z],
                                                  srcPrimitive,
                                                  visualizer)

        print('For iteration {0} - {1} points modified || Energy = {2}'.format(iteration, count, total_energy_temp))
        TIME_FINISH = time.time()
        print('*********************TIME*********************')
        print('Time for {0} : {1}'.format(iteration, TIME_FINISH - TIME_START))


def transform_neighbor_points_for(original_point_index,
                                  best_point_position,
                                  primitive, visualizer):
    from parameters import ModellingParameters
    from io_primitive import PRIMITIVE_NEIGHBOR_POINT

    neighbors = list()
    affected_area = euclidian_distance_2points(primitive.point_cloud.points[original_point_index],
                                               best_point_position) * ModellingParameters.CAR.MODELLING_AFFECTED_AREA_FACTOR
    dx = primitive.point_cloud.points[original_point_index][0] - best_point_position[0]
    dy = primitive.point_cloud.points[original_point_index][1] - best_point_position[1]
    dz = primitive.point_cloud.points[original_point_index][2] - best_point_position[2]

    kd_tree = geometry.KDTreeFlann()


    if affected_area > 0:
        kd_tree.set_geometry(primitive.point_cloud)
        (neighbors_count, pointIdxRadiusSearch, pointRadiusSquareDistance) = kd_tree.search_radius_vector_3d(best_point_position, affected_area)

        if neighbors_count > 0:
            for i in range(0, len(pointIdxRadiusSearch)):
                if primitive.point_cloud.points[original_point_index] is not primitive.point_cloud.points[pointIdxRadiusSearch[i]]:
                    tempdx = primitive.point_cloud.points[original_point_index][0] - primitive.point_cloud.points[pointIdxRadiusSearch[i]][0]
                    tempdy = primitive.point_cloud.points[original_point_index][1] - primitive.point_cloud.points[pointIdxRadiusSearch[i]][1]
                    tempdz = primitive.point_cloud.points[original_point_index][2] - primitive.point_cloud.points[pointIdxRadiusSearch[i]][2]

                    temp_dist = np.sqrt((tempdx ** 2) + (tempdy ** 2) + (tempdz ** 2))
                    tempNeighbor = PRIMITIVE_NEIGHBOR_POINT(
                        pointIdxRadiusSearch[i],
                        primitive.point_cloud.points[pointIdxRadiusSearch[i]],
                        temp_dist,
                        tempdx,
                        tempdy,
                        tempdz)
                    neighbors.append(tempNeighbor)
            dist_max = max(neighbor.dist for neighbor in neighbors)

            for neighbor in neighbors:
                if neighbor.dist != 0:
                    neighbor.ptNeighborhood[0] = neighbor.ptNeighborhood[0] - dx * (1 - neighbor.dist / dist_max)
                    neighbor.ptNeighborhood[1] = neighbor.ptNeighborhood[1] - dy * (1 - neighbor.dist / dist_max)
                    neighbor.ptNeighborhood[2] = neighbor.ptNeighborhood[2] - dz * (1 - neighbor.dist / dist_max)
                    primitive.point_cloud.points[neighbor.position_in_primitive_vect] = neighbor.ptNeighborhood
                    primitive.primitiveModelledVertices[neighbor.position_in_primitive_vect].isModified = True
                    visualizer.update_geometry()
                    visualizer.poll_events()
                    visualizer.update_renderer()
            primitive.primitiveModelledVertices[original_point_index].isModified = True
            primitive.point_cloud.points[original_point_index] = best_point_position