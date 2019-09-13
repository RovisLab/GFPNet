from open3d import *
from transformations_utils import get_diagonal, delaunay_triangulation, upsample_cloud_kd_tree, downsample_cloud_random, normalize_cloud
from io_utils import read_off_file
from parameters import ModellingParameters
from path_utils import PATHS


# Class used for holding data of primitive neighbor point, when calculating it's functional and external energy.


class PPoint:
    def __init__(self, idx, isModified, x, y, z, f_eng, neighborsCount, isControlPoint):
        '''
            Class used for holding a primitive point's properties
        :param idx: point index
        :param isModified: flag TRUE if the point whas modelled using active contours
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :param f_eng: functional energy
        :param neighborsCount: number of neighbors
        :param isControlPoint: flag to show if it's a control point
        '''
        self.index = idx
        self.isModified = isModified
        self.x = x
        self.y = y
        self.z = z
        self.functional_energy = f_eng
        self.nOfNeighbors = neighborsCount
        self.isControlPoint = isControlPoint

    def is_modified(self):
        return self.isModified

    def get_functional_energy(self):
        return self.functional_energy


class PRIMITIVE_NEIGHBOR_POINT():
    def __init__(self, pos_in_primitive_vect, ptNeighborhood, dist,
                 dx,dy,dz):
        self.position_in_primitive_vect = pos_in_primitive_vect
        self.ptNeighborhood = ptNeighborhood
        self.dist = dist
        self.dx = dx
        self.dy = dy
        self.dz = dz


# Used for active contours modelling
class NEXT_PREV_POINT_DEP:
    def __init__(self,
                 ptNextPoint,
                 ptPreviousPoint,
                 nextPointID,
                 prevPointID):
        self.ptNextPoint = ptNextPoint
        self.ptPreviousPoint = ptPreviousPoint
        self.nextPointID = nextPointID
        self.prevPointID = prevPointID


class IoPrimitive:
    def __init__(self, path_to_primitive):
        self.point_cloud = geometry.PointCloud()
        self.aux_cloud = geometry.PointCloud()
        self.primitiveModelledVertices = []
        self.allPrimitivePointsNeighboursDependinces = list()
        self.path = path_to_primitive
        self.filename = str(path_to_primitive).rsplit("\\")[-1:][0].split(".")[0]
        self.cloud_down = geometry.PointCloud()
        self.cloud_fpfh = registration.RegistrationResult()
        self.mesh = geometry.TriangleMesh()
        self.mesh_lines = geometry.LineSet()
        self.control_points_idx = []
        self.primitive_center = [.0, .0, .0]
        self.cloud_size = 0
        self.height = 0
        self.width = 0
        self.scale = 0
        self.count = 0
        self.RADIUS_SEARCH = 0
        self.STEPS = 0
        self.STEP_SIZE = 0
        self.NORMALS_RADIUS = 0


    def get_scale(self):
        return get_diagonal(self.point_cloud)


    def scale_relative_to(self, dest_cloud):

        self.scale = self.get_scale()
        dest_cloud_scale = get_diagonal(dest_cloud)

        scale_ratio = dest_cloud_scale / self.scale * 1.1
        self.scale_with_factor(scale_ratio)

    def reset_points(self):
        self.point_cloud.points = Vector3dVector(self.aux_cloud.points)


    def scale_with_factor(self, factor):
        cloud_size = len(self.point_cloud.points)
        for i in range(0, cloud_size):
            self.point_cloud.points[i][0] *= factor
            self.point_cloud.points[i][1] *= factor
            self.point_cloud.points[i][2] *= factor
        return self.point_cloud


    def align_primitive_on_z_axis(self, target_cloud):
        rotz = [[np.cos(0.05), -np.sin(0.05), 0.0, 0.0],
                [np.sin(0.05), np.cos(0.05), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]

        eval_total = 999
        for alpha in range(0, 360):
            transformation_matrix = [[np.cos(np.deg2rad(1)), -np.sin(np.deg2rad(1)), 0.0, 0.0],
                                     [np.sin(np.deg2rad(1)), np.cos(np.deg2rad(1)), 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]]
            self.point_cloud.transform(transformation_matrix)

            eval = registration.evaluate_registration(self.point_cloud,
                                                      target_cloud, 1,
                                                      transformation_matrix)
            from decimal import Decimal
            if Decimal(eval.inlier_rmse) < Decimal(eval_total):
                eval_total = eval.inlier_rmse
                transformation_matrix = \
                    [[np.cos(np.deg2rad(alpha)), -np.sin(np.deg2rad(alpha)), 0.0, 0.0],
                     [np.sin(np.deg2rad(alpha)), np.cos(np.deg2rad(alpha)), 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]]
                rot_angle_deg = alpha
                print(alpha)
                self.point_cloud.transform(
                    [[np.cos(np.deg2rad(rot_angle_deg)), -np.sin(np.deg2rad(rot_angle_deg)), 0.0, 0.0],
                     [np.sin(np.deg2rad(rot_angle_deg)), np.cos(np.deg2rad(rot_angle_deg)), 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


    def compute_normals(self, max_nn):
        """
            Computes pointcloud normals using the open3d library functionality.
        :param radius: [in] Radius in within to search for.
        :return:
        """
        estimate_normals(self.point_cloud, KDTreeSearchParamHybrid(radius=self.NORMALS_RADIUS, max_nn=max_nn))

    def load_mesh(self, compute_vertex_normals=False):
        if compute_vertex_normals:
            self.mesh = io.read_triangle_mesh(self.path)
            self.mesh.compute_vertex_normals()
            self.mesh.compute_triangle_normals()
        else:
            self.mesh = io.read_triangle_mesh(self.path)
        self.point_cloud.points = Vector3dVector(self.mesh.vertices)

    def get_delaynay_mesh(self, compute_vertex_normals=False):
        if compute_vertex_normals:
            self.mesh = delaunay_triangulation(self.point_cloud.points)
            self.mesh.compute_vertex_normals()
            self.mesh.compute_triangle_normals()
        else:
            self.mesh = delaunay_triangulation(self.point_cloud.points)

    def smoothen_mesh(self, method=None, iterations = 1):
        if method.lower() =='simple':
            self.mesh.filter_smooth_simple(iterations)
        elif method.lower() == 'laplacian':
            self.mesh.filter_smooth_laplacian(iterations)
        elif method.lower() == 'taubin':
            self.mesh.filter_smooth_taubin(iterations)
        elif method.lower() == 'sharpen':
            self.mesh.filter_sharpen(iterations)

    def get_mesh_lines(self, lines_color=[0, 0, 0]):
        mesh = self.mesh
        triangles_list = np.asarray(mesh.triangles)
        points_list = [[[mesh.vertices[idx[0]][0], mesh.vertices[idx[0]][1], mesh.vertices[idx[0]][2]],
                        [mesh.vertices[idx[1]][0], mesh.vertices[idx[1]][1], mesh.vertices[idx[1]][2]],
                        [mesh.vertices[idx[2]][0], mesh.vertices[idx[2]][1], mesh.vertices[idx[2]][2]]] for idx in
                       triangles_list]

        n = 0
        triangle_points = []
        triangle_lines = []
        for tri_point_set in points_list:
            triangle_points.extend(tri_point_set)
            lines = [[n, n + 1],
                     [n + 1, n + 2],
                     [n + 2, n]]
            triangle_lines.extend(lines)
            n += 3
        triangles_lines_set = geometry.LineSet()
        triangles_lines_set.points = utility.Vector3dVector(triangle_points)
        triangles_lines_set.lines = utility.Vector2iVector(triangle_lines)
        triangles_lines_set.colors = utility.Vector3dVector([lines_color for i in range(len(triangle_lines))])
        self.mesh_lines = triangles_lines_set

    def load_primitive(self, normalize=False):
        """ Method used for reading the off file that holds the pcd points and populate the point_cloud and normal points inside this object."""
        try:
            file_extension = (str(self.path).rsplit("\\", 1)[1]).rsplit(".", 1)[1]
            if file_extension.lower() == "off":
                if normalize:
                    points_array, normals_array = read_off_file(self.path)
                    normalized_points = normalize_cloud(points_array)
                    self.point_cloud.points = Vector3dVector(normalized_points)
                else:
                    points_array, normals_array = read_off_file(self.path)
                    self.point_cloud.points = Vector3dVector(points_array)
            elif file_extension == 'pcd':
                io.read_point_cloud(self.path, self.point_cloud)
            elif file_extension == 'ply':
                self.mesh = io.read_triangle_mesh(self.path)
                assert len(self.mesh.vertices) > 0
                self.point_cloud.points = Vector3dVector(self.mesh.vertices)
            self.load_primitive_control_points(set_all=True)
            self.cloud_size = len(self.point_cloud.points)
            self.aux_cloud.points = Vector3dVector(np.copy(self.point_cloud.points))

            self.RADIUS_SEARCH = ModellingParameters.CAR.RADIUS_SEARCH
            self.STEPS = ModellingParameters.CAR.STEPS
            self.STEP_SIZE = ModellingParameters.CAR.STEP_SIZE
            self.NORMALS_RADIUS = self.STEPS * self.RADIUS_SEARCH * 15

        except Exception as e:
            print('Exception at reading off file.', e)


    def load_primitive_control_points(self, set_all = False):
        if set_all:
            for i in range(0, len(self.point_cloud.points)):
                pt = PPoint(i, False,
                            self.point_cloud.points[i][0],
                            self.point_cloud.points[i][1],
                            self.point_cloud.points[i][2],
                            None, 0, True)
                self.primitiveModelledVertices.append(pt)
        else:
            final_path = PATHS.PATH_TO_PRIMITIVES.CAR.root + self.filename + "_cp.txt"  # cp stands for control points
            file = open(final_path, "r")
            assert file is not None
            try:
                lines = file.read().split(" ")
                for i in range(0, len(self.point_cloud.points)):
                    pt = PPoint(i, False,
                                self.point_cloud.points[i][0],
                                self.point_cloud.points[i][1],
                                self.point_cloud.points[i][2],
                                None, 0, True)
                    self.primitiveModelledVertices.append(pt)
                for index in lines:
                    self.control_points_idx.append(index)
                    if str(i) in set(self.control_points_idx):
                        pt.isControlPoint = True
            except:
                print('File may not exist!')


    def upsample_cloud_to(self, number_of_points):
        """
            Method used for upsampling the cloud to a certain number of points.
        :param number_of_points: [in] Desired number of points
        :return: [out] Sets the cloud inside this object
        """
        if len(self.point_cloud.points) < number_of_points:
            upsample_cloud_kd_tree(self.point_cloud, number_of_points)
        else:
            self.point_cloud = downsample_cloud_random(self.point_cloud, number_of_points)
