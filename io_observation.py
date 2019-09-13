from open3d.open3d import *
from transformations_utils import translate_point_cloud, upsample_cloud_kd_tree, downsample_cloud_random
from transformations_utils import compute_centroid
from open3d.open3d import geometry, utility
from transformations_utils import normalize_cloud
from io_utils import read_off_file


class IoObservation:
    def __init__(self, points):
        self.point_cloud = geometry.PointCloud()
        self.point_cloud.points = utility.Vector3dVector(points)
        self.cloud_down = geometry.PointCloud()
        self.cloud_fpfh = registration.RegistrationResult()
        self.centroid = []


    def translate_to_origin(self, point=None):
        """
            Translates object into origin, based on its centroid
        :return: [out] Sets the point cloud translated into origin.
        """
        assert len(self.point_cloud.points) > 0
        if point is None:
            centroid = compute_centroid(self.point_cloud.points)
            translated_points = translate_point_cloud(self.point_cloud.points, centroid, action="translate")
        else:
            translated_points = translate_point_cloud(self.point_cloud.xyz, [0, 0, 0], action="translate")

        self.point_cloud.points = utility.Vector3dVector(translated_points)


    def compute_normals(self, radius, max_nn):
        geometry.estimate_normals(self.point_cloud, geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))


    def load_point_cloud(self, normalize=False):
        """ Method used for reading the off file that holds the pcd points and populate the point_cloud and normal points inside this object."""
        try:
            try:
                file_extension = (str(self.path).rsplit("\\", 1)[1]).rsplit(".", 1)[1]
            except Exception as e:
                print('Exception at path split.', e)
            if file_extension == "off" or file_extension == "OFF":
                    if normalize:
                        points_array, normals_array = read_off_file(self.path)
                        normalized_points = normalize_cloud(points_array)
                        self.point_cloud.points = utility.Vector3dVector(normalized_points)
                    else:
                        points_array, normals_array = read_off_file(self.path)
                        self.point_cloud.points = utility.Vector3dVector(points_array)
            elif file_extension == 'pcd':
                io.read_point_cloud(self.path, self.point_cloud)
            elif file_extension == 'ply':
                self.mesh = io.read_triangle_mesh(self.path)
                self.point_cloud.points = utility.Vector3dVector(self.mesh.vertices)
        except Exception as e:
            print('Exception at reading off file.', e)

    def upsample_cloud_to(self, number_of_points):
        """
            Method used for upsampling the cloud to a certain number of points.
        :param number_of_points: [in] Desired number of points
        :return: [out] Sets the cloud inside this object
        """
        if len(self.point_cloud.points) < number_of_points:
            upsample_cloud_kd_tree(self, number_of_points)
        else:
            self.point_cloud = downsample_cloud_random(self, number_of_points)