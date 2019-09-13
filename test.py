import numpy as np

from open3d.open3d import geometry, utility, visualization
from io_observation import IoObservation
from io_primitive import IoPrimitive
from transformations_utils import delaunay_triangulation, icp_align_clouds
from kitti_utils import load_cloud_and_labels, align_img_and_pc, get_segmented_cloud_points_after_fitting_box, translate_points
from path_utils import PATHS
from visualization_utils import draw_bbox_open3d
from modelling_utils import active_contour_modelling

def custom_visualizer_with_key_callback(primitive, observation):
    def rotate_vis(vis):
        ctr = vis.get_view_control()
        ctr.rotate(1.0, 0.0)
        return False

    def model_clouds(vis):
        srcPrimitive.point_cloud, io_obs.point_cloud = active_contour_modelling(srcPrimitive, io_obs, search_radius=srcPrimitive.RADIUS_SEARCH,
                                                                                steps=srcPrimitive.STEP_SIZE * srcPrimitive.STEPS,
                                                                                step_dist=srcPrimitive.STEP_SIZE, visualizer=vis)
        return False

    vis = visualization.VisualizerWithKeyCallback()
    vis.create_window("--Presenting Window--", 1280, 1024)
    vis.add_geometry(primitive)
    vis.add_geometry(observation)
    #vis.register_animation_callback(rotate_vis)
    vis.register_key_callback(ord("M"), model_clouds)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    print('Cropping original velodyne clouds')
    print('Processing: ')
    full_cloud = geometry.PointCloud()
    seg_cloud = geometry.PointCloud()
    clouds = []

       # DECLARING OBJECTS
    source_primitive_cloud_path = PATHS.PATH_TO_PRIMITIVES.CAR.root + "CarPrimitive_15_500.off"
    srcPrimitive = IoPrimitive(source_primitive_cloud_path)
    srcPrimitive.load_primitive()
    srcPrimitive.compute_normals(30)
    srcPrimitive.point_cloud.normalize_normals()
    srcPrimitive.point_cloud.paint_uniform_color([1, 0, 0])



    frame = 8  # sample id from kitti dataset
    final_clouds = []
    lidar, labels = load_cloud_and_labels(frame)
    img_dir = str(PATHS.KITTI.IMG_ROOT + '%06d.png' % frame)
    pc_dir = str(PATHS.KITTI.PC_ROOT + '%06d.bin' % frame)
    calib_dir = str(PATHS.KITTI.CALIB_ROOT + '%06d.txt' % frame)
    full_cloud_points = align_img_and_pc(img_dir, pc_dir, calib_dir)
    full_cloud.points = utility.Vector3dVector(np.array(full_cloud_points[:, 0:3]))
    full_cloud.colors = utility.Vector3dVector([[0, 0, 0] for i in range(0, len(full_cloud.points))])
    # keep original labels, and work on original labels
    lab_cnt = 0
    for line in labels:
        # Clear cloud to be repopulated
        values = line.split(' ')
        lab_cnt += 1
        [cls, obj_lv_image, occl_state, obs_angl, bb_l, bb_t, bb_r, bb_d, h, w, l, y, z, x, rot] = values
        x = float(x)
        y = float(y)
        z = float(z)
        h = float(h)
        w = float(w)
        l = float(l)
        rot = float(rot)
        # select only cars
        if cls == PATHS.KITTI.CAR_CLASS:
            box = [[y, z, x, h, w, l,rot]]
            extracted_points, box_center = get_segmented_cloud_points_after_fitting_box(box, lidar)
            clouds.append([extracted_points, box_center])
    print('Labels Count : {0}'.format(lab_cnt))
    line_set = []
    box_count = 0
    all_pts = []
    concatenated_cloud = []
    primitive_obs_pairs_list = []
    mesh = geometry.TriangleMesh()
    for point_set, translation_point in clouds:
        # all extracted clouds colored in red
        point_set = translate_points(point_set, translation_point, 'neg')
        io_obs = IoObservation(point_set)
        srcPrimitive.scale_relative_to(io_obs.point_cloud)
        srcPrimitive.align_primitive_on_z_axis(io_obs.point_cloud)
        io_obs.point_cloud.paint_uniform_color([0, 0, 1])
        custom_visualizer_with_key_callback(srcPrimitive.point_cloud, io_obs.point_cloud)
        #icp_align_clouds(srcPrimitive, io_obs, threshold=0.5, show_on_visualizer=True)
        concatenated_cloud.extend(srcPrimitive.point_cloud.points)
        mesh = delaunay_triangulation(concatenated_cloud)
        translated_pts = translate_points(concatenated_cloud, translation_point, "pos")
        all_pts.extend(translated_pts)
        mesh.vertices = utility.Vector3dVector(translated_pts)
        mesh.compute_triangle_normals(True)
        mesh.compute_vertex_normals(True)
        mesh.paint_uniform_color([1, 0, 0])
        seg_cloud.points = utility.Vector3dVector(all_pts)
        line_set = draw_bbox_open3d(mesh.vertices)
        final_clouds.append(line_set)
        final_clouds.append(mesh)
        srcPrimitive.reset_points()
    seg_cloud.colors = utility.Vector3dVector([[0, 255, 0] for i in range(0, len(seg_cloud.points))])
    final_clouds.append(seg_cloud)
    final_clouds.append(full_cloud)
