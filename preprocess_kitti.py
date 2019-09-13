from kitti_utils import align_img_and_pc, load_calib, load_velodyne_points, project_velo_points_in_img, prepare_velo_points
from kitti_dataset import KITTIPaths
IMG_ROOT = KITTIPaths.IMG_ROOT
PC_ROOT = KITTIPaths.PC_ROOT
CALIB_ROOT = KITTIPaths.CALIB_ROOT

for frame in range(0, 7481):
    img_dir = IMG_ROOT + '%06d.png' % frame
    pc_dir = PC_ROOT + '%06d.bin' % frame
    calib_dir = CALIB_ROOT + '%06d.txt' % frame

    points = align_img_and_pc(img_dir, pc_dir, calib_dir)

    output_name = PC_ROOT + frame + '.bin'
    points[:, :4].astype('float32').tofile(output_name)
