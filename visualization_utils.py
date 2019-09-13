import numpy as np
from open3d.open3d import geometry, utility

def draw_bbox_open3d(cloud_points_array):
    cloud = geometry.PointCloud()
    cloud.points = utility.Vector3dVector(np.asarray(cloud_points_array))
    minBound = cloud.get_min_bound()
    maxBound = cloud.get_max_bound()

    box_center = np.array([maxBound[0] - minBound[0], maxBound[1] - minBound[1], maxBound[2] - minBound[2]])
    points = [[minBound[0], maxBound[1], minBound[2]], [maxBound[0], maxBound[1], minBound[2]],
              [maxBound[0], minBound[1], minBound[2]], [minBound[0], minBound[1], minBound[2]],
              [minBound[0], minBound[1], maxBound[2]], [maxBound[0], minBound[1], maxBound[2]],
              [maxBound[0], maxBound[1], maxBound[2]], [minBound[0], maxBound[1], maxBound[2]],
              [0, 0, 0], [box_center[0], box_center[1], box_center[2]]]
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [0, 7],
             [1, 6],
             [2, 5],
             [3, 4],
             [4, 5], [4, 7],
             [5, 6], [6, 7]]

    colors = [[0, 255, 0] for i in range(0, len(lines))]
    line_set = geometry.LineSet()
    line_set.points = utility.Vector3dVector(points)
    line_set.lines = utility.Vector2iVector(lines)
    line_set.colors = utility.Vector3dVector(colors)
    return line_set