import numpy as np
from EMS.EMS_recovery import EMS_recovery
from EMS.utilities import read_ply
from sklearn.cluster import DBSCAN
import pyvista
import open3d as o3d

def hierarchical_ems(
    point,
    OutlierRatio=0.9,           # prior outlier probability [0, 1) (default: 0.1)
    MaxIterationEM=20,           # maximum number of EM iterations (default: 20)
    ToleranceEM=1e-3,            # absolute tolerance of EM (default: 1e-3)
    RelativeToleranceEM=2e-1,    # relative tolerance of EM (default: 1e-1)
    MaxOptiIterations=2,         # maximum number of optimization iterations per M (default: 2)
    Sigma=0.3,                   # initial sigma^2 (default: 0 - auto generate)
    MaxiSwitch=2,                # maximum number of switches allowed (default: 2)
    AdaptiveUpperBound=True,    # Introduce adaptive upper bound to restrict the volume of SQ (default: false)
    Rescale=False,                # normalize the input point cloud (default: true)
    MaxLayer=5,                  # maximum depth
    Eps=1.7,                    # IMPORTANT: varies based on the size of the input pointcoud (DBScan parameter)
    MinPoints=60,               # DBScan parameter required minimum points
):

    point_seg = {key: [] for key in list(range(0, MaxLayer+1))}
    point_outlier = {key: [] for key in list(range(0, MaxLayer+1))}
    point_seg[0] = [point]
    list_quadrics = []
    quadric_count = 1
    for h in range(MaxLayer):
        for c in range(len(point_seg[h])):
            print(f"Counting number of generated quadrics: {quadric_count}")
            quadric_count += 1
            x_raw, p_raw = EMS_recovery(
                point_seg[h][c],
                OutlierRatio,
                MaxIterationEM,
                ToleranceEM,
                RelativeToleranceEM,
                MaxOptiIterations,
                Sigma,
                MaxiSwitch,
                AdaptiveUpperBound,
                Rescale,
            )
            point_previous = point_seg[h][c]
            list_quadrics.append(x_raw)
            outlier = point_seg[h][c][p_raw < 0.1, :]
            point_seg[h][c] = point_seg[h][c][p_raw > 0.1, :]
            if np.sum(p_raw) < (0.8 * len(point_previous)):
                clustering = DBSCAN(eps=Eps, min_samples=MinPoints).fit(outlier)
                labels = list(set(clustering.labels_))
                labels = [item for item in labels if item >= 0]
                if len(labels) >= 1:
                    for i in range(len(labels)):
                        point_seg[h + 1].append(outlier[clustering.labels_ == i])
                point_outlier[h].append(outlier[clustering.labels_ == -1])
            else:
                point_outlier[h].append(outlier)
    return point_seg, point_outlier, list_quadrics


# Load pointcloud 
point_cloud = o3d.io.read_point_cloud("EMS-superquadric_fitting/MATLAB/example_scripts/data/multi_superquadrics/dog.ply")
point_cloud = np.asarray(point_cloud.points)
point_seg, point_outlier, list_quadrics = hierarchical_ems(point_cloud, Eps=0.1)
# -----------    Plot multiquadric figure --------------
plotter = pyvista.Plotter()
for quadric in list_quadrics:
    grid = quadric.showSuperquadric(arclength=0.2)
    plotter.add_mesh(grid, color=np.random.rand(3), opacity=0.8)
    
plotter.add_points(point_cloud, color='g', point_size=2)
plotter.add_points(point_seg[0][0], color='r', point_size=3)
plotter.show()
