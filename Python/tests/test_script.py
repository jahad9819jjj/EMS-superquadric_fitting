import argparse
import sys
import pyvista
import numpy as np
import open3d as o3d

from EMS.utilities import read_ply
from EMS.EMS_recovery import EMS_recovery

import timeit

def main(argv):

    parser = argparse.ArgumentParser(
        description='Probabilistic Recovery of a superquadric surface from a point cloud file *.ply.')

    parser.add_argument(
        '--path_to_data',
        default='EMS-superquadric_fitting/MATLAB/example_scripts/data/multi_superquadrics/dog.ply',
        help='Path to the directory containing the point cloud file *.ply.'
    )

    parser.add_argument(
        '--visualize',
        action = 'store_true',
        help='Visualize the recoverd superquadric and the input point cloud.'
    )

    parser.add_argument(
        '--runtime',
        action = 'store_true',
        help='Show the runtime.'
    )

    parser.add_argument(
        '--result',
        action = 'store_true',       
        help='Print the recovered superquadric parameter.'
    )

    parser.add_argument(
        '--outlierRatio',
        type = float,
        default = 0.2,       
        help='Set the prior outlier ratio. Default is 0.2.'
    )

    parser.add_argument(
        '--adaptiveUpperBound',
        action = 'store_true',       
        help='Implemet addaptive upper bound to limit the volume of the superquadric.'
    )

    parser.add_argument(
        '--arcLength',
        type = float,
        default = 0.2,       
        help='Set the arclength (resolution) for rendering the superquadric. Default is 0.2.'
    )

    parser.add_argument(
        '--pointSize',
        type = float,
        default = 3,       
        help='Set the point size for plotting the point cloud. Default is 0.2.'
    )

    args = parser.parse_args(argv)
    
    print('----------------------------------------------------')
    print('Loading point cloud from: ', args.path_to_data, '...')
    point_cloud = o3d.io.read_point_cloud(args.path_to_data)
    point = np.asarray(point_cloud.points)
    print('Point cloud loaded.')
    print('----------------------------------------------------')

    # first run to eliminate jit compiling time
    sq_recovered, p = EMS_recovery(point)

    start = timeit.default_timer()
    sq_recovered, p = EMS_recovery(point, 
                                   OutlierRatio=args.outlierRatio, 
                                   AdaptiveUpperBound=args.adaptiveUpperBound
                      )
    stop = timeit.default_timer()
    print('Superquadric Recovered.')
    if args.runtime is True:
        print('Runtime: ', (stop - start) * 1000, 'ms')
    print('----------------------------------------------------')
    
    if args.result is True:
        print('shape =', sq_recovered.shape)
        print('scale =', sq_recovered.scale)
        print('euler =', sq_recovered.euler)
        print('translation =', sq_recovered.translation)
        print('----------------------------------------------------')
    
    if args.visualize is True:
        # fig = mlab.figure(size=(400, 400), bgcolor=(1, 1, 1))
        # showPoints(point, scale_factor=args.pointSize)
        # mlab.show()
        plotter = pyvista.Plotter()
        grid = sq_recovered.showSuperquadric(arclength = args.arcLength)
        plotter.add_mesh(grid, color=np.random.rand(3), opacity=0.8)
        plotter.add_points(point, color='r', point_size=args.pointSize)
        plotter.show()


if __name__ == "__main__":
    main(sys.argv[1:])
