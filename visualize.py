import numpy as np

import pyvista as pv
import vtk

import argparse
import sys
import os

import re

def coords(s):
    try:
        x, y, z = map(float, s.split(','))
        return np.array([x,y,z])
    except:
        raise argparse.ArgumentTypeError("Unkown point coordinate format")

def parse_arguments(args):
    argp = argparse.ArgumentParser(
            prog='3D visualizer',
            description='Plots in the same 3D environment the indicated elements by arguments')
    
    argp.add_argument('volume',  nargs = '+', type=str,
                      help='File list of the reconstructions to be plot in the visualization')

    argp.add_argument('-m', '--meshes_dir', type=str,
                      help='Dir of the meshes that will be plot iterative in the visualization')
    
    argp.add_argument('-p', '--points', nargs = '+', type=coords,
                      help = 'Coordinate points to be plotted in the 3D volume')
    
    return argp.parse_args(args)


def load_obj(obj_file):
    """
    Read an obj file for the vtk library
    """
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_file)
    reader.Update()
    obj = reader.GetOutput()
    return obj

def sort_filenames(filenames):
    numbers = list(map(lambda filename: int(re.findall(r'\d+', filename)[-1]),
                        filenames))
    numbered_files = list(zip(numbers, filenames))
    numbered_files.sort(key = lambda x: x[0])
    return list(map(lambda x: x[1], numbered_files))

def main(args):
    """
    Main function for visualizing
    """
    param = parse_arguments(args)
    volume = np.sum(list(map(lambda x: np.load(x), param.volume)), axis=0)


    p = pv.Plotter()
    p.add_title('3D visualization') 
    # Plot the volume
    norm_volume = 256*np.abs(volume)/np.max(np.abs(volume)).astype(np.uint8)
    volume_actor = p.add_volume(norm_volume, cmap='hot', opacity = 'linear')
    volume_shape = norm_volume.shape
    volume_actor.SetScale(2/volume_shape[0], 2/volume_shape[1], 2/volume_shape[2])
    volume_actor.SetPosition(-1,-1,-1)


    # Plot the points
    if param.points is not None:
        p.add_points(np.array(param.points), 
                     render_points_as_spheres=True, 
                     point_size = 30,
                     color='#FFFF00',
                     opacity = 0.8)

    # Plot the meshes
    if param.meshes_dir is not None:
        meshes_files = os.listdir(param.meshes_dir)
        objs_sorted = sort_filenames(meshes_files)
        meshes = list(map(lambda obj: load_obj(param.meshes_dir + obj), objs_sorted))
        def update_geometry(slider_value):
            mesh = meshes[int(round(slider_value))]
            p.add_mesh(mesh, color= '#00FFFF', opacity=0.6, name='geometry_mesh')
        
        p.add_slider_widget(update_geometry,
                            [0, len(objs_sorted) - 1],
                            value = 0,
                            title = 'Mesh selector')

    p.show_grid()
    p.show()


if __name__=='__main__':
    main(sys.argv[1:])