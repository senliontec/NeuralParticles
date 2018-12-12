import bpy
import os
import sys 
import argparse

argv = sys.argv

if "--" not in argv:
    argv = []  # as if no args are passed
else:
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

# When --help or no args are given, print this help
usage_text = (
        "Export animated meshes:"
        "  blender scene.blend --background --python " + __file__ + " -- [options]"
)

parser = argparse.ArgumentParser(description=usage_text)

parser.add_argument("-o", "--output", dest="output_path", type=str, default=None, required=True, metavar='DIR', help="This is the path to the folder where the obj will be exported. Prefix relative paths with '//'.")
parser.add_argument("-a", "--animation", dest="animation", type=str, default=None, help="This is the name of the used animation")
parser.add_argument("-t", "--timesteps", dest="timesteps", type=int, default=1, help="Amount of exported frames.")
args = parser.parse_args(argv)  # In this example we wont use the args

if not argv:
    parser.print_help()
    exit()


bpy.ops.export_scene.obj(filepath="", check_existing=True, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_selection=False, use_animation=False, use_mesh_modifiers=True, use_edges=True, use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True, use_uvs=True, use_materials=True, use_triangles=False, use_nurbs=False, use_vertex_groups=False, use_blen_objects=True, group_by_object=False, group_by_material=False, keep_vertex_order=False, global_scale=1, path_mode='AUTO')