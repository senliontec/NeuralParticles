import bpy
import os
import sys 
import argparse
import time
import json
import math
import numpy as np

argv = sys.argv

if "--" not in argv:
    argv = []  # as if no args are passed
else:
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

# When --help or no args are given, print this help
usage_text = (
        "Export of cam position:"
        "  blender scene.blend --background --python " + __file__ + " -- [options]"
)

parser = argparse.ArgumentParser(description=usage_text)

parser.add_argument("-o", "--output", dest="output_path", type=str, default=None, required=True, metavar='FILE', help="This is the output path.")
parser.add_argument("-c", "--cam", dest="cam", type=str, default="", help="Cam position.")
args = parser.parse_args(argv)  # In this example we wont use the args

if not argv:
    parser.print_help()
    exit()

cam_data = {}
cam_data['transform'] = []

cam = bpy.data.objects["Camera"]

vf = cam.data.view_frame()
cam_data['near'] = -vf[0][2]
cam_data['width'] = vf[0][0] - vf[2][0]
cam_data['height'] = vf[0][1] - vf[2][1]

if args.cam != "":
    cam_pos = args.cam.split(",")

    cam.location[0] = float(cam_pos[0])
    cam.location[1] = float(cam_pos[1])
    cam.location[2] = float(cam_pos[2])

    bpy.data.scenes["Scene"].update()

viewWorld = cam.matrix_world

cam_data['transform'].append([[x for x in y] for y in viewWorld])

with open(args.output_path, 'w') as outfile:
    json.dump(cam_data, outfile)
