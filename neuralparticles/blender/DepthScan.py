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
        "Virtual 3D Scan of a animated mesh:"
        "  blender scene.blend --background --python " + __file__ + " -- [options]"
)

parser = argparse.ArgumentParser(description=usage_text)

parser.add_argument("-o", "--output", dest="output_path", type=str, default=None, required=True, metavar='FILE', help="This is the output path.")
parser.add_argument("-obj", "--object", dest="object", type=str, default=None, help="This is the name of the object (the armature!)")
parser.add_argument("-a", "--animation", dest="animation", type=str, default=None, help="This is the name of the used animation")
parser.add_argument("-x", "--x_resolution", type=int, default=400, help="Resolution in x direction.")
parser.add_argument("-y", "--y_resolution", type=int, default=400, help="Resolution in y direction.")
parser.add_argument("-fs", "--frame_step", type=int, default=1, help="Overwrite frame step of rendering.")
parser.add_argument("-sf", "--start_frame", type=int, default=-1, help="Overwrite start frame of rendering.")
parser.add_argument("-ef", "--end_frame", type=int, default=-1, help="Overwrite end frame of rendering.")
args = parser.parse_args(argv)  # In this example we wont use the args

if not argv:
    parser.print_help()
    exit()

def get_keyframe(action):
    start_t = 1000
    end_t = 0
    for fcu in action.fcurves:
        for keyframe in fcu.keyframe_points:
            x, y = keyframe.co
            if x < start_t:
                start_t = x
            if x > end_t:
                end_t = x
    return int(start_t), int(end_t)

def writeNumpyOBJ(filename, data):
	with open(filename, "w") as f:
		f.writelines("v " + " ".join(str(el) for el in d) + "\n" for d in data[...,:3])

# Scene
scene = bpy.data.scenes["Scene"]

print("USING EEVEE")
scene.render.engine = 'BLENDER_EEVEE'

obj = bpy.data.objects[args.object]
obj.animation_data.action = bpy.data.actions[args.animation]

t_start, t_end = get_keyframe(bpy.data.actions[args.animation])
print(get_keyframe(bpy.data.actions[args.animation]))

# Render Settings
scene.render.image_settings.file_format = "OPEN_EXR"
scene.frame_start = t_start if args.start_frame < 0 else args.start_frame
scene.frame_end = t_end if args.end_frame < 0 else args.end_frame
scene.frame_step = args.frame_step

scene.frame_set(t_start)

# Quality Settings
scene.render.resolution_x = args.x_resolution
scene.render.resolution_y = args.y_resolution
scene.render.resolution_percentage = 100

print("Quality Settings")
print("\tOutput: {}".format(scene.render.filepath))
print("\tSequence: {}-{}".format(scene.frame_start, scene.frame_end))
print("\tResolution: {}x{}".format(scene.render.resolution_x, scene.render.resolution_y))

scene.use_nodes = True

"""
bpy.context.scene.render.use_compositing = True
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

for n in tree.nodes:
    tree.nodes.remove(n)
rl = tree.nodes.new('CompositorNodeRLayers')      

vl = tree.nodes.new('CompositorNodeViewer')   
vl.use_alpha = True
links.new(rl.outputs[0], vl.inputs[0])  # link Renger Image to Viewer Image
links.new(rl.outputs[2], vl.inputs[1])  # link Render Z to Viewer Alpha

#render
bpy.context.scene.render.resolution_percentage = 100 #make sure scene height and width are ok (edit)
bpy.ops.render.render()

#get the pixels and put them into a numpy array
pixels = np.array(bpy.data.images['Viewer Node'].pixels)
print(len(pixels))

width = bpy.context.scene.render.resolution_x 
height = bpy.context.scene.render.resolution_y

print(np.max(pixels))
print(np.min(pixels))
#reshaping into image array 4 channel (rgbz)
image = pixels.reshape(height,width,4)

#depth analysis...
z = image[:,:,3]
zf = z[z<1000] #
print(np.min(zf),np.max(zf))
"""

cam_pos = [[3,4,4]]
cam_data = {}
cam_data['transform'] = []

cam = bpy.data.objects["Camera"]

vf = cam.data.view_frame()
cam_data['near'] = -vf[0][2]
cam_data['width'] = vf[0][0] - vf[2][0]
cam_data['height'] = vf[0][1] - vf[2][1]

for i in range(len(cam_pos)):

    cam.location[0] = cam_pos[i][0]
    cam.location[1] = cam_pos[i][1]
    cam.location[2] = cam_pos[i][2]

    scene.update()

    viewWorld = cam.matrix_world

    cam_data['transform'].append([[x for x in y] for y in viewWorld])

    scene.render.filepath = args.output_path + "%04d/"%i
    start = time.time()
    bpy.ops.render.render(animation=True, scene="Scene", write_still=True)

    data = np.empty((math.ceil((scene.frame_end-scene.frame_start+1)/scene.frame_step), scene.render.resolution_x, scene.render.resolution_y, 1))
    cnt = 0
    for t in range(scene.frame_start, scene.frame_end+1, scene.frame_step):
        raw = bpy.data.images.load(args.output_path + "%04d/%04d.exr"%(i,t))
        data[cnt] = np.reshape(np.array(raw.pixels[:]), (scene.render.resolution_x, scene.render.resolution_y, 4))[...,0:1]
        cnt+=1
    np.savez_compressed(args.output_path + "%04d.npz"%i, data)
    end = time.time()
    print("Render Duration: {:.3f}s".format(end - start))
    
with open(args.output_path + "cam_data.json", 'w') as outfile:
    json.dump(cam_data, outfile)
