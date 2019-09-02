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

parser.add_argument("-o", "--output", dest="output_path", type=str, default=None, required=True, metavar='FILE', help="This is the output path.")
parser.add_argument("-obj", "--object", dest="object", type=str, default=None, required=True, help="This is the name of the object (the armature!)")
parser.add_argument("-a", "--animation", dest="animation", type=str, default=None, required=True, help="This is the name of the used animation")
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

def select_child_mesh(obj):
    for c in obj.children:
        if c.type == "MESH":
            c.select_set(state=True)
        select_child_mesh(c)

obj = bpy.data.objects[args.object]
obj.animation_data.action = bpy.data.actions[args.animation]

for ob in bpy.context.selected_objects:
    ob.select_set(state=False)
select_child_mesh(obj)

t_start, t_end = get_keyframe(bpy.data.actions[args.animation])
if args.start_frame >= 0:
    t_start = args.start_frame
if args.end_frame >= 0:
    t_end = args.end_frame
    
print(get_keyframe(bpy.data.actions[args.animation]))
i = 0
for t in range(t_start, t_end+1, args.frame_step):
    bpy.data.scenes["Scene"].frame_current = t
    bpy.ops.export_scene.obj(filepath=args.output_path%i, check_existing=False, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl", use_selection=True, use_animation=False, use_mesh_modifiers=True, use_edges=True, use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True, use_uvs=False, use_materials=False, use_triangles=False, use_nurbs=False, use_vertex_groups=False, use_blen_objects=True, group_by_object=False, group_by_material=False, keep_vertex_order=False, global_scale=1, path_mode='AUTO')
    i+=1