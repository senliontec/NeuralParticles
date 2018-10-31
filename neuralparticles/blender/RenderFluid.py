import sys 
import argparse
import glob
import time
import bpy

# How to call
# blender FluidTransparentRender.blend --background --python RenderFluid.py -- -i "//../path/to/obj/" -o "//../path/to/Render_" -sf 0 -ef -1 -x 1024 -y 720 --gpu --type reference
# blender FluidTransparentRender.blend --background --python RenderFluid.py -- -i "//../path/to/obj/" -o "//../path/to/Render_" -sf 0 -ef -1 -x 1024 -y 720 --gpu --type network --architecture split_pressure

# -----------------------------------------------------------------------------------------------
# Arguments
# get the args passed to blender after "--", all of which are ignored by blender so scripts may receive their own arguments
argv = sys.argv

if "--" not in argv:
    argv = []  # as if no args are passed
else:
    argv = argv[argv.index("--") + 1:]  # get all args after "--"

# When --help or no args are given, print this help
usage_text = (
        "Execute fluid rendering in the background with this scene:"
        "  blender scene.blend --background --python " + __file__ + " -- [options]"
)

parser = argparse.ArgumentParser(description=usage_text)

parser.add_argument("-i", "--input", dest="input_path", type=str, required=True, metavar='DIR', help="This is the path to the folder that contains the '.obj' files. Prefix relative paths with '//'.")
parser.add_argument("-is", "--surface", dest="surface_path", type=str, default=None, metavar='DIR', help="This is the path to the folder that contains the '.obj' files. Prefix relative paths with '//'.")
parser.add_argument("-o", "--output", dest="output_path", type=str, required=True, metavar='FILE', help="This is the path to the file where the rendered video will be saved. Prefix relative paths with '//'.")
parser.add_argument("-sf", "--start_frame", type=int, default=0, help="Overwrite start frame of rendering.")
parser.add_argument("-ef", "--end_frame", type=int, default=-1, help="Overwrite end frame of rendering.")
parser.add_argument("-fs", "--frame_step", type=int, default=1, help="Overwrite frame step of rendering.")
parser.add_argument("-x", "--x_resolution", type=int, default=400, help="Resolution in x direction.")
parser.add_argument("-y", "--y_resolution", type=int, default=400, help="Resolution in y direction.")
parser.add_argument("-g", "--gpu", action="store_true", help="select GPU devices automatically if available")
parser.add_argument("-t", "--type", choices=["reference", "network", "naive"], required=True)
parser.add_argument("-a", "--architecture", choices=["split_pressure", "total_pressure", "vel", "split_pressure_vae", "split_pressure_timeconv_enc", "split_pressure_fully_rec"])
parser.add_argument("-ot", "--output_type", choices=["PNG", "AVI_JPEG", "AVI_RAW"], default="PNG")
parser.add_argument("--cinematic", action="store_true", help="Cinematic camera settings.") # optimized for 1024x576


args = parser.parse_args(argv)  # In this example we wont use the args



# -----------------------------------------------------------------------------------------------
def main():
    if not argv:
        parser.print_help()
        return

    start_frame = args.start_frame
    end_frame = args.end_frame
    if end_frame == -1:
        input_path = args.input_path.replace("//", "./", 1)
        end_frame = len(glob.glob1(input_path,"*.bobj.gz")) - 1

    # Scene
    scene = bpy.data.scenes["Scene"]

    # Hardware
    if bpy.app.version < (2, 78, 0):
        sysp = bpy.context.user_preferences.system
        # Search for device type
        # If possible select GPU
        if args.gpu:
            for devt_entry in sysp.bl_rna.properties['compute_device_type'].enum_items.keys():
                if devt_entry == "CUDA" or devt_entry == "OPENCL":
                    sysp.compute_device_type = devt_entry
                    #sysp.compute_device = "CUDA_0"
                    break
        # Set the remaining settings, depending on device type
        devt = sysp.compute_device_type
        dev = sysp.compute_device
    else:
        sysp = bpy.context.user_preferences.addons['cycles'].preferences
        if args.gpu:
            for devt_entry in sysp.get_device_types(0):
                if devt_entry[0] == "CUDA" or devt_entry[0] == "OPENCL":
                    sysp.compute_device_type = devt_entry[0]
                    sysp.devices[0].use = True
                    break
        devt = sysp.compute_device_type
        dev = scene.cycles.device

    # Render Settings
    scene.render.image_settings.file_format = args.output_type
    scene.render.filepath = args.output_path
    scene.frame_start = start_frame
    scene.frame_end = end_frame
    scene.frame_step = args.frame_step

    scene.frame_set(start_frame)

    # Cycle Settings
    scene.render.engine = 'CYCLES'
    scene.cycles.min_bounces = 1
    scene.cycles.max_bounces = 8
    if dev == "CPU":
        scene.render.tile_x = 16
        scene.render.tile_y = 16
    else:
        scene.render.tile_x = 256
        scene.render.tile_y = 256
    print("Cycles")
    print("\tDevice: {}".format(dev))
    print("\tDevice Type: {}".format(devt))
    print("\tBounces: {}-{}".format(scene.cycles.min_bounces, scene.cycles.max_bounces))
    print("\tTiles: {}x{}".format(scene.render.tile_x, scene.render.tile_y))

    # Quality Settings
    scene.render.resolution_x = args.x_resolution
    scene.render.resolution_y = args.y_resolution
    scene.render.resolution_percentage = 100
    #scene.render.use_antialiasing = True
    print("Quality Settings")
    print("\tOutput: {}".format(scene.render.filepath))
    print("\tSequence: {}-{}".format(scene.frame_start, scene.frame_end))
    print("\tResolution: {}x{}".format(scene.render.resolution_x, scene.render.resolution_y))

    # Fluid Object Input
    if args.surface_path:
        bpy.data.objects["Surface"].hide_render = False
        bpy.data.objects["Surface"].modifiers["Fluidsim"].settings.filepath = args.surface_path
        print("Fluid Input Path: {}".format(bpy.data.objects["Surface"].modifiers["Fluidsim"].settings.filepath))

    bpy.data.objects["Foam"].modifiers["Fluidsim"].settings.filepath = args.input_path
    print("Fluid Input Path: {}".format(bpy.data.objects["Foam"].modifiers["Fluidsim"].settings.filepath))

    background_color_scale = 1.0
    '''if args.architecture == "vel":
        background_color_scale = 0.88
    elif args.architecture == "total_pressure":
        background_color_scale = 1.0
    elif args.architecture == "split_pressure":
        background_color_scale = 1.2
    elif args.architecture == "split_pressure_vae":
        background_color_scale = 0.95
    elif args.architecture == "split_pressure_dynamic":
        background_color_scale = 1.1'''

    # Background Material
    if args.type == "network":
        color = [0.6,0.7,0.8] # blueish
    if args.type == "reference":
        color = [0.8,0.7,0.6] # brownish
    if args.type == "naive":
        color = [1.0,0.72,0.95] # rosa
    color = [1.0,1.0,1.0]
    
    def change_material_color(mat_name):
        # Plane 0
        plane_mat = bpy.data.materials[mat_name]
        # get the nodes
        plane_nodes = plane_mat.node_tree.nodes
        # get some specific node
        plane_diffuse = plane_nodes.get("Diffuse BSDF")
        for i in range(3):
            plane_diffuse.inputs[0].default_value[i] = color[i]

    change_material_color("Material.002")
    change_material_color("Material.004")

    # Setup Camera
    if args.cinematic:
        bpy.data.cameras["Camera"].lens = 30.0
        bpy.data.objects["Camera"].location[0] = -0.67
        bpy.data.objects["Camera"].location[1] = -1.5
        bpy.data.objects["Camera"].location[2] = 1.17
        bpy.data.objects["Empty"].location[0] = 0.05
        bpy.data.objects["Empty"].location[1] = 0.0
        bpy.data.objects["Empty"].location[2] = 0.25

    # Render
    start = time.time()
    bpy.ops.render.render( animation=True, scene="Scene", write_still=True )
    end = time.time()
    print("Render Duration: {:.3f}s".format(end - start))

# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    