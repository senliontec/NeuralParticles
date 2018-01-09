#
# tool runs helpers
#
import sys
import shutil
import numpy as np

# ======================================================================================================================
# duplicated from main "helpers", not to require manta

def getParam(name, default, paramUsed):
	while( len(paramUsed)<len(sys.argv) ):
		paramUsed.append(0);
	for iter in range(1, len(sys.argv)):
		#if(iter <  len(sys.argv)-1): print("Param %s , used %d, val %s, name %s " %( sys.argv[iter].lower(), paramUsed[iter] , sys.argv[iter+1], name) ); # debug
		if(sys.argv[iter].lower() == name) and (iter+1<len(paramUsed)):
			paramUsed[iter] = paramUsed[iter+1] = 1;
			return sys.argv[iter+1];
	return default;

def getColParam(name, default, paramUsed):
	while( len(paramUsed)<len(sys.argv) ):
		paramUsed.append(0);
	for iter in range(1, len(sys.argv)):
		#if(iter <  len(sys.argv)-1): print("Param %s , used %d, val %s, name %s " %( sys.argv[iter].lower(), paramUsed[iter] , sys.argv[iter+1], name) ); # debug
		if(sys.argv[iter].lower() == name) and (iter+1<len(paramUsed)):
			paramUsed[iter] = paramUsed[iter+1] = 1;
			paramUsed[iter+2] = paramUsed[iter+3] = 1;
			return [ float(sys.argv[iter+1]), float(sys.argv[iter+2]), float(sys.argv[iter+3]) ]
	return default;

def checkUnusedParam(paramUsed, off=0):
	err = False;
	for iter in range(1+off, len(sys.argv)):
		if(paramUsed[iter]==0):
			print("Error: param %d '%s' not used!" % (iter,sys.argv[iter]) );
			err = True;
	if err:
		exit(1);

def backupSources(name):
	#return; # off
	# save scene file
	#shutil.copyfile( sceneSrcFile, '%s_source.py' % (name) )
	sceneFile = sys.argv[0];
	shutil.copyfile( sceneFile, '%s_scene.py' % (name) )

	# save command line call
	callfile = open( ("%s_call.txt"%name), 'w+')
	callfile.write("\n");
	callfile.write(str(" ".join(sys.argv) ) );
	callfile.write("\n\n");
	callfile.close();


def particle_range(arr, start, end):
	for i in range(len(start)):
		arr = arr[np.where((arr[:,i]>=start[i])&(arr[:,i]<=end[i]))]
	return arr

def particle_radius(arr, pos, radius):
	return np.where(np.linalg.norm(np.subtract(arr,pos), axis=1) < radius)

def insert_patch(data, patch, pos, func):
	patch_size = patch.shape[0]//2
	x0=pos[0]-patch_size
	x1=pos[0]+patch_size+1
	y0=pos[1]-patch_size
	y1=pos[1]+patch_size+1

	data[0,y0:y1,x0:x1] = func(data[0,y0:y1,x0:x1], patch)

def extract_particles(data, pos, cnt, constraint, aux_data={}):
	# select the 'cnt'th nearest particles to pos 
	#if constraint != None:
	'''constraint = constraint//2
	x0 = pos[0]-constraint
	x1 = pos[0]+constraint+1
	y0 = pos[1]-constraint
	y1 = pos[1]+constraint+1
	data = particle_range(data, [x0,y0], [x1,y1])
	if len(data) < cnt:
		return None'''
	par_idx = particle_radius(data, pos, constraint)
	par_pos = np.subtract(data[par_idx],pos)/constraint

	par_aux = {}
	for k, v in aux_data.items():
		par_aux[k] = v[par_idx]
	
	if len(par_pos) < cnt:
		par_pos = np.concatenate((par_pos,np.zeros((cnt-len(par_pos),par_pos.shape[-1]))))
		for k, v in par_aux.items():
			par_aux[k] = np.concatenate((v,np.zeros((cnt-len(v),v.shape[-1]))))
	
	rnd_idx = np.arange(len(par_pos))
	np.random.shuffle(rnd_idx)
	rnd_idx = rnd_idx[:cnt]

	par_pos = par_pos[rnd_idx]
	for k, v in par_aux.items():
		par_aux[k] = v[rnd_idx]
	return par_pos, par_aux
	
	#par = data[np.argpartition(np.linalg.norm(np.subtract(data,pos), axis=1), cnt-1)[:cnt]]
	#par = np.subtract(par,pos)
	#return par[np.argsort(np.linalg.norm(par, axis=1))]

def extract_patch(data, pos, patch_size):
	patch_size = patch_size//2
	x0 = pos[0]-patch_size
	x1 = pos[0]+patch_size+1
	y0 = pos[1]-patch_size
	y1 = pos[1]+patch_size+1
	return data[0,y0:y1,x0:x1]

def get_patches(sdf_data, patch_size, dimX, dimY, stride, surface):
	pos = []
	patch_size = patch_size//2
	for y in range(patch_size,dimY-patch_size, stride):
		for x in range(patch_size,dimX-patch_size, stride):
			z = 0
			if(abs(sdf_data[z,y,x,0]) < surface):
				x0 = x-patch_size
				x1 = x+patch_size+1
				y0 = y-patch_size
				y1 = y+patch_size+1

				pos.append([x,y,z])
	return np.array(pos)
