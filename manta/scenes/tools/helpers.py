#
# tool runs helpers
#
import sys, warnings
import shutil
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def particle_range(arr, pos, r):
	return np.where(np.all(np.abs(np.subtract(arr,pos)) < r, axis=-1))[0]

def particle_radius(arr, pos, radius):
	return np.where(np.linalg.norm(np.subtract(arr,pos), axis=1) < radius)[0]

def insert_patch(data, patch, pos, func):
	patch_size = patch.shape[0]//2
	x0=int(pos[0])-patch_size
	x1=int(pos[0])+patch_size+1
	y0=int(pos[1])-patch_size
	y1=int(pos[1])+patch_size+1

	data[0,y0:y1,x0:x1] = func(data[0,y0:y1,x0:x1], patch)

def remove_particles(data, pos, constraint, aux_data={}):
	par_idx = particle_radius(data, pos, constraint)
	par_aux = {}
	for k, v in aux_data.items():
		par_aux[k] = np.delete(v, par_idx, axis=0)
		
	return np.delete(data, par_idx, axis=0), par_aux

'''def extract_remove_particles(data, pos, cnt, constraint, aux_data={}):
	par_idx = particle_radius(data, pos, constraint)
	np.random.shuffle(par_idx)
	par_idx = par_idx[:min(cnt,len(par_idx))]

	par_pos = np.subtract(data[par_idx],pos)/constraint
	data = np.delete(data, par_idx, axis=0)
	par_aux = {}
	for k, v in aux_data.items():
		par_aux[k] = v[par_idx]
		v = np.delete(v, par_idx, axis=0)

	if len(par_pos) < cnt:
		par_pos = np.concatenate((par_pos,np.zeros((cnt-len(par_pos),par_pos.shape[-1]))))
		for k, v in par_aux.items():
			par_aux[k] = np.concatenate((v,np.zeros((cnt-len(v),v.shape[-1]))))
	
	return data, par_pos, aux_data, par_aux'''

def extract_particles(data, pos, cnt, constraint, aux_data={}):
	par_idx = particle_radius(data, pos, constraint)
	np.random.shuffle(par_idx)
	par_idx = par_idx[:min(cnt,len(par_idx))]

	par_pos = np.subtract(data[par_idx],pos)/constraint
	par_aux = {}
	for k, v in aux_data.items():
		par_aux[k] = v[par_idx]

	if len(par_pos) < cnt:
		par_pos = np.concatenate((par_pos,np.zeros((cnt-len(par_pos),par_pos.shape[-1]))))
		for k, v in par_aux.items():
			par_aux[k] = np.concatenate((v,np.zeros((cnt-len(v),v.shape[-1]))))
			
	return par_pos, par_aux

def extract_patch(data, pos, patch_size):
	patch_size = patch_size//2
	x0 = int(pos[0])-patch_size
	x1 = int(pos[0])+patch_size+1
	y0 = int(pos[1])-patch_size
	y1 = int(pos[1])+patch_size+1
	return data[0,y0:y1,x0:x1]

def get_patches(sdf_data, patch_size, dimX, dimY, bnd, stride, surface):
	pos = []
	patch_size = patch_size//2
	for y in range(patch_size+bnd,dimY-patch_size-bnd, stride):
		for x in range(patch_size+bnd,dimX-patch_size-bnd, stride):
			z = 0
			if(abs(sdf_data[z,y,x,0]) < surface):
				x0 = x-patch_size
				x1 = x+patch_size+1
				y0 = y-patch_size
				y1 = y+patch_size+1

				pos.append([x,y,z])
	return np.array(pos)+0.5

def plot_particles(data, xlim, ylim, s, path=None, ref=None, src=None):
	if not ref is None:
		plt.scatter(ref[:,0],ref[:,1],s=s,c='r')
	plt.scatter(data[:,0],data[:,1],s=s,c='b')
	if not src is None:
		plt.scatter(src[:,0],src[:,1],s=s,c='g')
	plt.xlim(xlim)
	plt.ylim(ylim)
	if path is None:
		plt.show()
	else:
		plt.savefig(path)
	plt.clf()

def plot_sdf(data, xlim, ylim, path=None, ref=None, src=None, s=1):
	if not ref is None:
		plt.contour(np.arange(xlim[0],xlim[1])+0.5, np.arange(ylim[0],ylim[1])+0.5, ref, np.arange(-1,1.1,0.2), cmap=plt.get_cmap('coolwarm'))
		plt.contour(np.arange(xlim[0],xlim[1])+0.5, np.arange(ylim[0],ylim[1])+0.5, ref, np.array([0]), linewidths=3, colors='r')
		'''for x in range(xlim[0],xlim[1],1):
			for y in range(ylim[0],ylim[1],1):
				if ref[y,x] <= 0:
					plt.plot(x+0.5,y+0.5, 'ro')'''
	plt.contour(np.arange(xlim[0],xlim[1])+0.5, np.arange(ylim[0],ylim[1])+0.5, data, np.arange(-1,1.1,0.2))
	plt.contour(np.arange(xlim[0],xlim[1])+0.5, np.arange(ylim[0],ylim[1])+0.5, data, np.array([0]), linewidths=3, colors='b')
	if not src is None:
		plt.scatter(src[:,0],src[:,1],s=s,c='g',zorder=10)
	
	if path is None:
		plt.show()
	else:
		plt.savefig(path)
	plt.clf()

def plot_vec(data, xlim, ylim, path=None, ref=None, src=None, s=1):
	for y in range(ylim[0],ylim[1],2):
		for x in range(xlim[0],xlim[1],2):
			if not ref is None:
				v = ref[y,x]
				plt.plot([x+0.5,x+0.5+v[0]],[y+0.5,y+0.5+v[1]], 'r-')
			v = data[y,x]
			plt.plot([x+0.5,x+0.5+v[0]],[y+0.5,y+0.5+v[1]], 'b-')
	
	if not src is None:
		plt.scatter(src[:,0],src[:,1],s=s,c='g')
	plt.xlim(xlim)
	plt.ylim(ylim)
	if path is None:
		plt.show()
	else:
		plt.savefig(path)
	plt.clf()