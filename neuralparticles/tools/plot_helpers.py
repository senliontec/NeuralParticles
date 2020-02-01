#
# tool runs helpers
#
import numpy as np
import csv

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

def write_dict_csv(path, data):
	with open(path, "w") as csvfile:
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(data.keys())
		csvwriter.writerows(zip(*data.values()))

def write_csv(path, data):
	with open(path, 'w') as csvfile:
		csvwriter = csv.writer(csvfile)
		for d in data:
			csvwriter.writerow(d)

def read_csv(path):
	data = None
	with open(path, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			if data is None:
				data = np.array([row]).astype(float)
			else:
				data = np.concatenate((data, np.array([row]).astype(float)))
	return data
	
def extract_stride_idx(data, z, offset, pos=None):
	if pos is None:
		pos = data
	return np.where(np.all([pos[:,2] < (z + offset), pos[:,2] > (z - offset)], axis=0))[0]

def extract_stride(data, z, offset, pos=None):
	idx = extract_stride_idx(data, z, offset, pos)
	return data[idx,0], data[idx,1]

def plot_particles(data, xlim=None, ylim=None, s=1, path=None, ref=None, src=None, vel=None, z=None, offset=None, c='c'):
	if z is not None and offset is None:
		if xlim is None:
			offset = (np.max(data) - np.min(data)) * 0.1
		else:
			offset = (xlim[1] - xlim[0]) * 0.1
	if z is None:
		dx,dy = (data[:,0], data[:,1])
	else:
		idx = extract_stride_idx(data,z,offset)
		dx,dy = (data[idx,0], data[idx,1])
		if type(c) is np.ndarray or type(c) is list:
			c = c[idx]

	if not ref is None:
		rx, ry = (ref[:,0], ref[:, 1]) if z is None else extract_stride(ref, z,offset)
		plt.scatter(rx,ry,s=s,c='#ff6c00')
	plt.scatter(dx,dy,s=s,c=c, cmap=plt.get_cmap("nipy_spectral"))
	if not src is None:
		sx,sy = (src[:,0], src[:,1]) if z is None else extract_stride(src, z,offset)
		plt.scatter(sx,sy,s=s,c='#0a3213')
		if not vel is None:
			vx, vy =  (vel[:,0],vel[:,1]) if z is None else extract_stride(vel, z, src)
			#TODO: make more efficient:
			for i in range(len(src)):
				plt.plot([sx[i],sx[i]+vx[i]],[sy[i],sy[i]+vy[i]], 'g-')
	if not xlim is None:
		plt.xlim(xlim)
	if not ylim is None:
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

def plot_vec(data, xlim=None, ylim=None, path=None, ref=None, src=None, s=1):
	for y in range(ylim[0],ylim[1],2):
		for x in range(xlim[0],xlim[1],2):
			if not ref is None:
				v = ref[y,x]
				plt.plot([x+0.5,x+0.5+v[0]],[y+0.5,y+0.5+v[1]], 'r-')
			v = data[y,x]
			plt.plot([x+0.5,x+0.5+v[0]],[y+0.5,y+0.5+v[1]], 'b-')
	
	if not src is None:
		plt.scatter(src[:,0],src[:,1],s=s,c='g')
	if not xlim is None:
		plt.xlim(xlim)
	if not ylim is None:
		plt.ylim(ylim)
	if path is None:
		plt.show()
	else:
		plt.savefig(path)
	plt.clf()
