# I have become what I feared the most
# British

import numpy as np
import math
import scipy.optimize as opt
import torch

import matplotlib.pyplot as plt

from scipy.interpolate import interpn
import scipy.interpolate as sc_intp

import subsphere_ico

from tqdm import tqdm

import tal

samplingPosition = (np.linspace(-1, 1, 256), np.linspace(-1, 1, 256), np.linspace(-1, 1, 256))

#
# Try cube
# Multiplication instead of sum (Already attempted)
# Weights
# Fair size between vertex
#

def plotSamplingFunction(V, room_length, k=127):
	plt.figure()
	im = np.zeros((256, 256))
	for i in range(256):
		for j in range(256):
			p = np.array([[i], [j], [k]])
			p = (2*(p/256)-1) * room_length
			im[i, j] = sampleVolume(V, p, room_length)
	plt.imshow(im, cmap='hot', interpolation='nearest')
	plt.show()

def optimizeSphere(input_x, input_n, input_neighbours, V0, V1, input_Vn0, input_Vn1, args):
	global samplingPosition

	res = V0.shape[0]
	x = torch.from_numpy(np.copy(input_x))
	# samplingPosition = (np.linspace(-1, 1, res), np.linspace(-1, 1, res), np.linspace(-1, 1, res))

	print(f"Checking for device...", end='', flush=True)
	dev = torch.device('cpu' if (not torch.cuda.is_available() or args.cpu) else 'cuda')
	print(f" Found: '{dev}'!")

	n = torch.from_numpy(input_n.astype('float64')).to(dev)
	Vn0 = torch.from_numpy(input_Vn0.astype('float64')).to(dev)
	Vn1 = torch.from_numpy(input_Vn1.astype('float64')).to(dev)
	neighbours = torch.argwhere(input_neighbours).to(dev)

	V0_interp = sc_intp.RegularGridInterpolator(samplingPosition,
					     						V0,
												bounds_error=False,
												fill_value = 0)
	V1_interp = sc_intp.RegularGridInterpolator(samplingPosition,
					     						V1,
												bounds_error=False,
												fill_value = 0)

	print(f"Optimising sphere to voxel data for {args.iterations} iterations:")
	# This is not working at all
	# for i in tqdm(range(args.iterations)):
	# 	res = opt.minimize(powerSphere, x.reshape(-1), args=(dev, n, neighbours, V0_interp, V1_interp, Vn0, Vn1, args), method='CG',
	# 		options={'disp': False, "maxiter": 1}, jac='3-point') #jac=jac_function
	# 	x = unpack(res.x)
	# 	if args.sequence is not None:
	# 		subsphere_ico.save_as_obj(x, args.subdivisions, f"{args.sequence}/{i}.obj")
	res = opt.minimize(powerSphere, x.reshape(-1), args=(dev, n, neighbours, V0_interp, V1_interp, Vn0, Vn1, args), method='CG',
			options={'disp': False, "maxiter": args.iterations, "return_all": True}, jac='3-point')
	print(f"Stopped after {len(res.allvecs) - 1} iterations")
	if args.sequence is not None:
		for i, it_x in enumerate(res.allvecs):
			subsphere_ico.save_as_obj(unpack(it_x), args.subdivisions, f"{args.sequence}/{i}.obj")

	return unpack(res.x), n

def powerSphere(data, device, n, neighbours, V0_interp, V1_interp, Vn0, Vn1, args):
	x = unpack(data)

	# Energy depending on the reconstruction value
	v0 = V0_interp(x.T)
	v1 = V1_interp(x.T)

	eValue = np.sum(args.weightvolume0 * v0 + args.weightvolume1 * v1)

	# Regularization based on the std of the distance between connected vertex
	edges = x[:, neighbours]
	edges_v = np.diff(edges, axis = 2)[:,:,0 ]
	# Square distance of all edges
	sq_dists = np.sum(edges_v**2, axis = 0)
	eEdges = np.std(sq_dists)

	# Estimate the normals for each vertex
	edges_v_grp = np.split(edges_v, 
							np.unique(neighbours[:,0], 
							return_index=True)[1][1:], axis = 1)
	# Cross product
	cross_edges= list(map(lambda e_v: np.cross(e_v, np.roll(e_v, 1, axis=1),
					     axisa=0, 
						 axisb=0), edges_v_grp))
	# Obtain the normals
	normals_v = np.array(list(map(lambda e_c_v: \
				  np.sum(e_c_v/np.linalg.norm(e_c_v, axis = 1)[:, np.newaxis], axis = 0),
					cross_edges)))
	normals_v /= np.linalg.norm(normals_v, axis = 0)

	# Cosine term respect the relay walls
	cos_term_0 = (np.abs(Vn0 @ n))
	cos_term_1 = (np.abs(Vn1 @ n))
	# High values with the normal pointing outside the relay wall are penalized
	cos_term_0 -= 0.5; cos_term_0 *= 2
	cos_term_1 -= 0.5; cos_term_1 *= 2
	# Reconstruction value depending on the normal
	eNormal = np.sum(args.weightvolume0 * v0*cos_term_0.numpy() \
		  				+ args.weightvolume1 * v1*cos_term_1.numpy())

	# Combined energy
	return -(args.weightvalues * eValue - args.weightedges * eEdges + args.weightnormals*eNormal)

	cos0 = (Vn0 @ n - 1) / 2
	cos1 = (Vn1 @ n - 1) / 2

	eNormal = torch.sum(args.weightvolume0 * torch.multiply(v0, cos0) + args.weightvolume1 * torch.multiply(v1, cos1))
	eNormal = eNormal.item()
	#eNormal = np.sum(np.multiply(v0.cpu().numpy(), cos0) + np.multiply(v1.cpu().numpy(), cos1))

	# Get distance² of every point to every other point
	dists = torch.cdist(x_torch, x_torch, p=1.0)
	dists_sorted = torch.sort(dists, dim=0)[0]

	# Energy based on proximity to closest point
	dists_closest = dists_sorted[1,:]
	eProximity = -1 * torch.sum(1/(100 * dists_closest))
	eProximity = eProximity.item()

	# Energy based on std dev of neighbours' distances
	n_neighbours = torch.sum(neighbours, dim=0)
	neighbour_dist_mean = torch.sum(dists * neighbours, dim=0) / n_neighbours
	neighbour_dist_stdev = torch.sum(neighbours * (dists - neighbour_dist_mean)**2, dim=0) / n_neighbours #Sigma squared
	eNeighbours = torch.sum(neighbour_dist_stdev)
	eNeighbours = eNeighbours.item()

	# Energy based on std dev of all the edges of the reconstruction
	n_edges = torch.sum(n_neighbours)
	edges_mean = torch.sum(dists * neighbours) / n_edges
	edges_stdev = torch.sum(neighbours * (dists - edges_mean)**2) / n_edges
	eEdges = edges_stdev.item()

	return -(args.weightvalues * eValue + args.weightnormals * eNormal + args.weightproximity * eProximity - args.weightneighbours * eNeighbours - args.weightedges * eEdges)

def sampleVolume(V, x):
	global samplingPosition
	return interpn(samplingPosition, V, x.T, bounds_error=False, fill_value=0)

def unpack(data):
	return data.reshape(3, -1)

def minDistanceBetweenPoints(x):
	dists = torch.cdist(x, x, p=1.0)
	dists = torch.sort(dists, dim=0)
	return dists