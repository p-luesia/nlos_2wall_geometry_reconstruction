import numpy as np
import torch
import optimisation

import scipy.ndimage as ndi

#
# Initialise sphere
#

def initialise_standard(p, V0, V1):
	barycenter = centerOfMass(V0, V1)
	print(f"Initialising sphere with center of mass {barycenter}")
	
	return p + barycenter[:, np.newaxis]

def initialise_stdev(p, V0, V1, room_length):
	mu, sigma = centerOfMassAndDeviation(V0, V1, room_length)
	print(f"Initialising sphere with center of mass {mu} and std dev {sigma}")

	#p += mu.reshape(-1,1)
	#p *= sigma.reshape(-1,1)
	return (p + mu.reshape(-1, 1)) * sigma.reshape(-1, 1)

def initialise_raymarching(p, V0, V1, march_divisions=32):
	mu = centerOfMass(V0, V1)

	dmax = 2 #Maximum possible distance between point and max volume value
	V = V0 + V1
	norm_p = p/np.linalg.norm(p,axis = 0)

	delta = np.tile(dmax * np.linspace(0, 1, march_divisions)**2, (p.shape[1],1))
	raymarched_points = delta[np.newaxis,:,:]*norm_p[:,:,np.newaxis]
	raymarched_points += mu[:, np.newaxis, np.newaxis]

	# Values for every point in the raymarching. Rows indicate point, column raymarched interation
	values = optimisation.sampleVolume(V, raymarched_points.reshape(3, -1))
	values = values.reshape(-1, march_divisions)
	values *= (1-delta/dmax)**2  # Give less importance to values far from the center

	best_delta = np.argmax(values, axis=1)
	best_positions = raymarched_points[:, np.arange(p.shape[1]), best_delta]
	return best_positions

#
# Calculate center of mass
#

def centerOfMass(V0, V1):
	c = np.asarray(ndi.center_of_mass(V0 + V1))
	return (2 * (c / 256) - 1)

def centerOfMassAndDeviation(V0, V1):
	res = V0.shape[0]
	c = np.mgrid[0:res,0:res,0:res].reshape((3, res**3))
	w = (V0 + V1).reshape((1, res**3)).squeeze()

	mu = np.average(c, weights=w, axis=1)
	sigma2 = np.average((c - mu.reshape(-1,1))**2, weights=w, axis=1)
	sigma = np.sqrt(sigma2)

	mu = (2 * (mu / res) - 1)
	sigma = (sigma / res) * 2

	return mu, sigma