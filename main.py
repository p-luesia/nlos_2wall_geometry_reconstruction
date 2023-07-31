import subsphere_ico
import sphere_initialisation as si

import matplotlib.pyplot as plt
import numpy as np
import optimisation as opt
import data_loader
from scipy.ndimage import gaussian_filter

import torch

import time

import argparse

#import pywavefront
#from pywavefront import visualization

import tal

def log(message, file):
	if f is None:
		print(message)
	else:
		file.write(f"{message}\n")
		file.flush()

def plotModel(ps, n, room_length, title='Reconstructed Model', showID=False, colors=['b']):
	fig = plt.figure(title)
	ax = fig.add_subplot(projection='3d')
	for p,c in zip(ps, colors):
		ax.scatter(p[0, :], p[1, :], p[2, :], color=c)
	ax.scatter(
		[room_length, room_length, room_length, room_length, -room_length, -room_length, -room_length, -room_length], 
		[room_length, room_length, -room_length, -room_length, room_length, room_length, -room_length, -room_length], 
		[room_length, -room_length, room_length, -room_length, room_length, -room_length, room_length, -room_length])
	if showID:
		for i in range(np.size(p, 1)):
			ax.text(p[0, i], p[1, i], p[2, i], f"{i + 1}")
	plt.show()

def blurVolume(V_LOD0, weight_lods):
	V_LOD1 = gaussian_filter(V_LOD0, 4, mode='nearest')
	V_LOD2 = gaussian_filter(V_LOD0, 32, mode='nearest')
	V_LOD3 = gaussian_filter(V_LOD0, 64, mode='nearest')
	
	V_LOD0 /= np.max(V_LOD0)
	V_LOD1 /= np.max(V_LOD1)
	V_LOD2 /= np.max(V_LOD2)
	V_LOD3 /= np.max(V_LOD3)

	V = weight_lods[0] * V_LOD0 + weight_lods[1] * V_LOD1 + weight_lods[2] * V_LOD2 + weight_lods[3] * V_LOD3
	V /= weight_lods[0] + weight_lods[1] + weight_lods[2] + weight_lods[3]

	return V

def program(args, f=None):
	# Obtain Base Sphere
	p = subsphere_ico.generate_points(args.subdivisions)
	n = p.copy()

	# Load 2-view data
	t = time.time()
	V0 = data_loader.load(args.volume1)
	V1 = data_loader.load(args.volume2)
	t = time.time() - t
	log(f"Time to load volumes: {t}s", f)

	# Read normals
	Vn0 = np.fromstring(args.normal1, dtype='float64', sep=' ')
	Vn1 = np.fromstring(args.normal2, dtype='float64', sep=' ')

	# Initialise sphere position and size
	if args.raymarching:
		p = si.initialise_raymarching(p, V0, V1)
	elif args.filltobounds:
		p = si.initialise_stdev(p, V0, V1)
	else:
		p = si.initialise_standard(p, V0, V1)

	#tal.plot.volume(V0 + V1)
	#subsphere_ico.save_as_obj(p, args.subdivisions, 'output/TEMP2.obj')
	#plotModel([p], n, args.roomsize, colors=['b'])

	# Obtain neighbours
	neighbours = subsphere_ico.neighbours_matrix(p, args.subdivisions)

	# Log GPU usage
	if torch.cuda.is_available() and not args.cpu:
		log(f"Using GPU", f)
	else:
		log(f"Using CPU", f)

	print(f"Applying blur to volumes...")
	t = time.time()
	weight_lods=[args.weightlod0, args.weightlod1, args.weightlod2, args.weightlod3]
	V0 = blurVolume(V0, weight_lods)
	V1 = blurVolume(V1, weight_lods)
	t = time.time() - t
	log(f"Time for blurring: {t}s", f)

	# Optimise
	#po, no = opt.optimizeTest(p, n)
	t = time.time()
	po, no = opt.optimizeSphere(p, n, neighbours, V0, V1, Vn0, Vn1, args)
	t = time.time() - t
	log(f"Time for optimising: {t}s", f)
	log(f"Time for optimising (Average per iteration): {t/args.iterations}s", f)

	# Save Result
	subsphere_ico.save_as_obj(po, args.subdivisions, args.output)

	if f is not None:
		f.close()

	# Plot
	if f is None:
		plotModel([po], no, args.roomsize)


if __name__ == "__main__":
	argp = argparse.ArgumentParser(
		prog='2-wall NLOS geometry reconstructor',
		description="Reconstructs geometry from data obtained from 2-wall NLOS Imaging"
	)

	argp.add_argument('volume1', type=str)
	argp.add_argument('volume2', type=str)
	argp.add_argument('normal1', type=str)
	argp.add_argument('normal2', type=str)
	argp.add_argument('output', type=str)
	argp.add_argument('-i', '--iterations', default=10, type=int)
	argp.add_argument('-s', '--subdivisions', default=1, type=int)
	argp.add_argument('-r', '--roomsize', default=5, type=float)

	argp.add_argument('-log', '--log', default=None, type=str)

	argp.add_argument('-wv0', '--weightvolume0', default=1, type=float)
	argp.add_argument('-wv1', '--weightvolume1', default=1, type=float)
	argp.add_argument('-wlod0', '--weightlod0', default=1, type=float)
	argp.add_argument('-wlod1', '--weightlod1', default=0.5, type=float)
	argp.add_argument('-wlod2', '--weightlod2', default=0.25, type=float)
	argp.add_argument('-wlod3', '--weightlod3', default=0.1, type=float)
	argp.add_argument('-wv', '--weightvalues', default=1, type=float)
	argp.add_argument('-wn', '--weightnormals', default=1, type=float)
	argp.add_argument('-wp', '--weightproximity', default=0, type=float)
	argp.add_argument('-wneigh', '--weightneighbours', default=0, type=float)
	argp.add_argument('-we', '--weightedges', default=0, type=float)
	argp.add_argument('-sq', '--sequence', default=None, type=str)

	argp.add_argument('-cpu', '--cpu', action='store_true')
	argp.add_argument('-fb', '--filltobounds', action='store_true')
	argp.add_argument('-rm', '--raymarching', action='store_true')

	args = argp.parse_args()

	f = None
	if args.log is not None:
		f = open(args.log, 'w')

	log(f"Volume 1: '{args.volume1}', Volume 2: '{args.volume2}'", f)
	log(f"Normal 1: '{args.normal1}', Normal 2: '{args.normal2}'", f)
	log(f"Output: '{args.output}', Force CPU?: {args.cpu}", f)
	log(f"Iterations: {args.iterations}, Subdivisions: {args.subdivisions}", f)
	log(f"Room radius: {args.roomsize}, Weight Normals: {args.weightnormals}, Weight Values: {args.weightvalues}", f)
	log(f"Weight Proximity: {args.weightproximity}, Weight Neighbours: {args.weightneighbours}", f)
	log(f"Weight Edges Similarity: {args.weightedges}", f)
	log(f"Weight LODs: ({args.weightlod0}, {args.weightlod1}, {args.weightlod2}, {args.weightlod3})", f)

	if args.filltobounds:
		log(f"Initialisation: Center of Mass + Standard Deviation", f)
	elif args.raymarching:
		log(f"Initialisation: Raymarching", f)
	else:
		log(f"Initialisation: Center of Mass", f)

	if args.sequence is not None:
		log(f"Saving sequence on directory '{args.sequence}'", f)
	log("---------------", f)

	program(args, f)