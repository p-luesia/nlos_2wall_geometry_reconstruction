*Written for the state of the program in **21/6/2023***

# How to run
Run the script ``main.py``. It has the following required arguments:

|Name|Position|Description|
|-|-|-|
|Volume 0|0|Path pointing to the first volume used|
|Volume 1|1|Path pointing to the second volume, using the same coordinate system as Volume 0|
|Normal 0|2|Normals that are seen at volume 0. Inverse vector of the relay wall 0. Write as string with spaces separating coordinates|
|Normal 1|3|Normals that are seen at volume 1. Inverse vector of the relay wall 1. Write as string with spaces separating coordinates|
|Output|4|Path pointing to where the resulting obj should be saved

An example of launching the script would be:

```sh
python main.py \
	"data/cube/cube_0.0_rec.npy" \ 
	"data/cube/cube_90.0_to_0.0_rec.npy" \
	"-1 0 0" "0 0 1" "output.obj"
```

There are also optional arguments:

|Short form|Long form|Description|Default Value|
|-|-|-|-|
|``-i``|``--iterations``|Integer, Number of iterations of optimisation|``10``|
|``-s``|``--subdivisions``|Integer, number of subdivisions done to the sphere, starting at 1|``1``|
|``-r``|``--roomsize``|Size of the volumes where a unit-radius sphere will be placed for optimisation. This size is half of the length of the room|``5``|
|``-log``|``-log``|File path for a log of the program's execution|*None*|
|``-wlod0``|``--weightlod0``|Weight given to the no blurry version of the volumes|``1``|
|``-wlod1``|``--weightlod1``|Weight given to the least blurry version of the volumes|``0.5``|
|``-wlod2``|``--weightlod2``|Weight given to the medium blurry version of the volumes|``0.25``|
|``-wlod3``|``--weightlod3``|Weight given to the high blurry version of the volumes|``0.1``|
|``-wv``|``-weightvalues``|Weight given to the value factor on the energy function|``1``|
|``-wn``|``--weightnormals``|Weight given to the normals factor on the energy function|``1``|
|``-wp``|``--weightproximity``|Weight given to the proximity factor on the energy function|``0``|
|``-we``|``--weightedges``|Weight given to the minimization of the standard deviation of the length of edges in the energy function|``0``|
|``-wneigh``|``--weightneighbours``|Weight given to the neighbours factor on the energy function|``0``|
|``-sq``|``--sequence``|Path where a sequence of ``obj`` files are saved for every iteration|*None*|

...and flags:

|Short form|Long form|Description|
|-|-|-|
|``-cpu``|``--cpu``|Forces script to use the CPU instead of GPU|
|``-fb``|``--filltobounds``|Reshapes sphere before optimisation to size of standard deviation|
|``-rm``|``--raymarching``|Uses raymarching to initialise the starting sphere before any optimisation occurs|

# Visualization

You can run the Python script ```visualize.py``` to obtain a 3D visualization of the optimization!!
Example of use:

```sh
python visualize.py volume_0_reconstruction.npy volume_1_reconstruction.npy ... volume_n_reconstruction.npy -p " -1,2,3" "0,0,0" -m mesh/dir/
```
This will plot all the reconstructed volume values, with two points placed at (-1, 2, 3) and (0,0,0), with a mesh from the mesh/dir/ directory selected iteratively with a slider.

Library requirements:
* pyvista>=0.41.1
* numpy>=1.23.3