from pathlib2 import Path
import h5py


def write_hdf5(dataset, canvas, positions, metadata):
	path = Path(f'data/{dataset}.hdf5')
	if not path.exists: 
		f = h5py.File(path, "w")
		f.create_dataset()
		f.close()
	
	with h5py.File(path, "r") as f:
		sofar = [int(k) for k in f.keys()]
		f.create_dataset()

		f.close()
	pass

def read_hdf5():
	pass