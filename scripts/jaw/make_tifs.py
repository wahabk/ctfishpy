import ctfishpy
import numpy as np
from tifffile import imsave


if __name__ == "__main__":
	dataset_path = "/home/ak18001/Data/HDD/uCT"
	# dataset_path = "/home/wahab/Data/HDD/uCT"
	ctreader = ctfishpy.CTreader(dataset_path)
	master = ctreader.master

	curated = [257,351,241,164,50,39,116,441,291,193,420,274,364,401,72,71,69,250,182,183,301,108,216,340,139,337,220,1,154,230,131,133,135,96,98,]
	damiano = [131,216,351,39,139,69,133,135,420,441,220,291,401,250,193]
	sophie = [96,183,337,71,72,182,274,364,301,164,116,230,50,241,340]
	me = [1,257,98,108,154]

	for n  in  damiano:
		print(f"Reading {n} for d")
		scan  = ctreader.read(n)
		out_path = f"/home/ak18001/Data/HDD/uCT/MISC/DS_SEGS/DAMIANO/{n}.tif"
		print(f"Saving {n} for d")
		imsave(out_path, scan)

	for n  in  sophie:
		print(f"Reading {n} for s")
		scan  = ctreader.read(n)
		out_path = f"/home/ak18001/Data/HDD/uCT/MISC/DS_SEGS/SOPHIE/{n}.tif"
		print(f"Saving {n} for for s")
		imsave(out_path, scan)