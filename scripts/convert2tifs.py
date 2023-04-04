import ctfishpy
from tifffile import imsave

if __name__ == '__main__':
    dataset_path = "/home/ak18001/Data/HDD/uCT"
    ctreader = ctfishpy.CTreader(dataset_path)

    for fish in ctreader.fish_nums:
        scan = ctreader.read(fish)

        # ctreader.view(scan)
 
        imsave(f"/home/ak18001/Data/HDD/uCT/TIFS/ak_{fish}.tif", scan)