import ctfishpy
import numpy as np
import json


if __name__ == "__main__":
    dataset_path = "/home/ak18001/Data/HDD/uCT"
    ctreader = ctfishpy.CTreader(data_path=dataset_path)
    master = ctreader.mastersheet()
    print(master)
    # print(master["old_n"].to_list())

    hdf5_path = "/home/ak18001/Data/HDD/uCT/MISC/Organs/Otoliths_unet2d/Otoliths_unet2d.h5"
    data_path = "output/results/2d_unet_results.json"
    keys = ctreader.get_h5_keys(hdf5_path)
    # print(keys)
    nclasses = 3

    all_data = {}

    for old_n in keys:
        label = ctreader.read_hdf5(hdf5_path, int(old_n))

        if int(old_n) not in list(master['old_n']):
            print(f"FISH: {old_n} skipping -------------------------------")
            continue

        new_n = int(master.loc[master['old_n'] == int(old_n)].index[0])

        scan = ctreader.read(new_n)
        metadata = ctreader.read_metadata(new_n)

        dens = ctreader.getDens(scan, label, nclasses)
        vols = ctreader.getVol(label, metadata, nclasses)

        all_data[new_n] = {}
        all_data[new_n]['dens'] = list(dens)
        all_data[new_n]['vols'] = list(vols)

        print(f"FISH: {old_n}/{new_n}. Dens: {dens}. Vol: {vols}")

    print(all_data)

    with open(data_path, 'w') as f:
        json.dump(all_data, f, indent=4)


