import ctfishpy
import numpy as np
import json
import pandas as pd


if __name__ == "__main__":
    dataset_path = "/home/ak18001/Data/HDD/uCT"
    ctreader = ctfishpy.CTreader(data_path=dataset_path)
    master = ctreader.mastersheet()
    print(master)
    # print(master["old_n"].to_list())

    hdf5_path = "/home/ak18001/Data/HDD/uCT/MISC/Organs/Otoliths_unet2d/Otoliths_unet2d.h5"
    data_path = "output/results/2d_unet_results.csv"
    keys = ctreader.get_h5_keys(hdf5_path)
    # print(keys)
    nclasses = 3

    all_data = pd.DataFrame(columns=['Density', 'Density2', 'Density3', 'Volume1', 'Volume2', 'Volume3'])

    for old_n in keys:
        label = ctreader.read_hdf5(hdf5_path, int(old_n))

        if int(old_n) not in list(master['old_n']):
            print(f"FISH: {old_n} skipping -------------------------------")
            continue

        new_n = int(master.loc[master['old_n'] == int(old_n)].index[0])
        print(f'FISH: {old_n}/{new_n}')

        scan = ctreader.read(new_n)
        metadata = ctreader.read_metadata(new_n)

        dens = ctreader.getDens(scan, label, nclasses)
        vols = ctreader.getVol(label, metadata, nclasses)

        all_data.loc[new_n] = list(dens) + list(vols)

        print(f"Dens: {dens}. Vol: {vols}")

        all_data.to_csv(data_path)

    print(all_data)




