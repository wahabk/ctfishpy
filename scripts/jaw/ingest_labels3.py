import ctfishpy
import napari
import numpy as np
from scipy import ndimage


if __name__ == "__main__":
    # dataset_path = "/home/ak18001/Data/HDD/uCT"
    dataset_path = "/home/wahab/Data/HDD/uCT"

    ctreader = ctfishpy.CTreader(dataset_path)

    bone = "JAW"
    # name = "JAW_manual"
    # name  = "JAW_20221208"
    name  = "JAW_20230124"
    new_name  = "JAW_20230223"
    roiSize = (256, 256, 320)
    keys = ctreader.get_hdf5_keys(dataset_path+f"/LABELS/JAW/{name}.h5")
    print(len(keys), keys)
    
    sophie = []
    sophie_done = [364,274,50,96,183,337,71,72,182,301,164,116,340,241]
    sophie_missing = [230]
    damiano = [131,216,351,39,139,69,133,135,420,441,220,291,401,250,193]

    for index in keys:
        print(index)
        index = 230

        scan = ctreader.read(index)
        # label = ctreader.read_label(bone, index, is_tif=True)
        label = ctreader.read_label(bone, index, name = new_name)

        # if index == 230:
        # 	new_label = np.zeros_like(label)
        # 	new_label[label==2] = 1
        # 	new_label[label==1] = 2
        # 	new_label[label==4] = 3
        # 	new_label[label==3] = 4
        # 	label = new_label


        # center = ctreader.jaw_centers[index]
        # scan_roi = ctreader.crop3d(scan, roiSize, center)
        # label_roi = ctreader.crop3d(label, roiSize, center)

        # label_roi = ctreader.label_array(scan_roi, label_roi)

        # label = ctreader.uncrop3d(label, label_roi, center=center)
        ctreader.view(scan, label)
    
        # ctreader.write_label(bone, label, index, name=new_name)
        exit()
        # yn = input("happy?(y/n)")
        # if yn == "y":	ctreader.write_label(bone, label, index, name=new_name)
