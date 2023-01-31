import ctfishpy

if __name__ == '__main__':
    dataset_path = "/home/ak18001/Data/HDD/uCT"
    ctreader = ctfishpy.CTreader(dataset_path)

    index = 1
    scan = ctreader.read(index)

    bone = ctreader.JAW
    name = 'jaw_unet_230124'
    jaw_label = ctreader.read_label(bone, index, name=name,)
    # jaw_label = jaw_label[1000:]
    print(f"jaw {jaw_label.shape}")

    bone = ctreader.OTOLITHS
    name = "OTOLITHS_FINAL"
    oto_label = ctreader.read_label(bone, index, name=name,)
    oto_label = oto_label
    oto_center = ctreader.otolith_centers[index]
    oto_label = ctreader.uncrop3d(scan, oto_label, oto_center)
    print(f"oto {oto_label.shape}")

    label = jaw_label + oto_label
 
    jaw_center = ctreader.otolith_centers[index]
    scan = ctreader.crop3d(scan, (650,400,400), center=jaw_center)
    label = ctreader.crop3d(label, (650,400,400), center=jaw_center)

    ctreader.view(scan, label=label)

