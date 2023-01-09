import ctfishpy

dataset_path = "/home/ak18001/Data/HDD/uCT"
ctreader = ctfishpy.CTreader(dataset_path)

scan = ctreader.read(1)

ctreader.view(scan)

