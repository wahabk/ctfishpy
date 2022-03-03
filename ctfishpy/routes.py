import controller
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--view", required=True,
	help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
	help="path to *specific* model checkpoint to load")
args = vars(ap.parse_args())



