
import ctfishpy
from tifffile import imsave
from pathlib2 import Path

input_dir = 'D:\Data\ak_1.dcm'
result_dir = "D:\Data"
bone_to_segment = "JAW" 



def predict_bone(bone_to_segment, input_dir):
  if bone_to_segment == None:
    raise Exception("Please select a bone to segment from the dropdown above")
  elif bone_to_segment == "JAW":
    bone_obj = ctfishpy.bones.Jaw()
  elif bone_to_segment == "OTOLITHS":
    bone_obj = ctfishpy.bones.Otoliths()


  ctreader = ctfishpy.CTreader()
  array = ctreader.read_tif(input_dir)
  label = bone_obj.predict(array)

  # Testing
  # dataset_path = "/home/ak18001/Data/HDD/uCT"
  # ctreader = ctfishpy.CTreader(dataset_path)
  # n = 200
  # center = ctreader.jaw_centers[n]
  # array = ctreader.read_roi(n, (100,100,100), center)
  # label = bone_obj.predict(array)
  
  return array, label



if __name__ == "__main__": 
  # Testing
  # bone_to_segment = "JAW"
  # input_dir = None

  input_array, label = predict_bone(bone_to_segment, input_dir)
  print(label.shape, label.max(), label.min())

  input_dir = Path(input_dir)
  output_dir = input_dir.parent / input_dir.stem / "_label.tiff"
  imsave(output_dir, label)
  
  # TODO show output projections
  # ctreader = ctfishpy.CTreader()
  # ctreader.make_max_projections(input_array)
  # ctreader.label_projections(projections, mask_projections)
