import os
import sys
import h5py
import zipfile
import imageio
from tqdm import tqdm

def convert_to_hdf5_file(index):
  
  hdf5_file = f"celeba/{index}_img_align_celeba.h5py"

  # how many of the 202,599 images to extract and package into HDF5
  total_images = int(index)

  with h5py.File(hdf5_file, 'w') as hf:
      with zipfile.ZipFile("celeba/img_align_celeba.zip", "r") as zf:
        for count, i in tqdm(enumerate(zf.namelist())):
          if (i[-4:] == ".jpg"):

            # extract image
            ofile = zf.extract(i)
            img = imageio.imread(ofile)
            os.remove(ofile)

            # add image data to HDF5 file with new name
            hf.create_dataset("img_align_celeba/"+str(count-1)+".jpg", data=img, compression="gzip", compression_opts=9)
              
            # stop when total_images reached
            if (count == total_images):
              break

try:
  convert_to_hdf5_file(sys.argv[1])
except:
  raise IndexError()