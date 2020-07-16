import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

def show_img(index, path="celeba/small_img_align_celeba.h5py", img_object="img_align_celeba"):
    with h5py.File(path, 'r') as file_object:
        dataset = file_object[img_object]
        image = np.array(dataset[f"{index}.jpg"])
        plt.imshow(image, interpolation='none')
        plt.show()
try:
    show_img(sys.argv[1])
except:
    raise IndexError()