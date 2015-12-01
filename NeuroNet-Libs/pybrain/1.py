import numpy as np
from skimage.io import imread
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure import SigmoidLayer

def compute_rows(image_array):
    pixels_per_row = image_array.sum(axis=1) 
    max_pixels_per_row = pixels_per_row.max()
    min_pixels_per_row = pixels_per_row[pixels_per_row!=0].min()
    nbr_max_rows = len(pixels_per_row[pixels_per_row==max_pixels_per_row])    
    return min_pixels_per_row, max_pixels_per_row, nbr_max_rows
    
    
def compute_angles(image_array):
    nrows = image_array.shape[0]-1
    ncols = image_array.shape[1]-1
    nangles = 0    
    for i in range(nrows):
        for j in range(ncols):
            if (image_array[i:i+2, j:j+2].sum()%2)!=0:
                nangles+=1                
    return nangles

    
def image_to_inputs(image):
    image = np.where(image==0, 1, 0) 
    inputs = []
    rows_info = compute_rows(image)    
    for value in rows_info:
        inputs.append(value)        
    angles_nbr = compute_angles(image)
    inputs.append(angles_nbr)    
    return inputs


def import_dataset(path, shapes, used_for, samples_nbr):
    ds = ClassificationDataSet(4, nb_classes=3)    
    for shape in sorted(shapes):
        for i in range(samples_nbr):
            image = imread(path+used_for+'/'+shape+str(i+1)+'.png', as_grey=True, plugin=None, flatten=None)
            image_inputs = image_to_inputs(image)
            ds.appendLinked(image_inputs, shapes[shape])            
    return ds

