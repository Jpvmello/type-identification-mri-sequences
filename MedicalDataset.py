import os
import pandas
import torch
import nibabel
import numpy
import random
import cv2
import SimpleITK as sitk
import torchvision.transforms.functional as TF
import glob
import operator
import time

from math import floor, ceil
from torch.utils.data import Dataset
from imgaug import augmenters as aug
from dicom2nifti import dicom_series_to_nifti
from dicom2nifti.image_reorientation import reorient_image, _reorient_3d
from dicom2nifti.image_volume import load, ImageVolume
from SimpleITK import ImageFileReader, ImageSeriesReader
from time_util import time_format

class MedicalDataset(Dataset):
    def __init__(self, data_csv, min_slices = 10, consider_other_class = True, test = False):
        self.images = pandas.read_csv(data_csv)
        self.min_slices = min_slices
        self.consider_other_class = consider_other_class
        self.is_train = not test
        self.loaded_data = self.load_data()
        if not self.consider_other_class:
            print('Actually loaded:', self.__len__(), '("Other" class discarded)')
    
    def load_data(self):
        data = []
        start = time.time()
        print('Starting loading data...')
        #print(self.images), exit()
        for i, (image_path, label) in self.images.iterrows():
            image_path = os.path.join(os.getcwd(), image_path)
            if not self.consider_other_class and label == 4:
                print('DISCARDED: "Other" class -', image_path)
                continue
            print('Loading', image_path)
            if os.path.isdir(image_path):
                reader = ImageSeriesReader()
                sorted_file_names = reader.GetGDCMSeriesFileNames(image_path)
                reader.SetFileNames(sorted_file_names)
                image = reader.Execute()
            else:
                image = sitk.ReadImage(image_path)

            '''if self.is_train:
                image = self.rotate(image)'''
            pixel_data, color_channels, direction = self.normalize(sitk.GetArrayFromImage(image)), image.GetNumberOfComponentsPerPixel(), image.GetDirection()

            if (color_channels > 1):
                pixel_data = numpy.array([cv2.cvtColor(slice, cv2.COLOR_RGB2GRAY) for slice in pixel_data])

            n_slices = pixel_data.shape[0]
            
            min_slices = 16 #self.min_slices
            if n_slices < min_slices:
                extended = numpy.zeros((min_slices, pixel_data.shape[1], pixel_data.shape[2])).astype(numpy.uint8)
                extended[min_slices//2 - n_slices//2 : min_slices//2 + n_slices//2 + n_slices%2, :, :] = pixel_data.copy()
                for i in range(min_slices//2 - n_slices//2):
                    extended[i] = pixel_data[0]
                for i in range(min_slices//2 + n_slices//2 + 1, min_slices):
                    extended[i] = pixel_data[-1]
                pixel_data = extended
            else:
                pixel_data = pixel_data[n_slices//2 - min_slices//2 : n_slices//2 + min_slices//2 + min_slices%2]
                #pixel_data = numpy.array([pixel_data[int(i*n_slices/(min_slices+1))] for i in range(1, min_slices + 1)])
            
            height, width = pixel_data.shape[1], pixel_data.shape[2]
            if height != width:
                min_dim = min(height, width)
                max_dim = max(height, width)
                ratio = min_dim/max_dim
                zoom_out = numpy.zeros(pixel_data.shape).astype(numpy.uint8)
                for i, slice in enumerate(pixel_data):
                    slice = cv2.resize(slice, (0, 0), fx = ratio, fy = ratio)
                    zoom_out[i, :slice.shape[0], :slice.shape[1]] = slice
                    '''ymin = zoom_out.shape[1]//2 - floor(slice.shape[0]/2)
                    ymax = zoom_out.shape[1]//2 + ceil(slice.shape[0]/2)
                    xmin = zoom_out.shape[2]//2 - floor(slice.shape[1]/2)
                    xmax = zoom_out.shape[2]//2 + ceil(slice.shape[1]/2)
                    zoom_out[i, ymin:ymax, xmin:xmax] = slice'''
                pixel_data = zoom_out
                pixel_data = pixel_data[:, :min_dim, :min_dim]  
                assert(pixel_data.shape[1] == pixel_data.shape[2])
            pixel_data = numpy.array([cv2.resize(slice,(200, 200)) for slice in pixel_data])
            #new_pixel_data = []
            #for slice in pixel_data:
            #    slice = cv2.resize(slice, (200, 200))
            #    if (color_channels > 1):
            #        slice = cv2.cvtColor(slice, cv2.COLOR_RGB2GRAY)
            #    new_pixel_data.append(slice)
            #pixel_data = numpy.array(new_pixel_data)
            assert(pixel_data.shape == (min_slices, 200, 200))
            data.append((pixel_data, label))
                
            print('Loaded', i + 1, '/', len(self.images), '' if self.consider_other_class else '(counting discarded).')
        print('\nLoading time:', time_format(time.time() - start))
        return data
    
    def rotate(self, image):
        def cos(vector1, vector2):
            for axis in range(0, 3):
                axes_relation = vector1[axis] * vector2[axis]
                if abs(axes_relation) == 1:
                    return axes_relation
            return 0
        
        direction = image.GetDirection()
        sagittal = (round(direction[0]), round(direction[1]), round(direction[2])) #Width, 2 (Lateral)
        coronal = (round(direction[3]), round(direction[4]), round(direction[5])) #Height, 1 (Front-Back)
        axial = (round(direction[6]), round(direction[7]), round(direction[8])) #Layers, 0   (Axial)
        
        coords_are_left_hand = (axial == numpy.cross(sagittal, coronal)).all()
        
        vectors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        random.shuffle(vectors)
        x, y, z = vectors[0], vectors[1], vectors[2]
        new_coords_are_left_hand = not coords_are_left_hand
        while new_coords_are_left_hand != coords_are_left_hand:
            x = tuple( random.choice([-1, 1]) * numpy.array(x) )
            y = tuple( random.choice([-1, 1]) * numpy.array(y) )
            z = tuple( random.choice([-1, 1]) * numpy.array(z) )
            new_coords_are_left_hand = (z == numpy.cross(x, y)).all()
            
        rotation_matrix = numpy.array([
            [cos(sagittal, x), cos(coronal, x), cos(axial, x)],
            [cos(sagittal, y), cos(coronal, y), cos(axial, y)],
            [cos(sagittal, z), cos(coronal, z), cos(axial, z)]
        ]).astype(numpy.double)
        
        rotation_transform = sitk.AffineTransform(3)
        rotation_transform.SetMatrix(rotation_matrix.ravel())
        
        color_channels = image.GetNumberOfComponentsPerPixel()
        pixel_type = sitk.sitkFloat32 if color_channels == 1 else sitk.sitkVectorFloat32
        return sitk.Resample(image, rotation_transform, sitk.sitkLinear, 0, pixel_type)
    
    def normalize(self, pixel_data):
        pixel_data = pixel_data.astype(numpy.float32)
        min = numpy.min(pixel_data)
        max = numpy.max(pixel_data)

        pixel_data = numpy.uint8(255 * (pixel_data - min)/(max - min + 1e-10))

        return pixel_data
    
    def rotate_RAS(self, image):
        #image = self.normalize(image)
        
        '''sagittal = (round(direction[0]), round(direction[1]), round(direction[2])) #Width, 2 (Lateral)
        coronal = (round(direction[3]), round(direction[4]), round(direction[5])) #Height, 1 (Front-Back)
        axial = (round(direction[6]), round(direction[7]), round(direction[8])) #Layers, 0   (Axial)
        
        'vectors = [axial, coronal, sagittal]'''
        #shape = [0, 1, 2], random.shuffle(shape)
        '''for i, vector in enumerate(vectors):
            if -1 in vector:
                image = numpy.flip(image, axis = vectors.index(vector))
            
            if vector in [(1, 0, 0), (-1, 0, 0)]:
                shape[i] = vectors.index(sagittal)
            if vector in [(0, 1, 0), (0, -1, 0)]:
                shape[i] = vectors.index(coronal)
            if vector in [(0, 0, 1), (0, 0, -1)]:
                shape[i] = vectors.index(axial)'''
        
        #if shape != [0, 1, 2]:
            #If transposed, matrix must be flipped to achieve the sensation of being rotated over plane
            #image = numpy.flip(numpy.transpose(image, tuple(shape)))
            
        for axes in [(1, 0)]: #[(0, 1), (0, 2), (1, 2)]:
            for n_rots in range(random.choice(range(4))):
                image = numpy.rot90(image, axes = axes)
         
        height, width = image.shape[0], image.shape[1]
        yroll, xroll = random.randint(-height//10, height//10), random.randint(-width//10, width//10)
        image = numpy.roll(image, yroll, axis = 0)
        image = numpy.roll(image, xroll, axis = 1)
        
        if yroll < 0:
            image[yroll:, :] = 0
        else:
            image[:yroll, :] = 0
        
        if xroll < 0:
            image[:, xroll:] = 0
        else:
            image[:, :xroll] = 0
        
        return image
    
    def transform(self, image):
        prob_hflip = random.random() > 0.5
        prob_blur = random.random() > 0.5
        rotation_angle = random.uniform(-25, 25)
        sigma = random.uniform(0, 1)

        if self.min_slices > 1:
            image = numpy.transpose(image, (1, 2, 0)) #HWC
        else:
            image = image[0]
        
        #if prob_hflip:
            #print('Flipping!')
            #image = numpy.flip(image, 1)
        #print('Rotating', rotation_angle, 'degrees!')
        if self.is_train:
            rot_mat = cv2.getRotationMatrix2D(tuple(numpy.array(image.shape[1::-1]) / 2), rotation_angle, 1.0)
            image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
           
            image = self.rotate_RAS(image)
           
            noise = aug.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255))
            image = noise.augment_image(image)

            bright = aug.Multiply((0.1, 2.0))
            image = bright.augment_image(image)
            '''contrast = aug.LinearContrast((0, 1.5))
            image = contrast.augment_image(image)'''
            
            if prob_blur:
                blur = aug.GaussianBlur(sigma)
                image = blur.augment_image(image)
        
        if self.min_slices > 1:
            image = numpy.transpose(image, (2, 0, 1))
        else:
            image = numpy.array([image])

        aug_tensor = torch.tensor(image.astype(numpy.float32))
        aug_tensor = TF.normalize(aug_tensor, tuple(image.shape[0]*[0]), tuple(image.shape[0]*[1]))
        return aug_tensor
    
    def __getitem__(self, idx):
        image, label = self.loaded_data[idx]
        path, _ = self.images.iloc[idx]
        #print(image.shape)
        #if self.is_train:
        first_slice_idx = numpy.random.randint(16 - self.min_slices + 1)
        last_slice_idx = first_slice_idx + self.min_slices
        image = image[first_slice_idx:last_slice_idx]
        #print(image.shape)
        return self.transform(image), torch.tensor(label), path
    
    def __len__(self):
        return len(self.loaded_data)
