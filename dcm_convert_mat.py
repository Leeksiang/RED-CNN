import os
import pydicom
import numpy as np
from PIL import Image
import scipy.io as sio

"""
This python code is to convert the dcm file into numpy format
"""


def get_pixels_hu(dicom):
    image = dicom.pixel_array
    # 转换为int16
    image = image.astype(np.int16) if not image.dtype == np.int16 else image

    # 转换为HU单位
    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def normalize_pixel(image, MIN=-1000, MAX=2000, RANGE=1):
    image = (image - MIN) / (MAX - MIN) * RANGE
    return image.astype(np.float32)


def denormalize_pixel(image, MIN=-1000, MAX=2000, RANGE=1):
    image = image / RANGE * (MAX - MIN) + MIN
    return image.astype(np.int16)


def trunc(image, MIN=0.0, MAX=1.0):
    image[image < MIN] = MIN
    image[image > MAX] = MAX
    return image.astype(np.float32)


def dcm2npy(path_to_file):
    (path_and_filename, extension) = os.path.splitext(path_to_file)
    dcm = pydicom.dcmread(path_to_file)
    arr = get_pixels_hu(dcm)
    img = trunc(normalize_pixel(arr))
    np.save(path_and_filename+'.npy', img)
    #np_file = np.load('E:/PostGraduate/Master/Medical/CTData/120kv_160ma_20ma_0.5/DPC004476/lung/1mm/low/FMI000001.npy')
    #print("is it ok?")
    #sio.savemat(path_and_filename + '.mat', {'imgsave':img})

def mat2png(path_to_file):
    (path_and_filename, extension) = os.path.splitext(path_to_file)
    arr = np.load(path_to_file)
    img = Image.fromarray(arr*255).convert('L')
    img.save(path_and_filename+'.png')


if __name__ == "__main__":
    counter = 0
    print('dcm to npy')
    for root, dirs, files in os.walk('G:\\120kv_160ma_30ma_0.5'):
        for file in files:

            if file.endswith('.dcm'):
                path_to_file = os.path.join(root, file)
                try:
                    dcm2npy(path_to_file)
                except Exception:
                    print('convert ' + path_to_file + ' failed')
                counter += 1
                if counter % 500 == 0:
                    print('complete file: ' + str(counter))