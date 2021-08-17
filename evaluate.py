import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom
import pydicom.uid
import sys
import numpy as np
import PIL.Image as Image
import scipy.io as scio
import os
#import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio as PSNR

def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


def get_pixel_hu(dicom):
    image = dicom.pixel_array
    #
    image = image.astype(np.int16) if not image.dtype == np.int16 else image

    #
    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def SetDicomWinWidthWinCenter(img, winwidth, wincenter, rows=512, cols=512):
    img_temp = img
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in np.arange(rows):
        for j in np.arange(cols):
            img_temp[i, j] = int((img_temp[i, j] - min) * dFactor)

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255

    return img_temp.astype(np.int16)


def mat2win(img, min=0, max=0.33, rows=512, cols=512):
    img[img < min] = min
    img[img > max] = max
    img_out_adjust = (img - min) * 255 / (max - min)

    return img_out_adjust


def denormalize_pixel(image, MIN=-1000, MAX=2000, RANGE=1):
    image = image / RANGE * (MAX - MIN) + MIN
    return image.astype(np.int16)


def cal_avg(img, x=330, y=140, scope=40):
    size = scope * scope
    average = 0
    for i in range(x, x + scope):
        for j in range(y, y + scope):
            average += img[i][j] / size
    return round(average, 2)

def cal_var(img, x=330, y=140, scope=40):
    var = np.std(img[x:x+scope, y:y+scope])
    return round(var, 2)


def showImgResults(img_source_file, windowsWidth, windowsCenter):
    if img_source_file.endswith('mat'):
        img_source_data = scio.loadmat(img_source_file)
        if img_source_file.find('out') >= 0:
            img_source = img_source_data['out_img']
        else:
            img_source=  img_source_data['imgsave']
    elif img_source_file.endswith('npy'):
        img_source = np.load(img_source_file)
    #img_source_adjust = mat2win(img_source, 0, 0.4) # lung window
    #img_source_adjust = mat2win(img_source, 0.2, 0.3) # soft tissue window
    img_source_hu = denormalize_pixel(img_source)
    img_source_adjust = SetDicomWinWidthWinCenter(img_source_hu, windowsWidth, windowsCenter)
    img_avg = cal_avg(img_source_hu)
    img_std = cal_var(img_source_hu)
    img_source_info = 'avg: ' + str(img_avg) + ' ' + 'std: ' + str(img_std)

    return img_source_adjust, img_source_info
def compute_psnr_ssim(im_true, im_test):
    img_true = np.load(im_true)
    img_test = np.load(im_test)

    ps = PSNR(img_true, img_test)
    ps = np.around(ps, 4)
    ss = structural_similarity(img_true * 255, img_test * 255, data_range=255)
    ss = np.around(ss, 4)
    return ps, ss

if __name__ =='__main__':
    #windowsWidth, windowsCenter = 1400, -500# lung window
    windowsWidth, windowsCenter = 400, -50# tissue window

    current_dir = '20210429_0'
    #hu in [-1000, 2000]
    #left, right = 0.25, 0.38 #tissue window
    #left, right = 0.0, 0.4 #lung window
    for file in open('mayo_test_2.txt'):
        #low image
        img_low_file = file.rstrip()
        img_file_name = os.path.basename(img_low_file)

        img_low_adjust, img_low_info =  showImgResults(img_low_file, windowsWidth, windowsCenter)
    
        #generated image
        img_out_file =  current_dir + '/outimg_' + img_file_name.replace('L', 'H')
        img_out_adjust, img_out_info =  showImgResults(img_out_file, windowsWidth, windowsCenter)

        #normal image
        img_normal_file = img_low_file.replace('L', 'H')
        print(img_normal_file)
        img_normal_adjust, img_normal_info =  showImgResults(img_normal_file, windowsWidth, windowsCenter)

        psnr_low, ssim_low = compute_psnr_ssim(img_normal_file, img_low_file)
        psnr_out, ssim_out = compute_psnr_ssim(img_normal_file, img_out_file)
        psnr_normal, ssim_normal = 'inf', 1.0#compute_psnr_ssim(img_normal_file, img_normal_file)
        #plt.figure(figsize=(24, 24))
        img_list = [img_low_adjust, img_out_adjust, img_normal_adjust]
        title_list = ['low', 'genereted', 'normal']
        info_list = [img_low_info, img_out_info, img_normal_info]
        psnr_list = [psnr_low, psnr_out, psnr_normal]
        ssim_list = [ssim_low, ssim_out, ssim_normal]

        for i in range(3):
            ax = plt.subplot(1, 3, i + 1)
            plt.title(title_list[i])
            rect = patches.Rectangle((140, 370), 40, 40, linewidth=1.0, edgecolor='r', fill=False)
            ax.add_patch(rect)
            plt.imshow(img_list[i], cmap=plt.cm.gray)
            plt.xlabel(info_list[i] + '\n' + 'psnr:' + str(psnr_list[i]) + '\n' + 'ssim:' + str(ssim_list[i]))
            plt.xticks([])
            plt.yticks([])

        plt.subplots_adjust(left=0.01, top=0.88, right=0.98, bottom=0.08, wspace=0.02, hspace=0.02)
        plt.show()
