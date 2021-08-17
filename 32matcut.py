
#import matplotlib.pyplot as plt
#from PIL import Image
import numpy as np
import os


# 数据矩阵转图片的函数
# def MatrixToImage(data):
#     data = data * 255
#     new_im = Image.fromarray(data.astype(np.uint8))
#     return new_im


num = 0
for file in open("deprecatedCode/120kv_160ma_30ma_0.5_lung_1mm_low_shuffle.txt"):
    line = file.rstrip()
    mat  = np.load(line)

    filepaths = line.split('120kv_160ma_30ma_0.5\\')
    pathname = filepaths[1].split('FMI')
    #print('pathname is %s', pathname[0])
    filename = os.path.basename(filepaths[1])
    first_name, second_name = os.path.splitext(filename)
    for i in range(15):
        for j in range(15):
            cut = mat[i * 32: i * 32 + 64,j * 32:j * 32 + 64]
            img = cut * 255
            img = img.astype(np.uint8)
            if np.mean(img) > 30.0:
                np.save('G:\\CT_ZL\\cut\\32cut64_120kv_160ma_30ma_0.5_1mm_1024\\low\\' + pathname[0].replace('\\', '_') + first_name + '_' + str(i) + '_' + str(j) + '.npy', cut)
                #save("../cut\\20ma_0.5_lung_1mm_low_cut\\" +pathname[0].replace('\\', '_') + first_name + "_" + str(i)+ str(j) + ".mat", {"imgsave": cut})

    num += 1
    if num%6 == 0:
        print(num, " finished")

print('finished')

for file in open("120kv_160ma_30ma_0.5_lung_1mm_normal_shuffle.txt"):
    line = file.rstrip()
    mat = np.load(line)

    filepaths = line.split('120kv_160ma_30ma_0.5\\')
    pathname = filepaths[1].split('FMI')
    filename = os.path.basename(filepaths[1])
    first_name, second_name = os.path.splitext(filename)
    for i in range(15):
        for j in range(15):
            cut = mat[i * 32: i * 32 + 64,j * 32:j * 32 + 64]
            img = cut * 255
            img = img.astype(np.uint8)
            if np.mean(img) > 30.0:
                np.save('G:\\CT_ZL\\cut\\32cut64_120kv_160ma_30ma_0.5_1mm_1024\\normal\\' + pathname[0].replace('\\', '_') + first_name + '_' + str(i) + '_' + str(j) + '.npy', cut)
                #save("../cut\\20ma_0.5_lung_1mm_normal_cut\\" +pathname[0].replace('\\', '_') + first_name + "_" + str(i)+ str(j) + ".mat", {"imgsave": cut})

    num += 1
    if num%6 == 0:
        print(num, " finished")

print('finished')