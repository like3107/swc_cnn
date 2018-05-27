#from pyzem.swc import swc
import swc
import dataprovider
import cv2
import os,os.path
import numpy as np 
import scipy.misc
import transfer
print('start data preprocessing')
'''
tree = swc.SwcTree()
dp = dataprovider.dataprovider()
dp.load(tree)
dp.get_level_list()
dp.get_length_t()
dp.get_radius_t()
dp.get_plocation_t()
dp.get_angle_t()
dp.get_type_t()
dp.tran_file('../celldata/GABAergic')
dp.tran_file('../celldata/granule01')
dp.tran_file('../celldata/nitrergic')
dp.tran_file('../celldata/pyrimidal')
'''
print("step1 complete!")
pathdir = ['data/GABAergic','data/granule01','data/nitrergic','data/pyrimidal']
for i in range(0,len(pathdir)):
    filelist = os.listdir(pathdir[i])
    for j in range(0,len(filelist)):
        path = os.path.join(pathdir[i],filelist[j])
        if os.path.isfile(path):
            print("%s"%path)
            a = scipy.misc.imread(path)
            if j%5 == 1:
                scipy.misc.imsave('test/'+'%s'%i+'/'+'%s.png'%path[15:][:-4],a)
            else :
                scipy.misc.imsave('train/'+'%s'%i+'/'+'%s.png'%path[15:][:-4],a)
print("step2 complete!")
transfer.get_input('test')
transfer.get_input('train')
print("step3 complete!")