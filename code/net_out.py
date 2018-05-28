import tensorflow as tf
import numpy as np
import os,os.path
import inference
import time 
import math 
import pandas as pd
import scipy.misc

result = np.zeros((0,5))
new = True
model_ckpt = 'model2s/model.ckpt.index'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = raw_input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False
sess = tf.InteractiveSession()
siamese = inference.siamese()
saver = tf.train.Saver()
tf.global_variables_initializer().run()

if new:
    print("can not find model")
else:
    saver.restore(sess, 'model2s/model.ckpt')
    pathdir = 'dataset3/test5/0'
    filelist = os.listdir(pathdir)
    for i in range(0,97):
        path = os.path.join(pathdir,filelist[i])
        if os.path.isfile(path):
            print("%s"%path)
            image_raw_data = tf.gfile.FastGFile(path).read()
            #a = scipy.misc.imread(path)
            image = tf.image.decode_png(image_raw_data)
            inputim = sess.run([image])
            embed = siamese.o1.eval({siamese.x1: inputim})
            embed = np.array(embed)
            embed = np.insert(embed,4,values=0,axis = 1)
            result = np.row_stack((result,embed))

    pathdir = 'dataset3/test5/1'
    filelist = os.listdir(pathdir)
    for i in range(0,433):
        path = os.path.join(pathdir,filelist[i])
        if os.path.isfile(path):
            print("%s"%path)
            image_raw_data = tf.gfile.FastGFile(path).read()
            #a = scipy.misc.imread(path)
            image = tf.image.decode_png(image_raw_data)
            inputim = sess.run([image])
            embed = siamese.o1.eval({siamese.x1: inputim})
            embed = np.array(embed)
            embed = np.insert(embed,4,values=1,axis = 1)
            result = np.row_stack((result,embed))

    pathdir = 'dataset3/test5/2'
    filelist = os.listdir(pathdir)
    for i in range(0,409):
        path = os.path.join(pathdir,filelist[i])
        if os.path.isfile(path):
            print("%s"%path)
            image_raw_data = tf.gfile.FastGFile(path).read()
            #a = scipy.misc.imread(path)
            image = tf.image.decode_png(image_raw_data)
            inputim = sess.run([image])
            embed = siamese.o1.eval({siamese.x1: inputim})
            embed = np.array(embed)
            embed = np.insert(embed,4,values=2,axis = 1)
            result = np.row_stack((result,embed))

    pathdir = 'dataset3/test5/3'
    filelist = os.listdir(pathdir)
    for i in range(0,167):
        path = os.path.join(pathdir,filelist[i])
        if os.path.isfile(path):
            print("%s"%path)
            image_raw_data = tf.gfile.FastGFile(path).read()
            #a = scipy.misc.imread(path)
            image = tf.image.decode_png(image_raw_data)
            inputim = sess.run([image])
            embed = siamese.o1.eval({siamese.x1: inputim})
            embed = np.array(embed)
            embed = np.insert(embed,4,values=3,axis = 1)
            result = np.row_stack((result,embed))

print('finish!')

#np.savetxt('data.txt',result,delimiter=' ',newline='\n')

data1 = pd.DataFrame(result)
data1.rename(columns={'Unnamed: 0':'id'})
data1.rename(columns={'Unnamed: 1':'labels'})
data1.to_csv('tests500.csv')
'''
comat = np.zeros((1386,1386))
for i in range(0,1386):
    for j in range(0,1386):
        cox=np.sqrt(np.sum(np.square((result[i]-result[j])),0))
        comat[i,j] = cox
'''
'''
codata1 = pd.DataFrame(comat)
codata1.to_csv('codata2.csv')
scipy.misc.imsave('2.jpg', comat)
cv2.imwrite('cv2.png', 255*comat/float(comat.max()))
print(comat.mean())

judge = np.where(comat<20,0,1)

cv2.imwrite('cv2j.png', 255*judge/float(judge.max()))


#retrieval
judge2=np.zeros((1386))
comat1 = np.where(comat==0,255,comat)
for i in range(0,1386):
    t = comat1[i].argmin()
    if 0<=t<98:
        judge2[i]=1
    elif 98<=t<98+432:
        judge2[i] =2 
    elif 98+432<=t<98+432+447:
        judge2[i] =3
    else:
        judge2[i] =4

print(judge2) 
np.savetxt('judge2.txt',judge2)
'''
