import os  
import tensorflow as tf  
#import numpy as np 
from PIL import Image

    
def get_input(path):
    classes = os.listdir(path)  
    print(classes)

    writer = tf.python_io.TFRecordWriter("%s.tfrecords"%path)  
    for index, name in enumerate(classes):  
        class_path = path+"/" + name
        print(class_path)
        if os.path.isdir(class_path):  
            for img_name in os.listdir(class_path):  
                img_path = class_path + '/' + img_name  
                img = Image.open(img_path)   
                img = img.resize((112, 112))
                img_raw = img.tobytes()              
                example = tf.train.Example(features=tf.train.Features(feature={  
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),  
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))  
                }))  
                writer.write(example.SerializeToString())  
                print(img_name,name)
    writer.close()  
            

'''
for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):  
    example = tf.train.Example()  
    example.ParseFromString(serialized_example)  
  
    image = example.features.feature['image'].bytes_list.value  
    label = example.features.feature['label'].int64_list.value   
    print image, label

'''
'''
writer = tf.python_io.TFRecordWriter("train1.tfrecords")  
for i in range(100):  
    name = np.random.randint(1,4)
    img = np.random.randint(1,100,[112,112,4])
    #img = img.resize((112, 112))
    img_raw = img.tobytes()              
    example = tf.train.Example(features=tf.train.Features(feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])), 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))  
    writer.write(example.SerializeToString())  
    writer.close()  
    print(i)

'''