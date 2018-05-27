import tensorflow as tf
import numpy as np
import os
import inference
import time 
import math 
batch_size = 50
max_steps=5000
model_path = 'model.ckpt'
def read_and_decode(filename):      
    filename_queue = tf.train.string_input_producer([filename]) #[filename]   
    reader = tf.TFRecordReader()      
    _, serialized_example = reader.read(filename_queue)           
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([], tf.int64),'img_raw' : tf.FixedLenFeature([],tf.string),})      
    img = tf.decode_raw(features['img_raw'], tf.uint8)      
    img = tf.reshape(img, [112, 112, 4])    
    img = tf.image.per_image_standardization(img)  
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5      
    label = tf.cast(features['label'], tf.int64)      
    return img, label

images_train, labels_train =read_and_decode('train35.tfrecords')
images_test, labels_test = read_and_decode('test35.tfrecords')
image_batch,label_batch = tf.train.shuffle_batch([images_train,labels_train],batch_size,capacity=6000,min_after_dequeue=500)
image_batch1,label_batch1 = tf.train.shuffle_batch([images_test,labels_test],batch_size,capacity=1200,min_after_dequeue=500)

sess = tf.InteractiveSession()

# setup siamese network
siamese = inference.siamese()
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
saver = tf.train.Saver()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()
print('start...')
# if you just want to load a previously trainmodel?
new = True
model_ckpt = 'model2/model.ckpt.index'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = raw_input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False


# start training
if new:
    for step in range(max_steps):
        start_time = time.time()
        batch_x1,batch_y1 = sess.run([image_batch,label_batch])
        batch_x2,batch_y2 = sess.run([image_batch,label_batch])

        batch_y = (batch_y1 == batch_y2).astype('float')

        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.x1: batch_x1, 
                            siamese.x2: batch_x2, 
                            siamese.y_: batch_y})
        duration = time.time() - start_time

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            break
            #quit()

        if step % 10 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            format_str = ('step %d,loss = %.2f(%.1f examples/sec; %.3f sec/batch)')
            print(format_str % (step,loss_v,examples_per_sec,sec_per_batch))
            #print ('step %d: loss %.3f' % (step, loss_v))

        if step % 500 == 0 :
            saver.save(sess, 'model2/model500.ckpt')
        if step % 1000 == 0 :
            saver.save(sess, 'model2/mode1000.ckpt')
        if step % 2000 == 0 :
            saver.save(sess, 'model2/mode2000.ckpt')
        if step % 3000 == 0 :
            saver.save(sess, 'model2/mode3000.ckpt')
        if step % 4000 == 0 :
            saver.save(sess, 'model2/mode4000.ckpt')


    save_path = saver.save(sess, model_path)  
    print("Model saved in file: %s" % save_path)

else:
    saver.restore(sess, 'model2/model5000.ckpt')



num_examples = 1500
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    batch_x4,batch_y4 = sess.run([image_batch1, label_batch1])
    batch_x5,batch_y5 = sess.run([image_batch1, label_batch1])
    batch_y6 = (batch_y4 == batch_y5).astype('float')
    #label_batch = tf.one_hot(label_batch,3,1,0)
    o1, o2 = sess.run([siamese.o1, siamese.o2], feed_dict={
                            siamese.x1: batch_x4, 
                            siamese.x2: batch_x5,
                            siamese.y_: batch_y6})
    cox=np.sqrt(np.sum(np.square((np.array(o1)-np.array(o2))),1))
    judge = np.where(cox<2.5,1,0)
    print(batch_y4,batch_y5,batch_y6)
    step_t=np.sum(judge==batch_y6)
    true_count = true_count + step_t
    #true_count += np.sum(predictions)
    print(cox,judge,true_count)
    step += 1
precision = float(true_count) / float(total_sample_count)
print(true_count, total_sample_count)
print('precision @ 1 = %.3f'%precision)

