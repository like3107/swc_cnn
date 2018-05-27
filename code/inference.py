import tensorflow as tf 
import numpy as np
import time 
import math

class siamese:

    # Create model
    def __init__(self):
        self.batch_size = 1
        self.x1 = tf.placeholder(tf.float32, [self.batch_size,112,112,4])
        self.x2 = tf.placeholder(tf.float32, [self.batch_size,112,112,4])

        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1)
            scope.reuse_variables()
            self.o2 = self.network(self.x2)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [self.batch_size])
        self.loss = self.loss_with_spring()

    def network(self, x):
        #tf.contrib.layers.fully_connected(inputs=x,num_outputs=1024,scope="siamese")
        
        weights = []
        '''
        conv1 = self.conv_op(x,"conv1",3,3,4,64)
        ac3 = tf.nn.relu(conv1)
        conv2 = self.conv_op(ac3,"conv2",3,3,64,64)
        ac4 = tf.nn.relu(conv2)
        pool1 = self.mpool_op(ac4,"pool1", 2, 2, 2, 2)

        conv3 = self.conv_op(pool1,"conv3",3,3,64,128)
        ac5 = tf.nn.relu(conv3)
        conv4 = self.conv_op(ac5,"conv4",3,3,128,128)
        ac6 = tf.nn.relu(conv4)
        pool2 = self.mpool_op(ac6,name="pool2", kh =2, kw = 2, dh = 2, dw = 2)

'''
        reshape = tf.reshape(x,[self.batch_size,-1])

        fc1 = self.fc_layer(reshape, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 4, "fc3")
        
        return fc3

    def fc_layer(self, bottom, n_weight, name):
        #assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

    def conv_op(self, input_op, name, kh, kw, n_in, n_out):
        W = tf.get_variable(name+"W",
            shape = [kh,kw,n_in,n_out], dtype = tf.float32,
            initializer = tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(input_op, W,[1, 1, 1, 1], padding='SAME')
        b = tf.get_variable(name+'b', dtype = tf.float32, initializer=tf.constant(0.0, shape = [n_out], dtype = tf.float32))
        z = tf.nn.bias_add(conv,b)
        return z

    def mpool_op(self,input_op, name, kh, kw, dh, dw):
        return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding = 'SAME',name=name)

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
