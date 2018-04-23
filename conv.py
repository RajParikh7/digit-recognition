# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:35:19 2018

@author: rajpa
"""

import tensorflow as tf
from sklearn import preprocessing as sl
import numpy as np


x=tf.placeholder(tf.float32,shape=(None,784),name='x')
y=tf.placeholder(tf.float32,shape=(None,10),name='y')
def parse_data(filename):
    f=open(filename)
    file=f.read()
    f1=file.split("\n")
    
    X=np.ndarray(shape=(len(f1)-2,784),dtype=float, order='F')#for storing training data
    labels=np.ndarray(shape=(len(f1)-2,1),dtype=float, order='F')#for storing labels
    
    '''starting from index 1 to avoid header column,
    removing two rows(header and last empty row) thus subtracted 2 from len(f1)'''
    for i in range(1,len(f1)-2):     
        x=f1[i].split(",")
        X[i,:]=x[1:]
        labels[i]=x[0]
    return X,labels
    f.close() 
def parameters():
    
    f={
       'f1':tf.Variable(tf.random_normal([5,5,1,32],seed=7),'f1'),
       'b1':tf.Variable(tf.random_normal([32],seed=7),'b1'),
       'f2':tf.Variable(tf.random_normal([5,5,32,64],seed=7),'f2'),
       'b2':tf.Variable(tf.random_normal([64],seed=7),'b2'),
       'f3':tf.Variable(tf.random_normal([1024,128],seed=7),'f3'),
       'b3':tf.Variable(tf.random_normal([128],seed=7),'b3'),       
       'f4':tf.Variable(tf.random_normal([128,10],seed=7),'f4'),
       'b4':tf.Variable(tf.random_normal([10],seed=7),'b4'),       
       }    
    return f

def conv_net(x,f):
   
    
    
    
    x=tf.reshape(x,shape=(-1,28,28,1))
    out_1 = tf.nn.conv2d(x,f['f1'],[1,1,1,1],padding='VALID')
    out_1 = tf.nn.relu(out_1 + f['b1'])
    out_1 = tf.nn.max_pool(out_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    out_2 = tf.nn.conv2d(out_1,f['f2'],[1,1,1,1],padding='VALID')
    out_2 = tf.nn.relu(out_2 + f['b2'])
    out_2 = tf.nn.max_pool(out_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    out_2 = tf.reshape(out_2,shape=(-1,1024))
    
    out_3 = tf.matmul(out_2,f['f3']) 
    out_3 = tf.nn.relu(out_3)
    
    y_ = tf.matmul(out_3,f['f4']) 
    #y_ = tf.nn.softmax(y_)
    #y_=tf.clip_by_value(y_,1e-10,0.9)
    return y_
    
def train_network(X,labels):
    

    #train_writer = tf.summary.FileWriter( 'C:/Users/rajpa/Desktop/graphs ', graph=tf.get_default_graph())
    el = sl.OneHotEncoder(sparse=False)
    labels = el._fit_transform(labels)
    learning_rate=0.01
    epochs=0
    batch_size=100
    f=parameters()
    y_ = conv_net(x,f)    
   
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_,labels= y))
    
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    tf.summary.scalar("cross_entropy",cross_entropy)
    
    with tf.Session() as sess:  
        total = int(len(labels)/batch_size)
        sess.run(tf.initialize_all_variables())
        #merged = tf.summary.merge_all()

        for epoch in range(epochs):
            loss=0
            for i in range(total):
                x_batch,y_batch = X[i*batch_size:(i+1)*batch_size,:],labels[i*batch_size:(i+1)*batch_size,:]
                #x_batch= np.reshape(x_batch,[-1,28,28,1])
                #y_= sess.run([y_],feed_dict={x:x_batch})
                #print(y_)
                _,cost = sess.run([optimizer,cross_entropy],feed_dict={x:x_batch,y:y_batch})
                #train_writer.add_summary(summary_str, epoch*total + i)
                loss+=cost/total
            print("epoch:",epoch,"loss:",loss)
        #X,labels=parse_data('train.csv')
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accr=sess.run([accuracy],feed_dict={x:X,y:labels})
        print("accr: ",accr)
        #yx=sess.run([y_],feed_dict={x:X[0:5,:]})
        #print(yx)
        #print(np.argmax(np.array(yx)))
X,labels=parse_data('train.csv')
train_network(X,labels)       

    