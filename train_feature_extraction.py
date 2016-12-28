import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from alexnet import AlexNet
import numpy as np
from sklearn.utils import shuffle
import time

# TODO: Load traffic signs data.
with open("train.p", mode='rb') as f:
    train = pickle.load(f)

# TODO: Split data into training and validation sets.
features = train['features']
labels = train['labels']

X_train, X_val, y_train, y_val = train_test_split(features, labels, 
                                                  test_size = 0.33, 
                                                  random_state = 2016)

graph = tf.Graph()

with graph.as_default():

# TODO: Define placeholders and resize operation.
    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, None)
    resized = tf.image.resize_images(x, [227,227])

# TODO: pass placeholder as first argument to `AlexNet`.
# By keeping `feature_extract` set to `True`
# we indicate to NOT keep the 1000 class final layer
# originally used to train on ImageNet.
    fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
    fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
    nb_classes = 43
    shape = (fc7.get_shape().as_list()[-1], nb_classes)  
    w_tz = tf.Variable(tf.truncated_normal(shape=shape, 
                                       stddev=tf.sqrt(2.0/shape[0])))
    b_tz = tf.zeros(shape[1])
    logits = tf.nn.xw_plus_b(fc7, w_tz, b_tz)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    
# TODO: Train and evaluate the feature extraction model.

acc_val = []
loss_val = []    

with tf.Session(graph=graph) as sess: 
    init = tf.global_variables_initializer()
    sess.run(init)
    
    nb_epochs = 1
    batch_size = 32
    
    def accuracy(pred, labels):
        return (np.sum(np.equal(np.argmax(pred, 1),labels)))/pred.shape[0]
        
    
    for epoch in range(nb_epochs):
        t0 = time.time()
        total_batch = np.int(X_train.shape[0]/batch_size)
        X_train, y_train = shuffle(X_train, y_train)
        for i in range(total_batch):
            offset = i * batch_size
            batch_x = X_train[offset:(offset+batch_size), ]/255.0
            batch_y = y_train[offset:(offset+batch_size)]
            sess.run([optimizer, train_prediction], 
                     feed_dict={x: batch_x, y: batch_y})
    
        for k in range(0, X_val.shape[0], 57):
            l, p = sess.run([loss, train_prediction], 
                            feed_dict={x: X_val[k:(k+57), ],
                                       y: y_val[k:(k+57)]})
            acc_val.append(accuracy(p, y_val[k:(k+57)]))
            loss_val.append(l)
        print("Epoch {}: ".format(epoch))
        print("Time spend: {}".format(time.time()-t0))
        print("Validation Loss: {}".format(np.mean(loss_val)))
        print("Validation Accuracy: {:.3%}".format(np.mean(acc_val)))

"""
(As a point of reference one epoch over the training set takes roughly 53-55 
seconds with a GTX 970.)
Epoch 0: 
Time spend: 932.5579879283905
Validation Loss: 16.838653564453125
Validation Accuracy: 21.470%
"""            
            
            
            