from __future__ import division, print_function, absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import urllib
import numpy as np
import tensorflow as tf
import pickle 
import math

data = pickle.load( open( "dataset.p", "rb" ) )

train_maleX=data['x_train']
train_maleY=data['y_train']
train_maleY1=data['y_train1']
train_inits=data['initials']
train_names=data['names']


print("Features: ",train_maleX)
print("labels: ",train_maleY)
print("onehot enc: ",train_maleY1)
#print("inits: ",train_inits)
#print("names: ",train_names)


print(train_maleX.shape,train_maleY.shape)
N=train_maleX.shape[0]
male=(train_maleY==1).reshape(train_maleY.shape[0],)
female=(train_maleY==0).reshape(train_maleY.shape[0],)

male_features=train_maleX[male,:]
male_labels=train_maleY1[male,:]

female_features=train_maleX[female,:]
female_labels=train_maleY1[female,:]
print(np.sum(male),np.sum(female))

N=np.random.choice(male_features.shape[0],int(female_features.shape[0]*0.8)+100)
Not=[]
for i in range(male_features.shape[0]):
    if(i not in N):
       Not.append(i)

N2=int(0.8*female_features.shape[0])

m2=np.random.choice(male_features[Not,:].shape[0],female_labels[N2+1:,:].shape[0])
X=np.vstack((male_features[N,:],female_features[:N2,:]))
Y=np.vstack((male_labels[N,:],female_labels[:N2,:]))

testX=np.vstack((male_features[m2,:],female_features[N2+1:,:]))
testY=np.vstack((male_labels[m2,:],female_labels[N2+1:,:]))

print("Shape of X and Y: ",X.shape,Y.shape)
print("Shape of Xtest and Ytest: ",testX.shape,testY.shape)
print(male_labels[m2,:].shape,female_labels[N2+1:,:].shape)

Not=[]
for i in range(male_features.shape[0]):
    if(i not in m2):
       Not.append(i)


print(np.sum(np.argmax(Y,1)==1))
print(np.sum(np.argmax(Y,1)==0))

print("Final shape: ",X.shape,Y.shape)
# Parameters
learning_rate = 1000
batch_size = 100
display_step = 1
model_path = "./models/"

# Network Parameters
n_hidden_1 = 4000 # 1st layer number of features
n_hidden_2 = 500 # 2nd layer number of features
n_hidden_3 = 500 # 2nd layer number of features
n_input = 3600 # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

#Input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_1 = tf.layers.dropout(layer_1, rate=0.8, training=True)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    layer_2 = tf.layers.dropout(layer_2, rate=0.75, training=True)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    # Output layer with linear activation
    layer_3 = tf.layers.dropout(layer_3, rate=0.75, training=True)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    #out_layer=tf.nn.softmax(out_layer)
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.001)),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.001)),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],stddev=0.001)),
    'out': tf.Variable(tf.truncated_normal(([n_hidden_3, n_classes]))),
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'out': tf.Variable(tf.truncated_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

#get next batches
def next_batch(batch_size,i):
    global X,Y
    max_len=X.shape[0]
    i=i*batch_size
    next_b=None
    if(i+batch_size>max_len):
       next_b=(X[i:,:],Y[i:,:])
    else:
       next_b=(X[i:i+batch_size,:],Y[i:i+batch_size,:])
    return next_b           


#Running first session
print("Starting 1st session...")
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    # Training cycle
    for epoch in range(1000):
        avg_cost = 0.0
        total_batch = int(X.shape[0]/batch_size)+1
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = next_batch(batch_size,i)
            k=list(range(batch_x.shape[0]))
            np.random.shuffle(k)
            batch_x=batch_x[k,:]
            batch_y=batch_y[k] 
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            #print("accuracy on train: ",accuracy.eval({x:batch_x, y:batch_y}))
            print("ce: ",sess.run(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y),{x:testX,y:testY}))
            # Compute average loss
            avg_cost += c
            #print("Average cost per batch: ",i,c)
        # Display logs per epoch step
        avg_cost/=total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))

        V=tf.trainable_variables()
        #print(sess.run(weights['h1']),sess.run(weights['h3']))  
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        print("Pred argmax: ",sess.run(pred,{x:testX,y:testY}))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x:testX, y: testY}))
        print("Pred: ",sess.run(tf.reduce_sum(tf.argmax(pred, 1)),{x:testX,y:testY}))
        print("y: ",sess.run(tf.reduce_sum(tf.argmax(y, 1)),{x:testX,y:testY}))
        print("correct pred: ",sess.run(tf.cast(correct_prediction,"float"),{x:testX,y:testY}))
        print("y: ",sess.run((tf.argmax(y, 1)),{x:testX,y:testY}))
        #Save model weights to disk
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)



'''
# Running a new session
#print("Starting 2nd session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)
    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
    print("Model restored from file: %s" % save_path)

    # Resume training
    for epoch in range(7):
        avg_cost = 0.
        total_batch =int(X.shape[0]/batch_size)+1
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y=next_batch(batch_size,i)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Second Optimization Finished!")
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: testX, y: testY}))
'''


