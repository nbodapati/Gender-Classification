import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import pandas as pd
import numpy as np
import urllib
import numpy as np
import tensorflow as tf
import pickle 
import math
import json

data = pickle.load( open( "dataset.p", "rb" ) )

train_maleX=data['x_train']
train_maleY=data['y_train']
train_maleY1=data['y_train1']
train_inits=data['initials']
train_names=data['names']


print(train_maleX.shape,train_maleY.shape)
N=train_maleX.shape[0]
male=(train_maleY==1).reshape(train_maleY.shape[0],)
female=(train_maleY==0).reshape(train_maleY.shape[0],)

male_features=train_maleX[male,:]
male_labels=train_maleY[male]

female_features=train_maleX[female,:]
female_labels=train_maleY[female]
print(np.sum(male),np.sum(female))

N=np.random.choice(male_features.shape[0],int(female_features.shape[0]*0.8)+100)
Not=[]
for i in range(male_features.shape[0]):
    if(i not in N):
       Not.append(i)

N2=int(0.8*female_features.shape[0])

m2=np.random.choice(male_features[Not,:].shape[0],female_labels[N2+1:,:].shape[0])
X=np.zeros((len(male_features[N,:])+len(female_features[:N2,:]),3600))
Y=np.zeros((len(male_features[N,:])+len(female_features[:N2,:]),1))

mf=male_features[N,:]
ff=female_features[:N2,:]
i=0
for i_ in range(0,len(mf)+len(ff),2):
    if(i==len(mf)):
      break
    X[i_,:]=mf[i,:]
    Y[i_]=1
    print(i_,X[i_,:],Y[i_])
    i+=1

i=0
for i_ in range(1,len(mf)+len(ff),2):
    if(i==len(ff)):
      break
    X[i_,:]=ff[i,:]
    Y[i_]=0
    print(i_,X[i_,:],Y[i_])
    i+=1 
#X=np.vstack((male_features[N,:],female_features[:N2,:]))
#Y=np.vstack((male_labels[N],female_labels[:N2]))
#print(X)
#print(Y)
X=(X-np.mean(X))/np.std(X,axis=0)

'''
testX=np.zeros((len(male_features[m2,:])+len(female_features[N2+1:,:]),3600))
testY=np.zeros((len(male_features[m2,:])+len(female_features[N2+1:,:]),1))

mf=male_features[m2,:]
ff=female_features[N2+1:,:]

i=0
for i_ in range(0,len(mf)+len(ff),2):
    if(i==len(mf)):
      break
    testX[i_,:]=mf[i,:]
    testY[i_]=1
    i+=1

i=0
for i_ in range(1,len(mf)+len(ff),2):
    if(i==len(ff)):
      break
    testX[i_,:]=ff[i,:]
    testY[i_]=0
    i+=1
'''
testX=np.vstack((male_features[m2,:],female_features[N2+1:,:]))
testY=np.vstack((male_labels[m2],female_labels[N2+1:]))
print(testX,testY)
testX=(testX-np.mean(X))/np.std(testX,axis=0)

test_names=[]
for i in m2:
    test_names.append(train_names[i])

for i in range(N2+1,female_labels.shape[0]):
    test_names.append(train_names[i])

print(test_names,len(test_names))

print("Shape of X and Y: ",X.shape,Y.shape)
print("Shape of Xtest and Ytest: ",testX.shape,testY.shape)

X=np.ndarray.astype(X,'float32')
testX=np.ndarray.astype(testX,'float32')

Y=np.ndarray.astype(Y,'int64')
testY=np.ndarray.astype(testY,'int64')

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

# Hyper Parameters 
input_size = 3600
hidden_size1 = 5000
hidden_size2 = 500
num_classes = 2
num_epochs = 50
batch_size = 100
learning_rate = 0.0001

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1) 
        self.relu = nn.ReLU()
        self.dropout=nn.Dropout()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2) 
        self.fc3 = nn.Linear(hidden_size2, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out1 = self.relu(out)
        out1=self.dropout(out1)
        out = self.fc2(out1)
        out2 = self.relu(out)
        out2=self.dropout(out2)
        out3 = self.fc3(out2)
        return out3,out2,out1
    
net = Net(input_size, hidden_size1,hidden_size2, num_classes)
net.load_state_dict(torch.load('model.pkl'))
    
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

def test_train_model():
    # Test the Model
    correct = 0
    total = 0
    images = Variable(torch.FloatTensor(X))
    labels=Y.reshape(Y.shape[0],)#Variable(torch.LongTensor(testY.reshape(testY.shape[0],)))
    outputs,_,_ = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.shape[0]
    predicted=predicted.numpy()
    print(predicted)
    print(labels)
    correct= (predicted == labels).sum()
    print(correct,total)
    print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))

def test_model():
    # Test the Model
    correct = 0
    total = 0
    images = Variable(torch.FloatTensor(testX))
    labels=testY.reshape(testY.shape[0],)#Variable(torch.LongTensor(testY.reshape(testY.shape[0],)))
    outputs,_,_ = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.shape[0]
    predicted=predicted.numpy()
    print(predicted)
    print(labels)
    correct= (predicted == labels).sum()
    print(correct,total)
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Train the Model
for epoch in range(num_epochs):
    total_batch = int(X.shape[0]/batch_size)
    for i in range(total_batch):
        batch_x, batch_y = next_batch(batch_size,i)
        k=list(range(batch_x.shape[0]))
        np.random.shuffle(k)
        batch_x=batch_x[k,:]
        batch_y=batch_y[k]
        # Convert torch tensor to Variable
        images = Variable(torch.FloatTensor(batch_x))
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs,_,_ = net(images)
        labels=Variable(torch.LongTensor(batch_y.reshape(batch_y.shape[0],)))
        loss = criterion(outputs, labels)
       # print("epoch: ",epoch, "i: ",i,"loss: ",loss.data[0]))
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        predicted=predicted.numpy()
        labels=labels.data.numpy()
        #print(predicted)
        #print(labels)
        correct= (predicted == labels).sum()
        #print(correct,total)
        #print('Accuracy of the network on the train batch (%d:%d) images: %d %%' % (epoch,i,100 * correct / total))

            
    print ('*******************************Epoch [%d/%d], Loss: %.4f' 
              %(epoch+1, num_epochs, loss.data[0]))
    test_train_model()
    test_model() 
    

l2_activations=[]
prob_dist=[]
error=[]
true_class=[]

def softmax(n):
    exp_n=np.exp(n)
    sum_exp_n=np.sum(exp_n)
    return (exp_n/sum_exp_n)

for i in range(testX.shape[0]):
    o3,o2,o1=net(Variable(torch.FloatTensor(testX[i,:])))
    l2_activations.append(o2.data.numpy())
   # print(l3_activations)
    pd=softmax(o3.data.numpy())
    prob_dist.append(pd)
    #print(pd)
    error.append(-math.log(pd[testY[i]]))
   # print(error)
    true_class.append(testY[i])
    #print(true_class)     

with open('result.json','w')as fp:    
     json.dump({'l2_act':str(l2_activations),'prob_dist':str(prob_dist),'error':str(error),'true_class':str(true_class),'names':str(test_names)},
                 fp)    



# Save the Model
torch.save(net.state_dict(), 'model.pkl')
