import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import defaultdict
import pandas as pd
import numpy as np
import numpy as np
import tensorflow as tf
import pickle 
import math
import json

import torch.nn.functional as F

data = pickle.load( open( "dataset.p", "rb" ) )

train_maleX=data['x_train']
train_maleY=data['y_train']
train_maleY1=data['y_train1']
train_inits=data['initials']
train_names=data['names']

server_address='http://www-edlab.cs.umass.edu/~nbodapati/'

print(train_maleX.shape,train_maleY.shape)
N=train_maleX.shape[0]

male=(train_maleY==1).reshape(train_maleY.shape[0],)
female=(train_maleY==0).reshape(train_maleY.shape[0],)

male_features=train_maleX[male,:]
male_labels=train_maleY[male]
male_names=[]
female_names=[]
for i in range(train_maleX.shape[0]):
    if(train_maleY[i]==1):
       male_names.append(server_address+train_names[i]) 
    else:
       female_names.append(server_address+train_names[i]) 
 
print("Number of male_names: ",len(male_names))
print("Number of female_names: ",len(female_names))

female_features=train_maleX[female,:]
female_labels=train_maleY[female]

print(np.sum(male),np.sum(female))

N=np.random.choice(male_features.shape[0],int(female_features.shape[0]*0.9),replace=False)
Not=[]
for i in range(male_features.shape[0]):
    if(i not in N):
       Not.append(i)

N2=int(0.9*female_features.shape[0])
m2=np.random.choice(Not,female_labels[N2+1:,:].shape[0],replace=False)


Not_m2=[]
for i in Not:
    if(i not in m2):
       Not_m2.append(i)


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
    #print(i_,X[i_,:],Y[i_])
    i+=1

i=0
for i_ in range(1,len(mf)+len(ff),2):
    if(i==len(ff)):
      break
    X[i_,:]=ff[i,:]
    Y[i_]=0
   # print(i_,X[i_,:],Y[i_])
    i+=1 
X=(X-np.mean(X))/np.std(X,axis=0)

testX=np.vstack((male_features[m2,:],female_features[N2+1:,:]))
testY=np.vstack((male_labels[m2],female_labels[N2+1:]))
testX=(testX-np.mean(X))/np.std(testX,axis=0)


testX2=male_features[Not_m2,:]
testY2=male_labels[Not_m2]

print("testX2 shape:",testX2.shape,testY2.shape)

test_names=[]
for i in m2:
    test_names.append(male_names[i])

for i in range(N2+1,female_labels.shape[0]):
    test_names.append(female_names[i])

print("Length of test names: ",len(test_names))
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
num_epochs = 100
batch_size = 100
learning_rate = 0.1

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0001, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}in {}'.format(lr,epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 30,stride=2)
        self.pool = nn.MaxPool2d(4, 4)
        self.dropout=nn.Dropout(p=0.5)
        #self.conv2 = nn.Conv2d(8, 16, 5)
        #self.conv3 = nn.Conv2d(16, 32, 5)
        #self.fc1 = nn.Linear(32 * 4 * 4, 120)
        #self.fc2 = nn.Linear(120, 10)
        self.fc3 = nn.Linear(16*4*4, 2)

    def forward(self, x):
        hidden=defaultdict(list)
        x = self.pool(F.relu(self.conv1(x)))
        hidden['conv-pool1'].append(x.data.numpy().tolist())
        #x = self.pool(F.relu(self.conv2(x)))
        #hidden['conv-pool2'].append(x.data.numpy().tolist())
        #x = self.pool(F.relu(self.conv3(x)))
        #hidden['conv-pool3'].append(x.data.numpy().tolist())
        x =self.dropout(x.view(-1, 16 *4 *4))
        #x = self.dropout(F.relu(self.fc1(x)))
        #hidden['fc1'].append(x.data.numpy().tolist())
        #x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        hidden['fc3'].append(x.data.numpy().tolist())
        return (x,hidden)


net = Net()
   
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

def test_train_model():
    # Test the Model
    correct = 0
    total = 0
    images = Variable(torch.FloatTensor(X).view(-1,1,60,60))
    labels=Y.reshape(Y.shape[0],)
    #Variable(torch.LongTensor(testY.reshape(testY.shape[0],)))
    outputs,hidden = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.shape[0]
    predicted=predicted.numpy()
    correct= (predicted == labels).sum()
    print(correct,total)
    print('Accuracy of the network on the train images: %d %%' % (100 * correct / total))

def test_model():
    # Test the Model
    correct = 0
    total = 0
    images = Variable(torch.FloatTensor(testX).view(-1,1,60,60))
    labels=testY.reshape(testY.shape[0],)
    #Variable(torch.LongTensor(testY.reshape(testY.shape[0],)))
    outputs,hidden = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.shape[0]
    predicted=predicted.numpy()
    correct= (predicted == labels).sum()
    print(correct,total)
    acc=correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
   
    outputs=outputs.data.numpy()
    return (outputs,hidden,labels,acc)


def test_model2():
    correct = 0
    total = 0
    images = Variable(torch.FloatTensor(testX2).view(-1,1,60,60))
    labels=testY2.reshape(testY2.shape[0],)
    outputs,hidden= net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.shape[0]
    predicted=predicted.numpy()
    correct= (predicted == labels).sum()
    print(correct,total)
    print('Accuracy of the network on the test2 images: %d %%' % (100 * correct / total))

def softmax(n):
    exp_n=np.exp(n)
    sum_exp_n=np.sum(exp_n,axis=1,keepdims=True)
    return (exp_n/sum_exp_n)

def compute_error(outputs,labels):
    print(outputs.shape,len(labels))
    outputs_=softmax(outputs)
    output=[]
    error=[]
    for i in range(outputs.shape[0]):
        output.append(outputs_[i,:].tolist())
        error.append(-math.log(outputs_[i,labels[i]]))   
    return (output,error)


#net.load_state_dict(torch.load('model_cnn.pkl'))
# Train the Model
for epoch in range(num_epochs):
    total_batch = int(X.shape[0]/batch_size)
    optimizer=exp_lr_scheduler(optimizer,epoch) 
    for i in range(total_batch):
        batch_x, batch_y = next_batch(batch_size,i)
        # Convert torch tensor to Variable
        images = Variable(torch.FloatTensor(batch_x).view(-1,1,60,60))
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs,hidden = net(images)
        labels=Variable(torch.LongTensor(batch_y.reshape(batch_y.shape[0],)))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        predicted=predicted.numpy()
        labels=labels.data.numpy()
        correct= (predicted == labels).sum()

            
    print ('*******************************Epoch [%d/%d], Loss: %.4f' 
              %(epoch+1, num_epochs, loss.data[0]))
    
    test_train_model()
    outputs,hidden,labels,acc=test_model() 
    torch.save(net.state_dict(), 'model_cnn.pkl')
    #test_model2() 

    if (epoch)%5==0:
       output,error=compute_error(outputs,labels)  
       with open('result'+str(epoch+1)+'_'+str(acc)+'.json','w') as fp:    
            json.dump({'hidden':hidden,'error':error,'label':labels.tolist(),'input':test_names,'output':output,
                   'class_names':['female','male'],'description':'Gender classification on LFW dataset cnn\
                                    architecture\
                                      [cnn-relu-pool->cnn-relu-pool->cnn-relu-pool->120->2]'},fp)    

        
