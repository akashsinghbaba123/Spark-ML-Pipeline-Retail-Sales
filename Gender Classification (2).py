#!/usr/bin/env python
# coding: utf-8

# In[22]:


#Importing the important module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import InputLayer,Dense
from glob import glob
from keras.optimizers import Adam


# In[3]:


#Mouting google drive
from google.colab import drive
drive.mount('/content/drive')


# In[94]:


#Unziping the zip file which contain train images
get_ipython().system('unzip /content/drive/MyDrive/train_nLPp5K8.zip')


# In[4]:


#reading the csv file (Train and test)
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test_fkwGUNG.csv')


# In[5]:


# create random number generator
seed=42
rseed=np.random.RandomState(seed)


# In[6]:


# top five train data
train_data.head()


# In[7]:


#Top five test data
test_data.head()


# In[8]:


#counts of different classes
train_data['class'].value_counts()


# In[9]:


#module for reading ,saving,showing and resizing the image
from skimage.io import imread,imsave,imshow
from skimage.transform import resize


# In[10]:


#reading the image in gray scale and storing in numpy array
X=[]
for image in train_data.image_names:
  img=imread('images/'+image,as_gray=True)
  X.append(img)
X=np.array(X)
y=train_data['class']


# In[11]:


#reading the image in gray scale and storing in numpy array
X_test=[]
for image in test_data.image_names:
  img=imread('images/'+image,as_gray=True)
  X_test.append(img)
X_test=np.array(X_test)


# In[12]:


# resizing the image of test
final_re_image_test=[]
for i in range(len(X_test)):
    temp=resize(X_test[i],(100,100))
    final_re_image_test.append(temp)
final_re_image_test=np.array(final_re_image_test)
X_test=final_re_image_test


# In[13]:


#resizing the image of train to 100 ,100 because of google colab to avoid ram outof memory
final_re_image=[]
for i in range(len(X)):
    temp=resize(X[i],(100,100))
    final_re_image.append(temp)
final_re_image=np.array(final_re_image)
X=final_re_image


# In[15]:


fig,ax=plt.subplots(nrows=1,ncols=5,figsize=(20,20))
for i in range(5):
  ax[i].imshow(X[i*400])
  ax[i].axis('off')


# In[16]:


# libraries for performing image augmentation tasks
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.transform import AffineTransform, warp


# In[17]:


# augmenting the training images
final_train_data = []
final_target_train = []
for i in range(10000,train_data.shape[0]):
    # original image
    final_train_data.append(X[i])
    # image rotation
    final_train_data.append(rotate(X[i], angle=30, mode = 'edge'))
    # image flipping (left-to-right)
    final_train_data.append(np.fliplr(X[i]))
    # image flipping (up-down)
    final_train_data.append(np.flipud(X[i]))
    # image noising
    final_train_data.append(random_noise(X[i],var=0.2))
    for j in range(5):
        final_target_train.append(y[i])


# In[18]:


# converting images and target to array
X = np.array(final_train_data)
y= np.array(final_target_train)


# In[19]:


#shape of train images and test images
X.shape,X_test.shape


# In[20]:


#converting 2 dimensional image to 1 dimensional image
X=X.reshape(X.shape[0],100*100)
X_test=X_test.reshape(X_test.shape[0],100*100)


# In[21]:


#Normalizing the pixles values
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X=ss.fit_transform(X)
X_test=ss.fit_transform(X_test)


# **prevent overfitting**

# In[28]:


# importing the dropout layer
from keras.layers import Dropout
# importing batch normalization layer
from keras.layers import BatchNormalization
#importing the optimizer
from keras.optimizers import Adam
# importing different initialization techniques
from keras.initializers import random_normal, glorot_normal, he_normal
# defining the adam optimizer and setting the learning rate as 10^-5
adam = Adam(lr=1e-5, clipvalue=1)
# importing module for early stopping
from keras.callbacks import EarlyStopping


# In[29]:


# defining the architecture of the model
model=Sequential()
model.add(InputLayer(input_shape=X.shape[1],))
model.add(Dense(100,activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(50,activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(1,activation='sigmoid'))


# In[30]:


# summary of the model
model.summary()


# In[31]:


# compiling the model
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])


# In[32]:


# training the model
model_history=model.fit(X,y,epochs=50,batch_size=200)


# In[33]:


# getting predictions in the form of probabilities
#prediction for test data
prediction_probabilities_test = model.predict(X_test)[:,0]


# In[35]:


#test image probalilites
prediction_probabilities_test


# In[36]:


# converting probabilities to classes
prediction_probabilities_test= prediction_probabilities_test >= 0.5
prediction_probabilities_test = prediction_probabilities_test.astype(np.int)


# In[37]:


prediction_probabilities_test


# In[38]:


#Importing metrics
from sklearn.metrics import accuracy_score


# In[39]:


#predicting for train data
prediction_probabilities_train = model.predict(X)[:,0]


# In[40]:


# converting probabilities to classes
prediction_probabilities_train= prediction_probabilities_train >= 0.5
prediction_probabilities_train = prediction_probabilities_train.astype(np.int)


# In[41]:


prediction_probabilities_train


# In[42]:


# accuracy on training set
accuracy_score(y,prediction_probabilities_train)


# In[44]:


# pulling out the original images from the data which correspond to the validation data
import random as rng
# get a random index to plot image randomly
index = rng.choice(range(len(test_data)))

# get the corresponding image name and probability
img_name = test_data['image_names'][index]
prob = (prediction_probabilities_test * 100).astype(int)[index]

# read the image
img = plt.imread('images/' + img_name)

# print probability and actual class
print('Model is', prob , '% sure that it is male')
print(prediction_probabilities_test[index])
# plot image
plt.imshow(img)


# Adding weight initializers

# In[45]:


# defining the architecture of the model
model=Sequential()
model.add(InputLayer(input_shape=X.shape[1],))
model.add(Dense(100,activation='sigmoid',kernel_initializer=he_normal(seed=seed)))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(50,activation='sigmoid',kernel_initializer=he_normal(seed=seed)))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(1,activation='sigmoid',kernel_initializer=he_normal(seed=seed)))


# In[46]:


# summary of the model
model.summary()


# In[47]:


# compiling the model
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])


# In[48]:


# training the model
model_history=model.fit(X,y,epochs=50,batch_size=200)


# In[51]:


# getting predictions in the form of probabilities
#prediction for test data
prediction_probabilities_test = model.predict(X_test)[:,0]


# In[95]:


# converting probabilities to classes
prediction_probabilities_test= prediction_probabilities_test >= 0.5
prediction_probabilities_test = prediction_probabilities_test.astype(np.int)


# In[96]:


#predicting for train data
prediction_probabilities_train = model.predict(X)[:,0]
# converting probabilities to classes
prediction_probabilities_train= prediction_probabilities_train >= 0.5
prediction_probabilities_train = prediction_probabilities_train.astype(np.int)


# In[63]:


# accuracy on training set
a=accuracy_score(y,prediction_probabilities_train)
print('accuracy_score',a)


# In[98]:


# pulling out the original images from the data which correspond to the validation data
import random as rng
# get a random index to plot image randomly
index = rng.choice(range(len(test_data)))

# get the corresponding image name and probability
img_name = test_data['image_names'][index]
prob = (prediction_probabilities_test * 100).astype(int)[index]

# read the image
img = plt.imread('images/' + img_name)

# print probability and actual class
print('Model is', prob , '% sure that it is male')
print(prediction_probabilities_test[index])
# plot image
plt.imshow(img)


# Early stopping

# In[64]:


# increasing the patience and threshold value
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, mode='min')


# In[67]:


# defining the architecture of the model
model=Sequential()
model.add(InputLayer(input_shape=X.shape[1],))
model.add(Dense(100,activation='sigmoid',kernel_initializer=he_normal(seed=seed)))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(50,activation='sigmoid',kernel_initializer=he_normal(seed=seed)))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(1,activation='sigmoid',kernel_initializer=he_normal(seed=seed)))


# In[68]:


# compiling the model
model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])


# In[69]:


# training the model
model_history=model.fit(X,y,epochs=100,batch_size=200,callbacks=[early_stopping])


# In[70]:


# getting predictions in the form of probabilities
#prediction for test data
prediction_probabilities_test = model.predict(X_test)[:,0]


# In[99]:


# converting probabilities to classes
prediction_probabilities_test= prediction_probabilities_test >= 0.5
prediction_probabilities_test = prediction_probabilities_test.astype(np.int)


# In[100]:


#predicting for train data
prediction_probabilities_train = model.predict(X)[:,0]
# converting probabilities to classes
prediction_probabilities_train= prediction_probabilities_train >= 0.5
prediction_probabilities_train = prediction_probabilities_train.astype(np.int)


# In[73]:


# accuracy on training set
a=accuracy_score(y,prediction_probabilities_train)
print('accuracy_score',a)


# In[75]:


# pulling out the original images from the data which correspond to the validation data
import random as rng
# get a random index to plot image randomly
index = rng.choice(range(len(test_data)))

# get the corresponding image name and probability
img_name = test_data['image_names'][index]
prob = (prediction_probabilities_test * 100).astype(int)[index]

# read the image
img = plt.imread('images/' + img_name)

# print probability and actual class
print('Model is', prob , '% sure that it is male')
print(prediction_probabilities_test[index])
# plot image
plt.imshow(img)


# In[76]:


# using relu as activation function in hidden layer
model=Sequential()
model.add(InputLayer(input_shape=X.shape[1],))
model.add(Dense(100,activation='relu',kernel_initializer=he_normal(seed=seed)))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(50,activation='relu',kernel_initializer=he_normal(seed=seed)))
model.add(BatchNormalization())
model.add(Dropout(rate=0.5))
model.add(Dense(1,activation='sigmoid',kernel_initializer=he_normal(seed=seed)))

# compiling the model
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
# summary of the model
model.summary()
#training the model
model_history=model.fit(X,y,batch_size=200,epochs=100,callbacks=[early_stopping])


# In[101]:


# getting predictions in the form of class as well as probabilities

prediction_train = model.predict(X)[:, 0]
prediction_train = prediction_train.reshape(-1,)

# converting probabilities to classes
prediction_train = prediction_train >= 0.5
prediction_train = prediction_train.astype(np.int)


# In[79]:


# accuracy on training set
print('Accuracy on training set:', accuracy_score(y,prediction_train), '%')


# In[ ]:





# **Well we will be able to increase the accuracy score upto 93 while adding dropout, weight initializers, early stopping etc **

# In[88]:


# getting predictions in the form of probabilities
#prediction for test data
prediction_probabilities_test = model.predict(X_test)[:,0]
# converting probabilities to classes
prediction_probabilities_test= prediction_probabilities_test >= 0.5
prediction_probabilities_test = prediction_probabilities_test.astype(np.int)
# pulling out the original images from the data which correspond to the validation data
import random as rng
# get a random index to plot image randomly
index = rng.choice(range(len(test_data)))

# get the corresponding image name and probability
img_name = test_data['image_names'][index]
prob = (prediction_probabilities_test * 100).astype(int)[index]

# read the image
img = plt.imread('images/' + img_name)

# print probability and actual class
print('Model is', prob , '% sure that it is male')
print(prediction_probabilities_test[index])
# plot image
plt.imshow(img)


# In[89]:


prediction_probabilities_test


# In[91]:


#solution data frame
solution=pd.DataFrame({'image_names':test_data.image_names,'class':prediction_probabilities_test})


# In[102]:


solution.to_csv('solution.csv')


# In[103]:


#top five rows of solution
solution.head()


# In[ ]:




