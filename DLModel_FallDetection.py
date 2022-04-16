#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import History 
import time
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# ### Reading Data and Splitting X and Y

# In[114]:


df = pd.read_csv('fall_dataset.csv', sep=',',header=[0])
df.apply(pd.to_numeric)
df.head()


# In[115]:


df.shape


# In[116]:


df_x = df.iloc[:,:6]
df_x.head(3)


# In[117]:


df_y = df.iloc[:,6:]
df_y.head(3)


# ### Class Balance

# In[118]:


df_y['Label'].value_counts()/len(df_y)


# In[119]:


fig = sns.countplot(x='Label',data=df_y)
fig.set_xlabel("0: Walking 1:Fall", fontsize = 15)


# ### Pairwise Relationship in the Dataset

# In[122]:


g = sns.pairplot(df,hue= 'Label') 
g.fig.set_size_inches(10,10)


# ### Splitting the data for train, test and validation data

# In[135]:


X = df_x
Y = df_y
X_t, X_test, Y_t,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 123)
X_train, X_validation, Y_train,Y_validation = train_test_split(X_t,Y_t,test_size = 0.25, random_state = 123)


# In[136]:


print('Training Dataset :', "{0:,}".format(X_train.shape[0])  )
print('Validation Dataset :', "{0:,}".format(X_validation.shape[0]) )
print('Test Dataset :', "{0:,}".format(X_test.shape[0]) ) 


# ### Creating Deep Learning Model
# 
# - Only one layer simple sequential model

# In[137]:


class_weight = {0: 1,
                1: 1
                }


# In[138]:


# get the model
def get_model(n_inputs, n_outputs): 
    model = Sequential()
    model.add(Dense(64, input_dim=6, activation='relu')) 
    model.add(Dense(1, activation='sigmoid'))  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model_1 = get_model(X_train.shape[1], Y_train.shape[1])
history = model_1.fit(X, Y,epochs=500,batch_size=128,class_weight=class_weight,validation_data=(X_train, Y_train))

yhat = model_1.predict(X_test)
yhat = yhat.round()
acc = accuracy_score(Y_test.to_numpy(), yhat)
print(acc)


# In[139]:


print(model_1.summary())


# ### Accuracy and Loss

# In[140]:


import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc)) 

plt.plot(epochs, acc, label = 'train acc' )
plt.plot(epochs, val_acc, label = 'val_acc' )
plt.title ('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, label = 'Training Loss' )
plt.plot(epochs, val_loss, label = 'Validation Loss' )
plt.title ('Training and validation loss'   )
plt.legend()


# In[141]:


training_pred = model_1.predict(X_train)
validation_pred = model_1.predict(X_validation)
test_pred = model_1.predict(X_test)

tra_acc = accuracy_score(Y_train.to_numpy(), training_pred.round())
val_acc = accuracy_score(Y_validation.to_numpy(), validation_pred.round())
test_acc = accuracy_score(Y_test.to_numpy(), test_pred.round())

print('Training Acc :', "{0:,}".format(tra_acc)  )
print('Validation Acc :', "{0:,}".format(val_acc) )
print('Test Acc :', "{0:,}".format(test_acc) ) 


# ### Confusion Matrix

# In[142]:


print(confusion_matrix(Y_test, test_pred.round()))
print(classification_report(Y_test, test_pred.round()))


# ### Model Saving

# In[143]:


model_1.save("fall_detection_model.h5")


# ### Predictions From Model

# In[144]:


saved_model = keras.models.load_model("fall_detection_model.h5")
data = np.array(X_test.iloc[100]).reshape(1,6)
print(data)
label = saved_model.predict(data).round()
print(int(label))


# ### Saving Model TF-Lite Format to use it in Android

# In[145]:


import tensorflow as tf

pre_model = tf.keras.models.load_model("fall_detection_model.h5")
pre_model.save("TFversion")


# In[146]:


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("TFversion") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model_android.tflite', 'wb') as f:
  f.write(tflite_model)

