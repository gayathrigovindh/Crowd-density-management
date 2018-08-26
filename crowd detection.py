
# coding: utf-8

# In[17]:


import cv2
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image 
from IPython.display import display
from IPython.display import Image as _Imgdis
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, MaxPooling2D, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[6]:


cap = cv2.VideoCapture('Desktop/video_1.mp4')
succes, image = cap.read()
try:
    if not os.path.exists('Desktop/data/'):
        os.makedirs('Desktop/data/')
except OSError:
    print("Error : Creating directory")


# In[18]:


currentFrame = 0
success=True
while success:
    cv2.imwrite('Desktop/data/'+"Frame%d.jpg"%currentFrame, image)
    success , image = cap.read()
    print('creating....'+"Frame%d.jpg"%currentFrame)
    currentFrame+=1


# In[19]:


cap.release()
cv2.destroyAllWindows()


# In[36]:


folder = "Desktop/data/"
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f))]
for i in range(2,10):
    print(files[i])
    display(_Imgdis(filename = folder + "/" + files[i],width=240, height=320))

#plt.imshow('Desktop/data/Frame0.jpg')
#plt.show()


# In[8]:


classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))


# In[9]:


classifier.add(MaxPooling2D(pool_size=(2,2)))


# In[10]:


classifier.add(Flatten())


# In[11]:


classifier.add(Dense(units = 128, activation = 'relu'))


# In[12]:


classifier.add(Dense(units=1, activation='sigmoid'))


# In[27]:


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[28]:


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2,zoom_range=0.2,horizontal_flip=0.2)


# In[29]:


test_datagen = ImageDataGenerator(rescale=1./255)


# In[38]:


training_set = train_datagen.flow_from_directory('Desktop/data/Train/', target_size = (64,64),batch_size = 50, class_mode='binary')


# In[42]:


test_set=test_datagen.flow_from_directory('Desktop/data/Test/',target_size=(64,64),batch_size=50, class_mode='binary')


# In[43]:


classifier.fit_generator(training_set, steps_per_epoch =767, epochs = 2, validation_data = test_set, validation_steps = 489)


# In[73]:


test_image=image.load_img('Desktop/data/Test/No_crowd/Frame287.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]==0:
    prediction='Crowd'
    print(prediction)
else:
    prediction='No crowd'
    print(prediction)


# In[56]:


print(training_set.class_indices)

