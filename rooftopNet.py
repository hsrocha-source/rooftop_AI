Xtrain=[]
ytrain=[]
Xval=[]
yval=[]
Xtest=[]
Xtrain1=[]
ytrain1=[]
Xval1=[]
yval1=[]
Xtest1=[]
h=256
w=256
channels=3
batch_size=32
epochs=4000
i=0
import os
from PIL import Image, ImageFile
import numpy as np
import tensorflow as tf
#The images in AIRS include truncated training example.
ImageFile.LOAD_TRUNCATED_IMAGES = True
#The TIFF images are very large
Image.MAX_IMAGE_PIXELS = None
#Images are resized with bilinear filters, simple and fast
for image in os.listdir('scrapings/image'):
    img_raw=Image.open('scrapings/image' + '/' + image)
    img_filtered=img_raw.resize((h, w), Image.BILINEAR)
    Xtest.append(img_filtered)
    print(i)
    i=i+1
for image in os.listdir('scrapings/trainval/train/image'):
    img_raw=Image.open('scrapings/trainval/train/image' + '/' + image)
    img_filtered=img_raw.resize((h, w), Image.BILINEAR)
    Xtrain.append(img_filtered)
    print(i)
    i=i+1
for image in os.listdir('scrapings/trainval/train/label'):
    img_raw=Image.open('scrapings/trainval/train/label' + '/' + image)
    if "vis" in str(image):
        img_filtered=img_raw.resize((h, w), Image.BILINEAR)
        ytrain.append(img_filtered)
    print(i)
    i=i+1
for image in os.listdir('scrapings/trainval/val/image'):
    img_raw=Image.open('scrapings/trainval/val/image' + '/' + image)
    img_filtered=img_raw.resize((h, w), Image.BILINEAR)
    Xval.append(img_filtered)
    print(i)
    i=i+1
for image in os.listdir('scrapings/trainval/val/label'):
    img_raw=Image.open('scrapings/trainval/val/label' + '/' + image)
    if "vis" in str(image):
        img_filtered=img_raw.resize((h, w), Image.BILINEAR)
        yval.append(img_filtered)
    print(i)
    i=i+1
i=0
#The following code organizes the input data so that it is readable by Tensorflow 2
for a in Xtrain:
    a=np.array(a)
    Xtrain1.append(a)
    i+=1
i=0
for a in Xtest:
    a=np.array(a)
    Xtest1.append(a)
    i+=1
i=0
for a in Xval:
    a=np.array(a)
    Xval1.append(a)
    i+=1
i=0
for a in ytrain:
    a=np.array(a)
    ytrain1.append(a)
    i+=1
i=0
for a in yval:
    a=np.array(a)
    yval1.append(a)
    i+=1
print(i)

Xtrain=tf.dtypes.cast(tf.convert_to_tensor(Xtrain1), dtype=tf.float32)/255.
yval=tf.dtypes.cast(tf.convert_to_tensor(yval1), dtype=tf.float32)/255.
Xval=tf.dtypes.cast(tf.convert_to_tensor(Xval1), dtype=tf.float32)/255.
ytrain=tf.dtypes.cast(tf.convert_to_tensor(ytrain1), dtype=tf.float32)/255.
Xtest=tf.dtypes.cast(tf.convert_to_tensor(Xtest1), dtype=tf.float32)/255.
inputs = tf.keras.layers.Input((h, w, channels))
#The following code is an implementation of a UNet as described in https://arxiv.org/abs/1505.04597
s=inputs
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
 
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c5)
 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                            padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='linear')(c9)
#Model Optimization
adam=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
model.summary()
#The model is fit with several keras callbacks, including Early Stopping for better iteration and experimentation
model.fit(x=Xtrain, y=ytrain, validation_data=(Xval,yval), shuffle=True,
          batch_size=batch_size, epochs=epochs,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.005, patience=4, verbose=1,
                                                      mode='max', baseline=None, restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=3, verbose=2,
                                                          mode='auto', min_delta=0.014, cooldown=0, min_lr=1e-4),
                     tf.keras.callbacks.TensorBoard(log_dir=r'C:\Python Projects\ais\AI_logs',
                                                    histogram_freq=3, write_graph=True, write_images=True,
                                                    embeddings_freq=1, embeddings_layer_names=None,
                                                    embeddings_metadata=None, embeddings_data=None, update_freq='batch')
                    ]
         )
#Save trained model
model.save('rooftop_neural_net.h5')
