import numpy as np
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D,\
BatchNormalization, Flatten, Conv2D, AveragePooling2D,\
MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from glob import glob
import keras.backend as K

def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base = 'bn'+str(stage)+block+'_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid',
               name=conv_name_base+'2a', kernel_initializer= glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    # Second component
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', 
               name=conv_name_base+'2b', kernel_initializer= glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    # Third component
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', 
               name=conv_name_base+'2c', kernel_initializer= glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)
    
    # Add shortcut value
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component
    X = Conv2D(F1, kernel_size=(1,1), strides=(s,s), name=conv_name_base+'2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    # Second component
    X = Conv2D(F2, kernel_size=(f,f), strides=(1,1), padding='same', name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third component
    X = Conv2D(F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base+'2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)

    # Shortcut path
    X_shortcut = Conv2D(F3, kernel_size=(1,1), strides=(s,s), padding='valid', name=conv_name_base+'1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)
    print(X_shortcut.shape)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape=(64, 64, 3), classes=6):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7,7), strides=(2,2), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)
    
    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    
    
    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    
    # stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    
    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL 
    X = AveragePooling2D((2,2), name="avg_pool")(X)


    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


model = ResNet50(classes=6)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# MAking Dataset

IMAGE_SIZE = [64,64]
epochs=10
batch_size=32

train_path = 'small_dataset/Training'
valid_path = 'small_dataset/Validation'

image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

folders = glob(train_path+'/*')

gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input
)


test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k,v in test_gen.class_indices.items():
    labels[v] = k


train_generator = gen.flow_from_directory(
        train_path,
        target_size=IMAGE_SIZE,
        shuffle=True,
        batch_size=batch_size
)

valid_generator = gen.flow_from_directory(
        valid_path,
        target_size=IMAGE_SIZE,
        shuffle=True,
        batch_size=batch_size
)

r = model.fit_generator(
        train_generator,
        validation_data=valid_generator,
        epochs=epochs,
        steps_per_epoch=len(image_files)//batch_size,
        validation_steps=len(valid_image_files)//batch_size
)
plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))