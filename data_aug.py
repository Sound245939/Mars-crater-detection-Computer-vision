import keras.preprocessing.image
from keras_preprocessing.image import ImageDataGenerator
from skimage import io


data_gen = ImageDataGenerator (
    rotation_range =45 ,  #ratate between 0-45 degrees
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',cval=125 #fills nearest remaining pixels with 125 with grey pixels

)

x = io.imread('mars_crater.jpg')

x=x.reshape((1, )+ x.shape)

i = 0
for batch in data_gen.flow(x, batch_size=16, #creates 16 images at once
                           save_to_dir='imageAugm',
                           save_prefix='aug',
                           save_format='png'):
    i += 1
    if i > 20:
        break