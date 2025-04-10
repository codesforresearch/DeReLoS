import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Layer, BatchNormalization, Conv2D, Dense, Flatten,Softmax, Add, MaxPooling2D, AveragePooling2D, Dropout, Input
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import time
from tensorflow import keras
from tensorflow.keras import layers



test_images=np.load('./derelos_test_images.npy')
test_labels=np.load('./derelos_test_labels.npy')


model_baseline=load_model('./baseline_chestxray.h5')
model_random_shuffle=load_model('./random_shuffle_chestxray.h5')
model_bd=load_model('./binary_detector.h5')



!pip install cleverhans
import numpy as np
import tensorflow as tf
from absl import app, flags
from easydict import EasyDict
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent as pgd
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method as bim
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method as fgm
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2 as cw

FLAGS = flags.FLAGS
import cv2
import matplotlib.pyplot as plt
import random



import numpy as np

def shuffle_tiles(input_image, tiles_number):
    """Shuffle tiles of an image efficiently."""

    tile_size = 224 // tiles_number
    new_image = np.zeros_like(input_image)

    # Generate tile start indices
    indices = np.arange(0, 224, tile_size)
    grid_positions = np.array([(i, j) for i in indices for j in indices])

    # Shuffle positions
    shuffled_positions = grid_positions.copy()
    np.random.shuffle(shuffled_positions)

    # Rearrange tiles

    for (src, dst) in zip(grid_positions, shuffled_positions):
        i_src, j_src = src
        i_dst, j_dst = dst

        new_image[i_dst:i_dst+tile_size, j_dst:j_dst+tile_size] = input_image[i_src:i_src+tile_size, j_src:j_src+tile_size]

    return new_image



total_number_of_samples=100
eps=8/255
tiles_number=112
count=0
for sample in range(total_number_of_samples):
    image = np.float32(test_images[sample])
    label = test_labels[sample]
    label=tf.reshape(label,[1,2])
    image = tf.reshape(image, [1, 224, 224, 3])

    # Generate adversarial examples
    adv_x = fgm(model_random_shuffle, image, eps, np.inf)
    adv_x = tf.clip_by_value(adv_x, 0, 1)
    adv_x=adv_x[0]

    # Shuffle tiles
    transformed_image = shuffle_tiles(adv_x, tiles_number)

    # Evaluate models
        
    if np.argmax(model_bd(tf.reshape(adv_x, [1, 224, 224, 3]))) == 0:
        output = model_baseline(tf.reshape(adv_x, [1, 224, 224, 3]))

    else:
        output = model_random_shuffle(tf.reshape(transformed_image, [1, 224, 224, 3]))
        
    if np.argmax(output) == np.argmax(label):
        count=count+1
classification_accuracy=count/total_number_of_samples
print(count*100)