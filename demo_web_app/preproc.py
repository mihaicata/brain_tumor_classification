#from flask import Flask, request, render_template
#import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, losses, Model, utils
import glob


test_data_location="/Users/stefanbrasoveanu/Desktop/demo_app/static/images"
test_img_locations=glob.glob(test_data_location + '/**/*.jpg')
print(test_img_locations[5])
print(len(test_img_locations))
img_height=167
img_width=167
class_names=["glioma", "meningioma", "notumor", "pituitary"]
inference_img_path="/Users/stefanbrasoveanu/Desktop/demo_app/static/images/glioma/image.jpg"
img = tf.keras.preprocessing.image.load_img(
    inference_img_path, target_size=(img_height, img_width)
)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch
simple_cnn_model= tf.keras.models.load_model('models/inference_model.h5')
predictions = simple_cnn_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
