from flask import Flask, request, render_template
#import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models, losses, Model, utils
import glob

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    image_id = int(request.form["img_id"])-1
    
    
    
    test_data_location="/Users/stefanbrasoveanu/Desktop/demo_app/static/images"
    test_img_locations=glob.glob(test_data_location + '/**/*.jpg')
    print(test_img_locations[5])
    print(len(test_img_locations))
    img_height=167
    img_width=167
    class_names=["glioma", "meningioma", "notumor", "pituitary"]
    #inference_img_path="/Users/stefanbrasoveanu/Desktop/demo_app/static/images/glioma/image.jpg"
    inference_img_path=test_img_locations[image_id]
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
    
    
    output_text="This image most likely belongs to the class {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    #prediction = model.predict([[rooms, distance]])  # this returns a list e.g. [127.20488798], so pick first element [0]
    output = image_id

    #return render_template('pred.html', prediction_text=f'A house with {rooms} rooms and located {distance} meters from the city center has a value of ${output}', img_location=f'/static/images/glioma/image(7).jpg')
    #img_location=f'/static/images/glioma/image(7).jpg'
    #prediction_text=f'Image number ${output}'
    return render_template('pred.html', prediction_text=output_text, img_location=inference_img_path[40:], actual_class="The real class is "+inference_img_path[40:].split("/")[3]+".")


if __name__ == "__main__":
    app.run()
