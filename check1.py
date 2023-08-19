import numpy as np
from keras.preprocessing import image
import tensorflow as tf

# Load the TFLite model
model_path = 'model_unquant.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to preprocess the input image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image pixel values
    return img

# Define a function to predict the class of the input image
def predict_image_class(image_path, interpreter):
    img = preprocess_image(image_path)
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img)
    # Run inference
    interpreter.invoke()
    # Get the output tensor
    result = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(result)  # Get the index of the class with the highest probability
    # Assuming you have a label_map that maps class indices to their corresponding class names
    label_map = {0: 'Tomato___Bacterial_spot', 1: 'Tomato___Early_blight', 2: 'Tomato___healthy', 3: 'Tomato___Late_blight', 4: 'Tomato___Leaf_Mold', 5: 'Tomato___Tomato_mosaic_virus'}
    predicted_class = label_map[class_index]
    return predicted_class

# Example usage:
image_path = '0de16216-510d-48c1-9ef0-78ce39328ff2___GH_HL Leaf 240.jpg'
predicted_class = predict_image_class(image_path, interpreter)
print("Predicted class:", predicted_class)
