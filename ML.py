import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Reshape
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import concatenate
import cv2

def create_ssd_model(input_shape=(300, 300, 3), num_classes=21):

    base_model = MobileNetV2(input_shape=input_shape, include_top=False)

    feature_extractor = base_model.get_layer('out_relu').output

    conv4_3 = Conv2D(512, (3,3), padding='same', activation='relu', name='conv4_3')(feature_extractor)
    conv7 = Conv2D(256, (3,3), padding='same', activation='relu', name='conv7')(conv4_3)
    conv8_2 = Conv2D(256, (3,3), padding='same', activation='relu', name='conv8_2')(conv7)
    conv9_2 = Conv2D(128, (3,3), padding='same', activation='relu', name='conv9_2')(conv8_2)
    conv10_2 = Conv2D(128, (3,3), padding='same', activation='relu', name='conv10_2')(conv9_2)

    
    num_predictions = 4 * (num_classes + 4) 
    conv4_3_norm = Reshape((-1, 4))(Conv2D(num_predictions, (3,3), padding='same', name='conv4_3_norm')(conv4_3))
    fc7 = Reshape((-1, 4))(Conv2D(num_predictions, (3,3), padding='same', name='fc7')(conv7))
    conv8_2_norm = Reshape((-1, 4))(Conv2D(num_predictions, (3,3), padding='same', name='conv8_2_norm')(conv8_2))
    conv9_2_norm = Reshape((-1, 4))(Conv2D(num_predictions, (3,3), padding='same', name='conv9_2_norm')(conv9_2))
    conv10_2_norm = Reshape((-1, 4))(Conv2D(num_predictions, (3,3), padding='same', name='conv10_2_norm')(conv10_2))

 
    predictions = concatenate([conv4_3_norm, fc7, conv8_2_norm, conv9_2_norm, conv10_2_norm], axis=1, name='predictions')


    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def processing(image):
    model = create_ssd_model()
    model.summary()

    import cv2
    import numpy as np

    image = cv2.imread('Untitled.jpeg')

    input_image = cv2.resize(image, (300, 300))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.astype(np.float32) / 255.0
    predictions = model.predict(np.expand_dims(input_image,axis=0))
    bounding_boxes = predictions[..., :4].tolist()
    class_labels = predictions[..., 0].tolist()
    confidence_scores = predictions[..., 1].tolist()


    
    return bounding_boxes,class_labels,confidence_scores


