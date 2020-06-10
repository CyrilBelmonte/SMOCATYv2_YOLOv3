import numpy as np
import tensorflow as tf
import pandas as pd


def init_model(file_model):
    return


def get_more_cat(img):
    model = tf.keras.models.load_model("../resources/inceptionV2_cat_breed_finetuned_v1.h5")
    prediction = model.predict(img)
    arg = np.argmax(prediction)
    labels = pd.read_csv("..\\resources\\cats.csv")
    list_label = list(set(labels['breed']))
    list_label.sort()
    return list_label[arg], prediction[0][arg]


def get_more_dog(img):
    model = tf.keras.models.load_model("../resources/inceptionV2_dog_breed120_finetuned_v2.h5")
    prediction = model.predict(img)
    arg = np.argmax(prediction)
    labels = pd.read_csv("..\\resources\\labels_dog_breeds.csv")
    list_label = list(set(labels['breed']))
    list_label.sort()
    return list_label[arg], prediction[0][arg]
