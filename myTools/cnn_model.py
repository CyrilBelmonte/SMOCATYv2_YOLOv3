import numpy as np
import tensorflow as tf
import pandas as pd

import multiprocessing

import myTools.image_tools as img_t

info = []


def get_inception_v2_cat():
    model = tf.keras.models.load_model("../resources/inceptionV2_cat_breed_finetuned_v1.h5")
    return model


def get_inception_v2_dog():
    model = tf.keras.models.load_model("../resources/inceptionV2_dog_breed120_finetuned_v2.h5")
    return model


def get_more_cat(model, img):
    prediction = model.predict(img)
    arg = np.argmax(prediction)
    labels = pd.read_csv("../output/cats.csv")
    list_label = list(set(labels['breed']))
    list_label.sort()
    return list_label[arg], prediction[0][arg]


def get_more_dog(model, img):
    prediction = model.predict(img)
    arg = np.argmax(prediction)
    labels = pd.read_csv("..\\resources\\labels_dog_breeds.csv")
    list_label = list(set(labels['breed']))
    list_label.sort()
    return list_label[arg], prediction[0][arg]


def thread_for_cnn(i, classe, model, img):
    img_to_cnn = img_t.load_img_opencv(img)
    label = ""
    prob = 0
    if classe == "dog":
        label, prob = get_more_dog(model, img_to_cnn)
    else:
        label, prob = get_more_cat(model, img_to_cnn)
    info.append([i, label, prob])


def get_more_data(img, model_cat, model_dog, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    process_s = []
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

        if class_names[int(classes[i])] == "cat" or class_names[int(classes[i])] == "dog":
            x1, y1 = x1y1
            x2, y2 = x2y2
            img_crop = img[y1:y1 + (y2 - y1), x1:x1 + (x2 - x1)]

            if class_names[int(classes[i])] == "cat":
                process = multiprocessing.Process(target=thread_for_cnn, args=[i, "cat", model_dog, img_crop])
                process.start()
                process_s.append(process)
            else:
                process = multiprocessing.Process(target=thread_for_cnn, args=[i, "dog", model_cat, img_crop])
                process.start()
                process_s.append(process)
    for process in process_s:
        process.join()
    return info
