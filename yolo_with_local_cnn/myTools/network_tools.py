import os
import cv2
import numpy as np

import yolo_with_local_cnn.myTools.image_tools as img_tools

import socket
import sys
from absl import app, flags, logging

np.set_printoptions(threshold=sys.maxsize)


def write_log(tag, img_data, imgs):
    imgs, validate = tampon(imgs, img_data, imgs)
    if (not validate):
        file = open("./log/log.txt", "a")
        file.write("IMG" + "," + tag + ",")
        file.write(str(len(img_data)))
        file.write("\n")
        file.close()


def tampon(imgs, img_r, threshold):
    validate = False
    for img in imgs:
        tmp_res = cv2.absdiff(img, img_r)
        res = tmp_res.astype(np.uint8)
        percentage = (np.count_nonzero(res) * 100) / res.size

        if (percentage > threshold):
            validate = True
    if (validate == False):
        imgs.append(img_r)
    return imgs, validate


def init_connexion(host="127.0.0.1", port=12346):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # connect to server on local computer
    s.connect((host, port))

    # message you send to server
    message = "0;YOLO;CAT;DATA;"

    s.send(message.encode('utf-8'))

    data = s.recv(3000000)

    while data.decode('utf-8') != "ok":
        data = s.recv(3000000)

    logging.info("Connexion ok")

    return s


def send_data(cat, data, s):
    data = img_tools.load_img_opencv(data)
    print("Type : ", type(data), "Dimension : ", np.shape(data))
    message = "1;" + "YOLO" + ";" + cat.upper() + ";" + np.array2string(data) + ";"
    logging.info(
        "Send image category : " + cat + " message size = " + str(sys.getsizeof(message)))

    s.send(message.encode('utf-8'))
