import json
from tkinter import Image
from urllib import response
import boto3
from matplotlib import pyplot as plt
import cv2

import os
os.system('cls')


client = boto3.client(
    'rekognition',
    aws_access_key_id='aws_access_key_id',
    aws_secret_access_key='aws_secret_access_key',
    region_name='ap-southeast-1'
)

photo = "img/label.jpg"
img = cv2.imread(photo)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
with open(photo, 'rb') as source_image:
    source_bytes = source_image.read()

response = client.detect_labels(
    Image={'Bytes': source_bytes},
    MaxLabels=1,
    MinConfidence=99
)
print(response)

plt.subplot(1, 2, 1), plt.imshow(img)
plt.title('Input'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(img)
plt.title('Output'), plt.xticks([]), plt.yticks([])
plt.show()


def detect_faces(client):

    photo = "img/label.jpg"

    img = cv2.imread(photo)
    height_shape = img.shape[0]
    width_shape = img.shape[1]

    with open(photo, 'rb') as source_image:
        source_bytes = source_image.read()

    response = client.detect_faces(
        Image={'Bytes': source_bytes},
        Attributes=['ALL']
        # MinConfidence = 99
    )
    print(response)

    i = 1
    for person in response['FaceDetails']:
        print(i)
        print("Width: " + str(person['BoundingBox']['Width']))
        print("Height: " + str(person['BoundingBox']['Height']))
        print("Left: " + str(person['BoundingBox']['Left']))
        print("Top: " + str(person['BoundingBox']['Top']))
        print("Age Low: " + str(person['BoundingBox']['Low']))
        print("Age High: " + str(person['BoundingBox']['High']))

        left = int(person['BoundingBox']['Left'] * width_shape)
        top = int(person['BoundingBox']['Top'] * height_shape)
        width = int(person['BoundingBox']['Width'] * width_shape)
        height = int(person['BoundingBox']['Height'] * height_shape)

        img = cv2.rectangle(img, (left, top), (left+width,
                            top+height), (255, 0, 0), 2)
        i += 1

    cv2.imshow('Detection Faces', img)
    cv2.waitKey(0)

    return
