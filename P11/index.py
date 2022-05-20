import json
import boto3
import matplotlib.pyplot as plt
import cv2

import os
os.system('cls')

photo = 'img/barcelona.jpg'

img = cv2.imread(photo)
height_shape = img.shape[0]
width_shape = img.shape[1]

with open(photo, 'rb') as source_image:
    source_bytes = source_image.read()

client = boto3.client(
    'rekognition',
    aws_access_key_id='aws_access_key_id',
    aws_secret_access_key='aws_secret_access_key',
    region_name='ap-southeast-1'
)

response = client.detect_faces(
    Image={'Bytes': source_bytes},
    Attributes=['ALL']
)

print(response)

i = 1
for person in response['FaceDetails']:
    left = int(person['BoundingBox']['Left']*width_shape)
    top = int(person['BoundingBox']['Top']*height_shape)
    width = int(person['BoundingBox']['Width']*width_shape)
    height = int(person['BoundingBox']['Height']*height_shape)

    img = cv2.rectangle(img, (left, top), (left+width,
                        top+height), (255, 0, 0), 2)
    i += 1

cv2.imshow('Detection Face', img)
cv2.waitKey(0)