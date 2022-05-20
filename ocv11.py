from http import client
import boto3
import matplotlib.pyplot as plt
import cv2


def connect():
    client = boto3.client(
        'rekognition',
        aws_access_key_id='AKIATMBRT6Z6VZD4ONIB',
        aws_secret_access_key='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',
        region_name='ap-southeast-1'
    )

    return client


def detect_faces():

    client = connect()
    photo = "timnas.jpg"

    img = cv2.imread(photo)
    height_shape = img.shape[0]
    width_shape = img.shape[1]

    with open(photo, 'rb') as source_image:
        source_bytes = source_image.read()

    response = client.detect_faces(
        Image={'Bytes': source_bytes},
        Attributes=['ALL']
    )

    # MinConfidence=99
    print(response)

    for person in response['FaceDetails']:

        print("Width:"+str(person['BoundingBox']['Width']))
        print("Height:"+str(person['BoundingBox']['Height']))
        print("Left:"+str(person['BoundingBox']['Left']))
        print("Top:"+str(person['BoundingBox']['Top']))
        print("Age Low:"+str(person['AgeRange']['Low']))
        print("Age High:"+str(person['AgeRange']['High']))

        left = int(person['BoundingBox']['Left']*width_shape)
        top = int(person['BoundingBox']['Top']*height_shape)
        width = int(person['BoundingBox']['Width']*width_shape)
        height = int(person['BoundingBox']['Height'] * height_shape)

        img = cv2.rectangle(img, (left, top), (left+width,
                                               top+height), (255, 0, 0), 2)

    cv2.imshow('Detection Faces', img)
    cv2.waitKey(0)


detect_faces()
