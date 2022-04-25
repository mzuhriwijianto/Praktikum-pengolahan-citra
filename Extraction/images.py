import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image

def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x
img, x = load_image("/img/hijau05.jpg")
print("shape of x: ", x.shape)
print("data type: ", x.dtype)
plt.imshow(img)