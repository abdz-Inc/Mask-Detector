from PIL import Image
import numpy as np


def img_process(filename):

    path = "testimg\\"+filename
    img = Image.open(path).resize((32, 32))
    #Image.Image.show(img)
    img_array = (1/255)*np.array(img).reshape((1, 32, 32, 3))

    #print(img_array)

    return img_array


