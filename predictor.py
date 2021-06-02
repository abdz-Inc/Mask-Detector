import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from image_preprocess import img_process


with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("w8.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

'''
with open('test_mask.pkl', 'rb') as f:
    dataset = pickle.load(f)


x_ts_r = dataset['X'].T
x_ts = x_ts_r/255
x_ts = x_ts.reshape((991, 32, 32, 3))
permutation = list(np.random.permutation(x_ts.shape[0]))

x_ts = x_ts[permutation, :,:,:]

y_ts = dataset['Y'].T
y_ts = y_ts[permutation, :]

loaded_model.evaluate(x = x_ts, y = y_ts)

with open('train_maskv2.pkl', 'rb') as f:
    dataset = pickle.load(f)

x_tr = dataset['X']
print(x_tr.shape)
x = x_tr/255
#x = x.reshape((9999, 32, 32, 3))
permutation = list(np.random.permutation(x.shape[0]))
x = x[permutation, :, :, :]

y = dataset['Y']
y = y[permutation, :]

ex = x[1:2,:,:,:]#.reshape(64,64,3)
print(ex.shape)
plt.imshow(x[1,:,:,:]) #display sample training image
plt.show()
print(x_tr.shape)
print((loaded_model.predict(ex) > 0.5).astype("int32"))
'''
print("save the image to the testimg folder\nenter image name with extension '.jpg' or '.png'\n\nloss: 0.0731 - accuracy: 0.9840")
while True:
    filename = input("Enter img name: ")
    img_arr = img_process(filename)

    pred = np.squeeze((loaded_model.predict(img_arr) > 0.5).astype("int32"))
    if pred == 0:
        print("Mask Detected!")
    else:
        print("No mask Detected!")