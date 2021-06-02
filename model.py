import tensorflow.keras.layers as tfl
import pickle
from tensorflow.keras.models import Model, Sequential
import numpy as np
from tensorflow.python.keras.layers.core import Dropout


def seq_model(input_shape):

    model = Sequential([
        tfl.Conv2D(6, (5, 5), strides = (1,1), input_shape = input_shape),
        tfl.ReLU(),
        tfl.MaxPool2D(pool_size=(2, 2), strides = (2, 2)),
        tfl.Conv2D(16, (5, 5), strides = (1,1)),
        tfl.ReLU(),
        tfl.MaxPool2D(pool_size=(2, 2), strides = (2, 2)),
        tfl.Flatten(),
        tfl.Dense(120, activation='relu'),
        tfl.Dropout(0.7),
        #tfl.Dense(256, activation='relu'),
        #tfl.Dropout(0.5),
        #tfl.Dense(128, activation='relu'),
        #tfl.Dropout(0.5),
        tfl.Dense(84, activation='relu'),
        tfl.Dropout(0.7),
        #tfl.Dense(16, activation='relu'),
        #tfl.Dropout(0.5),
        tfl.Dense(1, activation='sigmoid')
    ])

    return model


def func_model(input_shape):

    input = tfl.Input(shape= input_shape)
    Z1 = tfl.Conv2D(32, (5, 5), strides = (1,1))(input)
    A1 = tfl.ReLU()(Z1)
    P1 = tfl.MaxPool2D(pool_size=(2, 2), strides = (1, 1))(A1)
    Z2 = tfl.Conv2D(32, (5, 5), strides = (1,1))(P1)
    A2 = tfl.ReLU()(Z2)
    P2 = tfl.MaxPool2D(pool_size=(2, 2), strides = (1, 1))(A2)
    Z3 = tfl.Conv2D(16, (3, 3), strides = (1,1))(P2)
    A3 = tfl.ReLU()(Z3)
    Z4 = tfl.Conv2D(16, (3, 3), strides = (1,1))(A3)
    A4 = tfl.ReLU()(Z4)
    Z5 = tfl.Conv2D(16, (3, 3), strides = (1,1))(A4)
    A5 = tfl.ReLU()(Z5)
    P3 = tfl.MaxPool2D(pool_size=(3, 3), strides = (2, 2))(A5)
    F = tfl.Flatten()(P3)
    D1 = tfl.Dense(120, activation='relu')(F)
    Dr = tfl.Dropout(0.5)(D1)
    D2 = tfl.Dense(120, activation='relu')(Dr)
    Dr2 = tfl.Dropout(0.5)(D2)
    output = tfl.Dense(1, activation='sigmoid')(Dr2)
    model = Model(inputs = input, outputs = output)

    return model

def model_func(input_shape):

  input = tfl.Input(shape = input_shape)
  pad1 = tfl.ZeroPadding2D(padding = ((3 ,3),(3,3)))(input)
  Z1 = tfl.Conv2D(32, (7,7), strides = 1)(pad1)
  B1 = tfl.BatchNormalization(axis = 3)(Z1)
  A1 = tfl.ReLU()(B1)
  P1 = tfl.MaxPool2D((8,8), strides=8)(A1)
  F = tfl.Flatten()(P1)
  output = tfl.Dense(1, activation = "sigmoid")(F)

  model = Model(inputs = input, outputs = output)
  return model


def model_compiler(X, Y, epochs):

    model = func_model((32, 32, 3))
    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(x = X, y = Y, epochs = epochs, batch_size=256)
    return model

'''
with open('train_maskv2.pkl', 'rb') as f:
    dataset = pickle.load(f)

x = dataset['X']
#x = x.reshape((9999, 32, 32, 3))
permutation = list(np.random.permutation(x.shape[0]))
x = x[permutation, :, :, :]

y = dataset['Y']
y = y[permutation, :]
'''
with open('train_maskv2.pkl', 'rb') as f:
    dataset = pickle.load(f)

#print(dataset['X'].shape)
#input()
x = dataset['X'][:8000,:,:,:]
y = dataset['Y'][:8000,:]

model = model_compiler(x, y, 50)
'''
with open('test_mask.pkl', 'rb') as f:
    dataset = pickle.load(f)

x_ts = dataset['X'].T
x_ts = x_ts/255
x_ts = x_ts.reshape((991, 32, 32, 3))
permutation = list(np.random.permutation(x_ts.shape[0]))

x_ts = x_ts[permutation, :,:,:]

y_ts = dataset['Y'].T
y_ts = y_ts[permutation, :]
'''
x_ts = dataset['X'][8000:,:,:,:]
y_ts = dataset['Y'][8000:,:]

model.evaluate(x = x_ts, y = y_ts)


#save model

model_json = model.to_json()
with open("model_v4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("w8_v4.h5")
print("Saved model to disk")

#model.save()
'''
train_maskv2
func_model
256
50
loss: 0.0731 - accuracy: 0.9840
saved in _v3
'''