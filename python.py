import numpy as np

# from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Show test image
# test_img = np.reshape(np.array(lines[0][0]),(24,24))
# print("\n\n\n",test_img)
# img = Image.fromarray(abs((test_img*255)-255))
# img.show()

# convert = lambda x: np.expand_dims(abs(np.reshape(np.array(x),(len(x),24,24)) - 1),-1)

# np_lines = [convert(lines[i]) for i in range(len(lines))]
# print(np_lines[0].shape)
# print("\n\n\n",np_lines[0][0,:,:,0])

model = keras.Sequential([
        keras.Input(shape=(24,24, 1), name="inputs"),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.Dropout(0.2),
    
        layers.Conv2D(32,(3,3),padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
    
        layers.Conv2D(64,(3,3),padding='same',activation='relu'),
        layers.Dropout(0.2),
    
        layers.Conv2D(64,(3,3),padding='same',activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
    
        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.Dropout(0.2),

        layers.Conv2D(128,(3,3),padding='same',activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
    
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(2048,activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(193, activation='softmax', name="outputs")
])
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.load_weights("./weights")
model.save("net_file")