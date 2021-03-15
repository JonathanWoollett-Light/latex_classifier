from cffi import FFI
import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import weakref
#import gc
#gc.disable()
# Prevents arrays from being garbage collected
# global_weakkeydict = weakref.WeakKeyDictionary()

EXTREME_BOUNDARY = 8
GLOBAL_BOUNDARY = 179
LOCAL_BOUNDARY = 18
FIELD_REACH = 35
FIELD_SIZE = 17

ffi = FFI()
ffi.cdef("""
    typedef struct { uint8_t extremeBoundary; uint8_t globalBoundary; uint8_t localBoundary; size_t fieldReach; size_t fieldSize; } BinarizationParameters;
    
    typedef struct { void* ptr; uint32_t size; } CArray;
    typedef struct { uint32_t x; uint32_t y; } Point;
    typedef struct { Point min; Point max; } Bound;
    typedef struct { CArray pixels; Bound bound; } SymbolPixels;

    CArray* segment(
        CArray*,
        BinarizationParameters*
    );
""")

C = ffi.dlopen("./target/release/latex_classifier.dll")

print("String:")
c_str = ffi.new("CArray*")
c_str.size = 18
c_str.ptr = ffi.new("char[]",b"tests/images/2.jpg")
# values = ffi.cast("char*",c_str.ptr)
# for i in range(c_str.size):
#     print(values[i],end=' ')
# print()

bin_params = ffi.new("BinarizationParameters*")
bin_params.extremeBoundary = EXTREME_BOUNDARY
bin_params.globalBoundary = GLOBAL_BOUNDARY
bin_params.localBoundary = LOCAL_BOUNDARY
bin_params.fieldReach = FIELD_REACH
bin_params.fieldSize = FIELD_SIZE

lines_arr = C.segment(c_str,bin_params)

print(lines_arr.size)
values = ffi.cast("CArray *",lines_arr.ptr) 

lines = []
for i in range(lines_arr.size):
    print("\t",values[i].size)
    inner_values = ffi.cast("SymbolPixels*",values[i].ptr)
    symbols = []
    for t in range(values[i].size):
        print("\t\t",end='')
        print("min",inner_values[t].bound.min.x,inner_values[t].bound.min.y,end=' ')
        print("max",inner_values[t].bound.max.x,inner_values[t].bound.max.y,end=' ')
        print("len:",inner_values[t].pixels.size,end='')
        print()

        pixels = ffi.cast("uint8_t*",inner_values[t].pixels.ptr)
        np_pixels = [pixels[p] for p in range(inner_values[t].pixels.size)]
        print("\t\tlen:",len(np_pixels))
        symbols.append(np_pixels)
    print("\tlen:",len(symbols))  
    lines.append(symbols)
print("len:",len(lines))

print(len(lines),len(lines[0]),len(lines[0][0]))

# test_img = np.reshape(np.array(lines[0][0]),(24,24))
# print("\n\n\n",test_img)
# img = Image.fromarray(abs((test_img*255)-255))
# img.show()

convert = lambda x: np.expand_dims(abs(np.reshape(np.array(x),(len(x),24,24)) - 1),-1)

np_lines = [convert(lines[i]) for i in range(len(lines))]
print(np_lines[0].shape)
# print("\n\n\n",np_lines[0][0,:,:,0])

model = keras.Sequential([
        keras.Input(shape=(24,24, 1)),
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
        layers.Dense(193, activation='softmax')
])
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

BATCH_SIZE = 256
EPOCHS = 2

model.load_weights("./weights")

# preds = [model.predict(np_lines[i]) for i in range(len(np_lines))]

print(np_lines[0].shape)
pred = model.predict(np_lines[0])

print(pred.shape)
print(np.argmax(pred,axis=1))
# print(pred)

trans_np_lines = np.transpose(np_lines[0],(0,2,1,3))

print(trans_np_lines.shape)
pred = model.predict(trans_np_lines)

print(pred.shape)
print(np.argmax(pred,axis=1))

# print(preds[0].shape)
# print(preds[0][0,:].shape)
# print(np.argmax(preds[0][0,:]))
# print(preds[0][0,:])
# print(np.argmax(preds[0][1,:]))
# print(preds[0][1,:])
# print(np.argmax(preds[0],axis=0))