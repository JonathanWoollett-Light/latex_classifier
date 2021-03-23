from PIL import Image
import numpy as np

from cffi import FFI
import ctypes

# import gc
# gc.disable

import time

import cv2 # display#

import weakref
global_weakkeydict = weakref.WeakKeyDictionary()

ffi = FFI()
ffi.cdef("""
    typedef struct { uint8_t extremeBoundary; uint8_t globalBoundary; uint8_t localBoundary; size_t fieldReach; size_t fieldSize; } BinarizationParameters;
    typedef struct { void* ptr; uint32_t size; } CArray;
    typedef struct { CArray symbols; CArray pixels; } CReturn;

    CReturn* segment_buffer(CArray*, size_t, size_t, BinarizationParameters*);
""")

C = ffi.dlopen("./target/release/latex_classifier.dll")

EXTREME_BOUNDARY = 8
GLOBAL_BOUNDARY = 179
LOCAL_BOUNDARY = 18
FIELD_REACH = 35
FIELD_SIZE = 17

img = Image.open("tests/images/3.jpg")
arr = np.asarray(img)
# print(img)
print("arr.shape:",arr.shape)

preStart = time.time()

flat_arr = arr.flatten('C').astype("uint8")
# print(flat_arr.shape,len(flat_arr),flat_arr.dtype)
# print(type(flat_arr.tolist()[0]))
# black = [i for i in range(len(flat_arr)) if flat_arr[i]==0]
# print(black[0:10])

c_arr = ffi.new("CArray*")
c_arr.size = len(flat_arr)

# print(flat_arr.tolist()[0:20])
c_arr_ptr = ffi.new("uint8_t[]",flat_arr.tolist())
c_arr.ptr = c_arr_ptr
global_weakkeydict[c_arr] = c_arr_ptr

bin_params = ffi.new("BinarizationParameters*")
bin_params.extremeBoundary = EXTREME_BOUNDARY
bin_params.globalBoundary = GLOBAL_BOUNDARY
bin_params.localBoundary = LOCAL_BOUNDARY
bin_params.fieldReach = FIELD_REACH
bin_params.fieldSize = FIELD_SIZE

preTime = time.time() - preStart

rustStart = time.time()
rust_rtn = C.segment_buffer(c_arr,arr.shape[1],arr.shape[0],bin_params)
rustTime = time.time() - rustStart

postStart = time.time()

pixels_arr = rust_rtn.pixels
print("SIZE:",pixels_arr.size)
pixels = ffi.cast("uint8_t*",pixels_arr.ptr)
debug_arr = np.array([pixels[i] for i in range(pixels_arr.size)]).astype("uint8")
# print("debug_arr.dtype:",debug_arr.dtype)
debug_img = np.reshape(debug_arr,arr.shape,'C')
# print(debug_img.shape)
img = Image.fromarray(debug_img, 'RGB')
img = Image.fromarray(debug_img, 'RGB')
postTime = time.time() - postStart

print("Time:",f"{preTime:.3f}","->",f"{rustTime:.3f}","->",f"{postTime:.3f}")


img.save("test_image.png")
img.show()

lines_arr = rust_rtn.symbols
print(lines_arr.size)
values = ffi.cast("CArray *",lines_arr.ptr)

# ------------------------------------------------------------
# Printing
# ------------------------------------------------------------
# lines = []
# for i in range(lines_arr.size):
#     print("\t",values[i].size)
#     inner_values = ffi.cast("SymbolPixels*",values[i].ptr)
#     symbols = []
#     for t in range(values[i].size):
#         print("\t\t",end='')
#         print("min",inner_values[t].bound.min.x,inner_values[t].bound.min.y,end=' ')
#         print("max",inner_values[t].bound.max.x,inner_values[t].bound.max.y,end=' ')
#         print("len:",inner_values[t].pixels.size,end='')
#         print()

#         pixels = ffi.cast("uint8_t*",inner_values[t].pixels.ptr)
#         np_pixels = [pixels[p] for p in range(inner_values[t].pixels.size)]
#         print("\t\tlen:",len(np_pixels))
#         symbols.append(np_pixels)
#     print("\tlen:",len(symbols))  
#     lines.append(symbols)
# print("len:",len(lines))

# print(len(lines),len(lines[0]),len(lines[0][0]))

# ------------------------------------------------------------
# Classification
# ------------------------------------------------------------
# convert = lambda x: np.expand_dims(abs(np.reshape(np.array(x),(len(x),24,24)) - 1),-1)

# np_lines = [convert(lines[i]) for i in range(len(lines))]
# print(np_lines[0].shape)
# print("\n\n\n",np_lines[0][0,:,:,0])

# model = keras.Sequential([
#         keras.Input(shape=(24,24, 1)),
#         layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
#         layers.Dropout(0.2),
    
#         layers.Conv2D(32,(3,3),padding='same', activation='relu'),
#         layers.MaxPooling2D(pool_size=(2,2)),
    
#         layers.Conv2D(64,(3,3),padding='same',activation='relu'),
#         layers.Dropout(0.2),
    
#         layers.Conv2D(64,(3,3),padding='same',activation='relu'),
#         layers.MaxPooling2D(pool_size=(2,2)),
    
#         layers.Conv2D(128,(3,3),padding='same',activation='relu'),
#         layers.Dropout(0.2),

#         layers.Conv2D(128,(3,3),padding='same',activation='relu'),
#         layers.MaxPooling2D(pool_size=(2,2)),
    
#         layers.Flatten(),
#         layers.Dropout(0.2),
#         layers.Dense(2048,activation='relu'),
#         layers.Dropout(0.2),
#         layers.Dense(193, activation='softmax')
# ])
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# BATCH_SIZE = 256
# EPOCHS = 2
# model.load_weights("./weights")

# print(np_lines[0].shape)
# pred = model.predict(np_lines[0])

# print(pred.shape)
# print(np.argmax(pred,axis=1))

# trans_np_lines = np.transpose(np_lines[0],(0,2,1,3))

# print(trans_np_lines.shape)
# pred = model.predict(trans_np_lines)

# print(pred.shape)
# print(np.argmax(pred,axis=1))

# ------------------------------------------------------------
# Video
# ------------------------------------------------------------
# vid = cv2.VideoCapture(0)

# while(True):
#     ret,frame = vid.read()
    
#     # for x in range(frame.shape[0]):
#     #     for y in range(frame.shape[1]):
#     #         if np.sum(frame[x,y,:]) < 200:
#     #             frame[x,y,:] = 255
        

#     # pixels_arr = rust_rtn.pixels
#     # pixels = ffi.cast("uint8_t*",pixels_arr.ptr)
#     # debug_arr = np.array([pixels[i] for i in range(pixels_arr.size)]).astype("uint8")
#     # debug_img = np.reshape(debug_arr,arr.shape,'C')
#     # img = Image.fromarray(debug_img, 'RGB')

#     cv2.imshow("frame",frame)

#     if cv2.waitKey(10) & 0xFF == ord('e'):
#         print(frame.shape)
        

# vid.release()
# cv2.destroyAllWindows()