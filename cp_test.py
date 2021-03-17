from PIL import Image
import numpy as np

from cffi import FFI
import ctypes

import cv2 # display

# ffi = FFI()
# ffi.cdef("""
#     typedef struct { uint8_t extremeBoundary; uint8_t globalBoundary; uint8_t localBoundary; size_t fieldReach; size_t fieldSize; } BinarizationParameters;
#     typedef struct { void* ptr; uint32_t size; } CArray;

#     CArray* segment_buffer(CArray*, size_t, size_t, BinarizationParameters*);
# """)

# C = ffi.dlopen("./target/release/latex_classifier.dll")

# EXTREME_BOUNDARY = 8
# GLOBAL_BOUNDARY = 179
# LOCAL_BOUNDARY = 18
# FIELD_REACH = 35
# FIELD_SIZE = 17

img = Image.open("tests/images/med_test.png")
arr = np.asarray(img)
print(img)
print(arr.shape)
flat_arr = arr.flatten()
print(flat_arr.shape,len(flat_arr))
# c_arr = ffi.new("CArray*")
# c_arr.size = len(flat_arr)
# c_arr.ptr = ffi.new("uint8_t[]",flat_arr.tolist())

# bin_params = ffi.new("BinarizationParameters*")
# bin_params.extremeBoundary = EXTREME_BOUNDARY
# bin_params.globalBoundary = GLOBAL_BOUNDARY
# bin_params.localBoundary = LOCAL_BOUNDARY
# bin_params.fieldReach = FIELD_REACH
# bin_params.fieldSize = FIELD_SIZE

# lines_arr = C.segment_buffer(c_arr,arr.shape[0],arr.shape[1],bin_params)

# print(lines_arr.size)
# values = ffi.cast("CArray *",lines_arr.ptr) 

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

# vid = cv2.VideoCapture(0)

# while(True):
#     ret,frame = vid.read()
    
#     # lines_arr = C.segment(c_str,bin_params)

#     cv2.imshow("frame",frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# vid.release()
# cv2.destroyAllWindows()