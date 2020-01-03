"""
    Pre-process raw Trump pics, by detecting a face, stripping it,
    and resizing it (keeping the aspect ratio) and then saving it
    on disk, under the `/data` directory.
"""

import os
from concurrent import futures
import threading
import cv2, imutils
import numpy as np
import face_recognition
from PIL import Image

rawdir = "raw"
#
images = []

#
#   Resize all pictures to the same dims, because dlib throws a bitchfit if they are
#   different! Go for a simple size (512x512) for face detection 
#   (if that's too small, then we don't really want it anyway)
#
def voodoo(fname):
    image  = face_recognition.load_image_file(fname)
    coords = face_recognition.face_locations(image)
    #
    # left, top right, bottom
    #
    i = 0
    for idx in coords:
        print(idx)
        top, right, bottom, left = idx
        face_image = image[top:bottom, left:right]
        pil_image  = Image.fromarray(face_image)
        size = 224, 224
        pil_image.thumbnail(size)
        file = os.path.basename(fname) + "_" + str(i)
        file = os.path.join("trump_faces", file)
        print(file)
        pil_image.save(file, "jpeg")
        i += 1

# 
#   Load the batch in RAM (if this brakes your puny puter then don't use a batch)
#   I have 6 cores and 12 threads, so do the math yo!
#
jobs = []
with futures.ThreadPoolExecutor(max_workers=12) as ex:
    for filename in os.listdir(rawdir):
        if filename.endswith(".jpg"):
            img_file = os.path.join(rawdir, filename)
            jobs.append(ex.submit(voodoo, img_file))

images  = []
i = 0
for f in jobs:
    f.result()
#
