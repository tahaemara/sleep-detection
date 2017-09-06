import sys
import os
import dlib
import glob
from skimage import io
import numpy as np

if len(sys.argv) != 3:
    print(

        "execute this program by running: \n"
        "python sleep_detection.py  ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat /path/to/image.jpg"
        "You can download a trained facial shape predictor from:\n "
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
image_path = sys.argv[2]
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

win = dlib.image_window()
np.set_printoptions(threshold=sys.maxsize)
img = io.imread(image_path)
win.clear_overlay()
win.set_image(img)
dets = detector(img, 1)
vec = np.empty([68, 2], dtype = int)

status="Not Sleeping"

print("Number of faces detected: {}".format(len(dets)))
for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)

        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        # Draw the face landmarks on the screen.
        
        for b in range(68):
         vec[b][0] = shape.part(b).x
         vec[b][1] = shape.part(b).y
            
        if (vec[46][1]-vec[44][1]<=5 and vec[47][1]-vec[43][1]<=5 and vec[40][1]-vec[38][1]<=5 and vec[41][1]-vec[37][1]<=5):
              status="sleeping"
	print(status)

        win.add_overlay(shape)

win.add_overlay(dets)
win.set_title(status)
dlib.hit_enter_to_continue()

