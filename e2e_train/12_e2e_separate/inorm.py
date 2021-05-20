#!/usr/bin/env python

import cv2, glob, os

video_files = glob.glob("*.mp4")
caps = [ cv2.VideoCapture(name) for name in video_files ]

cap_length = max([ int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps ])

if not os.path.isdir("./image_road/"):
    os.mkdir("./image_road/")
if not os.path.isdir("./image_background/"):
    os.mkdir("./image_background/")

for i in range(cap_length):
    for cap in caps:
        cap_name = str(cap).split(" ")[1][:-1]
        
        ret, img = cap.read()
        if not ret:
            continue
        img = cv2.resize(img, dsize=(200, 112))
        img_back = img[:56,:]
        img_road = img[56:,:]
        #img = img[46:,:]
        cv2.imwrite("image_road/"+str(i)+"-"+cap_name+".jpg", img_road)
        cv2.imwrite("image_background/"+str(i)+"-"+cap_name+".jpg", img_back)

for cap, video_file in zip(caps, video_files):
    current_name = str(video_file).split(".")[0]
    cap_name = str(cap).split(" ")[1][:-1]
    os.rename(current_name+".mp4", cap_name+".mp4")
    os.rename(current_name+".csv", cap_name+".csv")
 
for cap in caps:
    cap.release()

cv2.destroyAllWindows()