import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys

def detectWink(frame, location, ROI, cascade):
    if len(ROI) == 0:
        ROI=frame

    sharpness=cv2.Laplacian(ROI, cv2.CV_64F).var()
    # print("Face sharpness: ",sharpness)
    if sharpness < 20:
        kernel = np.array([[1,1,1], [1,-7,1], [1,1,1]])
        ROI = cv2.filter2D(ROI, -1, kernel)

    scaleFactor=1.1
    minNeighbors=15
    # print("scale: "+str(scaleFactor))
    # print("N: "+str(minNeighbors))
    eyes = cascade.detectMultiScale(ROI, scaleFactor, minNeighbors, 0|cv2.CASCADE_SCALE_IMAGE, (2  , 2) )

    while scaleFactor > 1.0005 and len(eyes) == 0:
        scaleFactor=(scaleFactor-1)/2+1
        minNeighbors+=10
        # print("Eyes: "+str(len(eyes)))
        # print("scale: "+str(scaleFactor))
        # print("N: "+str(minNeighbors))
        eyes = cascade.detectMultiScale(ROI, scaleFactor, minNeighbors, 0|cv2.CASCADE_SCALE_IMAGE, (2  , 2) )

    for e in eyes:
        e[0] += location[0]
        e[1] += location[1]
        x, y, w, h = e[0], e[1], e[2], e[3]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 2)

    return len(eyes)   # number of eyes is one

def detect(frame, faceCascade, eyesCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # possible frame pre-processing:
    rows, cols = gray_frame.shape
    total=0
    for i in range(0, rows) :
        for j in range(0, cols) :
            total+= gray_frame[i, j]
    
    avg=total/(rows*cols)
    # print("Average: " + str(avg))   
    eq_frame=frame  
    if avg > 190 or avg < 65:
        # print("Average: " + str(avg))
        # print("Equalization done")
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        eq_frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # sharpness=cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    # print("sharpness: ",sharpness)
    # if sharpness > 100:
        # frame = cv2.GaussianBlur(frame,(3,3),0)
    blur_frame = cv2.GaussianBlur(eq_frame,(3,3),0)
    # blur_frame = cv2.medianBlur(frame, 5)

    scaleFactor = 1.131 # range is from 1 to ..
    minNeighbors = 15  # range is from 0 to ..
    flag = 0|cv2.CASCADE_SCALE_IMAGE # either 0 or 0|cv2.CASCADE_SCALE_IMAGE 
    minSize = (30,30) # range is from (0,0) to ..
    faces = faceCascade.detectMultiScale(
        blur_frame, 
        scaleFactor, 
        minNeighbors, 
        flag, 
        minSize)

    detected = 0
    # print("Faces: "+str(len(faces)))
    for f in faces:
        x, y, w, h = f[0], f[1], f[2], f[3]
        h1=int(y+h/1.5)
        faceROI = eq_frame[y:h1, x:x+w]
        if detectWink(frame, (x, y), faceROI, eyesCascade) == 1:
            detected += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    if len(faces)==0:
        num=detectWink(frame, (0, 0), [], eyesCascade)
        if num == 1 :
            detected += 1
            cv2.rectangle(frame, (0,0), (cols,rows), (255, 0, 0), 2)
        if num == 2 :
            cv2.rectangle(frame, (0,0), (cols,rows), (0, 255, 0), 2)
    #print("WINK DETECTED: ",detected)
    return detected


def run_on_folder(cascade1, cascade2, folder):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files = [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]

    windowName = None
    totalCount = 0
    for f in files:
        img = cv2.imread(f, 1)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            totalCount += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCount

def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showlive = True
    while(showlive):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            exit()

        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showlive = False
    
    # outside the while loop
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1 
              + "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades 
                                      + 'haarcascade_eye.xml')

    if(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, eye_cascade, folderName)
        print("Total of ", detections, "detections")
    else: # no arguments
        runonVideo(face_cascade, eye_cascade)

