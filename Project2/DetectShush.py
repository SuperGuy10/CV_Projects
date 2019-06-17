import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import sys

def detectShush(frame, location, ROI, cascade):
    if len(ROI) == 0:
        ROI=frame
    img_yuv = cv2.cvtColor(ROI, cv2.COLOR_BGR2YUV)
    rows, cols, bands = img_yuv.shape
    total=0
    for i in range(0, rows) :
        for j in range(0, cols) :
            total+= img_yuv[i, j][0]
    
    avg=total/(rows*cols)
    # print("Average: " + str(avg))   
    if avg > 190 or avg < 65:
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # print("Equalization done")
        total=0
        for i in range(0, rows) :
            for j in range(0, cols) :
                total+= img_yuv[i, j][0]
        avg=total/(rows*cols)
        # print("Average: " + str(avg)) 
    eq_frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    sharpness=cv2.Laplacian(eq_frame, cv2.CV_64F).var()
    # print("Face sharpness: ",sharpness)
    ROI=eq_frame
    if sharpness < 20:
        kernel = np.array([[1,1,1], [1,-7,1], [1,1,1]])
        ROI = cv2.filter2D(ROI, -1, kernel)
    scaleFactor=1.05
    minNeighbors=14
    mouths = cascade.detectMultiScale(ROI, scaleFactor, minNeighbors, 0, (2, 2)) 
    # while len(mouths) > 1:
    #     # scaleFactor=(scaleFactor-1)/3+1
    #     minNeighbors+=10
    #     # print("Mouths: "+str(len(mouths)))
    #     # print("scale: "+str(scaleFactor))
    #     # print("N: "+str(minNeighbors))
    #     mouths = cascade.detectMultiScale(ROI, scaleFactor, minNeighbors, 0, (2, 2)) 
    # print("Mouths: "+str(len(mouths)))
    for (mx, my, mw, mh) in mouths:
        mx += location[0]
        my += location[1]
        cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
    return len(mouths)

def detect(frame, faceCascade, mouthsCascade):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

#    gray_frame = cv2.equalizeHist(gray_frame)
#    gray_frame = cv2.medianBlur(gray_frame, 5)
    rows, cols = gray_frame.shape
    total=0
    for i in range(0, rows) :
        for j in range(0, cols) :
            total+= gray_frame[i, j]
    
    avg=total/(rows*cols)
    eq_frame=frame  
    if avg > 190 or avg < 65:
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        eq_frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    sharpness=cv2.Laplacian(eq_frame, cv2.CV_64F).var()
    # print("Sharpness: ",sharpness)
    blur_frame = cv2.GaussianBlur(eq_frame,(5,5),0)
    # print("Blurring done")
    sharpness=cv2.Laplacian(blur_frame, cv2.CV_64F).var()
    # print("Sharpness: ",sharpness)

    faces = faceCascade.detectMultiScale(
                blur_frame, 1.061, 13, 0|cv2.CASCADE_SCALE_IMAGE, (30, 30))
    detected = 0
    for (x, y, w, h) in faces:
        # ROI for mouth
        mouthROI = eq_frame[int(y+h/1.5):y+h, x:x+w]

        if detectShush(frame, (x, int(y+h/1.5)), mouthROI, mouthsCascade) == 0:
            detected += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    if len(faces)==0:
        num=detectShush(frame, (0, 0), [], mouthsCascade)
        if num == 0 :
            detected += 1
            cv2.rectangle(frame, (0,0), (cols,rows), (255, 0, 0), 2)
        else :
            cv2.rectangle(frame, (0,0), (cols,rows), (0, 255, 0), 2)
    #print("SHUSH DETECTED: ",detected)
    return detected


def run_on_folder(cascade1, cascade2, folder):
    if(folder[-1] != "/"):
        folder = folder + "/"
    files =  [join(folder,f) for f in listdir(folder) if isfile(join(folder,f))]
    windowName = None
    totalCnt = 0
    for f in files:
        img = cv2.imread(f)
        if type(img) is np.ndarray:
            lCnt = detect(img, cascade1, cascade2)
            totalCnt += lCnt
            if windowName != None:
                cv2.destroyWindow(windowName)
            windowName = f
            cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(windowName, img)
            cv2.waitKey(0)
    return totalCnt

def runonVideo(face_cascade, eyes_cascade):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Can't open default video camera!")
        exit()

    windowName = "Live Video"
    showframe = True
    while(showframe):
        ret, frame = videocapture.read()

        if not ret:
            print("Can't capture frame")
            break
        detect(frame, face_cascade, eyes_cascade)
        cv2.imshow(windowName, frame)
        if cv2.waitKey(30) >= 0:
            showframe = False
    
    videocapture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # check command line arguments: nothing or a folderpath
    if len(sys.argv) != 1 and len(sys.argv) != 2:
        print(sys.argv[0] + ": got " + len(sys.argv) - 1 +
        "arguments. Expecting 0 or 1:[image-folder]")
        exit()

    # load pretrained cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
    # mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

    if(len(sys.argv) == 2): # one argument
        folderName = sys.argv[1]
        detections = run_on_folder(face_cascade, mouth_cascade, folderName)
        print("Total of ", detections, "detections")
    else: # no arguments
        runonVideo(face_cascade, mouth_cascade)
