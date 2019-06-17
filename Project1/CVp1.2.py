import cv2
import numpy as np
import sys

#inverse gamma correction
def invgamma(x):
    for i in range(0, x.shape[0]):
        if (x[i] < 0.03928):
            x[i] /= 12.92
        else:
            x[i] = pow(((x[i] + 0.055) / 1.055), 2.4)
    return x

#restrict the RGB value to be in the range of [0,1]
def limit(x): 
    for i in range(0, x.shape[0]):
        if x[i] > 1:
            x[i] = 1
        else:
            if x[i] < 0:
                x[i] = 0
    return x

def XYZ2Luv(x):
    #in case x=[0,0,0]
    if (x[0] == 0 and x[1] == 0 and x[2] == 0):
        return np.array([0, 0, 0], dtype = np.float_)
    if (x[1] > 0.008856):
        L = 116*pow(x[1], 1/3) -16
    else:
        L = 903.3 * x[1]
    d = x[0] + 15 * x[1] + 3 * x[2]
    up1 = 4 * x[0]/d
    vp1 = 9 * x[1]/d
    u = 13 * L * (up1-0.19771071800208116545265348595213)
    v = 13 * L * (vp1-0.46826222684703433922996878251821)
    return np.array([L, u, v], dtype = np.float_)

def Luv2XYZ(x):
    #in case L is 0
    if (x[0] == 0):
        return np.array([0, 0, 0], dtype = np.float_)
    up2 = (x[1] + 13 * 0.19793943 * x[0]) / (13 * x[0])
    vp2 = (x[2] + 13 * 0.46831096 * x[0]) / (13 * x[0])
    if (x[0] > 7.9996):
        Y = pow(((x[0] + 16) / 116), 3)
    else:
        Y = x[0] / 903.3
    if (vp2 == 0):
        X = 0
        Z = 0
    else:
        X = Y * 2.25 * up2 / vp2
        Z = Y * (3 - 0.75 * up2 - 5 * vp2) / vp2
    return np.array([X, Y, Z], dtype = np.float_)
    
#gamma correction
def gamma(x):
    for i in range(0,x.shape[0]):
        if x[i] < 0.00304:
            x[i] *= 12.92
        else:
            x[i] = 1.055*pow(x[i], 1/2.4)-0.055
    return x

'''
For the main function part:
including read image, arguments setting, etc
'''

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

#convert matrix references to https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
RGB2XYZ=np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334,0.119193,0.950227]], np.float_)
XYZ2RGB=np.array([[3.240479, -1.53715, -0.498535], [-0.969256, 1.875991, 0.041556], [0.055648,-0.204043,1.057311]], np.float_)

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None):
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

rows, cols, bands = inputImage.shape # bands == 3
sumP = rows * cols
W1 = round(w1 * (cols-1))
H1 = round(h1 * (rows-1))
W2 = round(w2 * (cols-1))
H2 = round(h2 * (rows-1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels
mL = 2147483647
ML = -2147483648
tmp = np.copy(inputImage)

#for specific area of the original image
area = np.zeros([H2-H1+1, W2-W1+1, bands], dtype = np.float_)
for i in range(H1, H2+1): 
    for j in range(W1, W2+1):
        
        #identify the area selected in the original image
        if (i == H1 or i == H2 or j == W1 or j == W2): 
           tmp[i, j] = [255, 255, 255]
        b, g, r = inputImage[i, j]
        area[i-H1, j-W1] = r, g, b
        area[i-H1, j-W1, 0], area[i-H1, j-W1, 1], area[i-H1, j-W1, 2] = area[i-H1, j-W1, 0]/255, area[i-H1, j-W1, 1]/255, area[i-H1, j-W1, 2]/255
        
        area[i-H1, j-W1] = np.array(limit(invgamma(area[i-H1, j-W1])), np.float_).T
        area[i-H1, j-W1] = np.array(np.dot(RGB2XYZ, area[i-H1, j-W1]))
        area[i-H1, j-W1] = np.array(XYZ2Luv(area[i-H1, j-W1]).T,np.float_)
        
        #find the min value of L
        if (area[i-H1, j-W1, 0] < mL):
            mL = area[i-H1, j-W1, 0]
        #find out the max value of L
        if (area[i-H1, j-W1, 0] > ML):
            ML = area[i-H1, j-W1, 0]


cv2.imshow('Selected Area', tmp)

#calculate L interval between discretized values
intervalL = (ML - mL) / 100 
sum = np.zeros([101], dtype = np.int_) 
mapValue = np.zeros([101], dtype = np.int_)

inputTemp = np.zeros([inputImage.shape[0], inputImage.shape[1], inputImage.shape[2]], dtype=np.float_)
outputTemp = np.zeros([inputImage.shape[0], inputImage.shape[1], inputImage.shape[2]], dtype=np.uint8)

for i in range(0,inputTemp.shape[0]):
    #apply scaling to entire image
    for j in range(0,inputTemp.shape[1]):
        b, g, r = inputImage[i, j]
        inputTemp[i, j] = np.array([r, g, b], dtype = np.float_)
        inputTemp[i, j] = np.dot(1/255, inputTemp[i, j])
        inputTemp[i, j] = np.array(limit(invgamma(inputTemp[i, j])), dtype=np.float_).T
        inputTemp[i, j] = np.array(np.dot(RGB2XYZ, inputTemp[i, j]))
        inputTemp[i, j] = np.array(XYZ2Luv(inputTemp[i, j]))
        if (inputTemp[i, j, 0] < mL):
            sum[0] += 1
        else:
            if (inputTemp[i, j, 0] > ML):
                sum[100] += 1
            else:
                #in case all L have same value
                if (mL != ML):
                    #calculate the index to which current pixel belong
                    interTemp = int(round((inputTemp[i, j, 0] - mL) / intervalL))
                    sum[interTemp] += 1
#print("Count:\n",sum)

mapValue[0] = sum[0] * 101 / (2 * sumP)
#calculate distribution
for i in range(1, sum.shape[0]):
    sum[i] = sum[i-1] + sum[i]
    mapValue[i] = round((sum[i-1] + sum[i]) * 101 / (2 * sumP) - 0.5)
    if (mapValue[i] == 101):
        mapValue[i] -= 1

for i in range(0, inputTemp.shape[0]):
    for j in range(0, inputTemp.shape[1]):
        if (inputTemp[i, j, 0] < mL):
            inputTemp[i, j, 0] = 0
        else:
            if (inputTemp[i, j, 0] > ML):
                inputTemp[i, j, 0] = 100
            else:
                if (mL != ML):
                    inputTemp[i, j, 0] = mapValue[int(round((inputTemp[i, j, 0] - mL) / intervalL))]
        inputTemp[i,j] = np.array(Luv2XYZ(inputTemp[i, j]), np.float_)
        inputTemp[i,j] = np.array(np.dot(XYZ2RGB,inputTemp[i, j].T), np.float_)
        inputTemp[i,j] = limit(inputTemp[i, j])
        inputTemp[i,j] = np.array(gamma(inputTemp[i, j]), np.float_)
        inputTemp[i,j] = np.array(np.dot(255, inputTemp[i,j].T), np.float_)
        outputTemp[i,j] = np.array([inputTemp[i, j, 2], inputTemp[i, j, 1], inputTemp[i, j, 0]], np.uint8).T

cv2.imshow('Final Result', outputTemp)


# end of example of going over area

outputImage = np.zeros([rows, cols, bands], dtype=np.uint8)

for i in range(0, rows) :
    for j in range(0, cols) :
        b, g, r = inputImage[i, j]
        outputImage[i,j] = [b, g, r]
cv2.imwrite(name_output, outputTemp);
# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
