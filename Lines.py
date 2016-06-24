import numpy as np
import cv2, sys, re, math
import urllib.request
from fractions import Fraction

def hough(img):
    Canny = 255-cv2.Canny(img.copy(), 100, 200)
    Diag = math.sqrt(img.shape[0]*img.shape[0] + img.shape[1]*img.shape[1])
    Huff = np.zeros((181,int(2*Diag+1)))
    Point = {}
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if Canny[row,col] == 0:
                for i in range(180):
                    dist = (row*math.cos(i/57.2958) -(col*math.sin(i/57.2958)))
                    dist+=Diag
                    Huff[i,int(dist)]+=1
                    Point[i,int(dist)] = (row,col)
    #print(np.max(Distance))
    return (((1.0*Huff)/np.max(Huff)*255).astype(np.uint8),Point)

def Lines(img, huff, point):
    Lineimg = np.copy(img)
    Lines = 255-np.zeros_like(img)
    Huff = (huff>200)
    Huff = 255*Huff
    print(np.where(Huff==255))
    Pospoints = {}
    Diag = math.sqrt(img.shape[0]*img.shape[0] + img.shape[1]*img.shape[1])

    for t in range(Huff.shape[0]):
        for d in range(Huff.shape[1]):
            if Huff[t,d] ==255:
                Pospoints[(t,d)] = {}
    #print(Pospoints)
    for dat in Pospoints:
        theta, dist = dat
        rad = theta/57.2958
        y, x = point[dat]
        slope = math.tan(rad)
        b = (slope*x*-1)+y

        if b>0:
            xO = 0
            yO = b

        else:
            yO = 0
            xO = -1*b/slope

        if slope*(img.shape[1])+b<img.shape[0]:
            x1 = img.shape[1]
            y1 = slope*(img.shape[1])+b
        else:
            x1 = (img.shape[0]-b)/slope
            y1 = img.shape[0]

        cv2.line(Lineimg,(int(x1),int(y1)),(int(xO),int(yO)),(0,0,255),1)
        cv2.line(Lines,(int(x1),int(y1)),(int(xO),int(yO)),(0,0,255),1)
        '''
        while x+dx<img.shape[1] and x+dx>=0 and y+dy<img.shape[0] and y+dy>=0:
            #print("Here")
            Lineimg[int(y+dy),int(x+dx)] = [0,0,255]
            Lines[int(y+dy),int(x+dx)] = [0,0,255]
            dx+=1
            dy+=slope
            #print(y+dy,x+dx)
        dx = 0
        dy = 0
        while x-dx<img.shape[1] and x-dx>=0 and y-dy<img.shape[0] and y-dy>=0:
            Lineimg[int(y-dy),int(x-dx)] = [0,0,255]
            Lines[int(y-dy),int(x-dx)] = [0,0,255]
            dx+=1
            dy+=slope
        '''
    return Lineimg, Lines

url = "https://www.krispykreme.com/SharedContent/User/aa/aa0ef72f-77be-4fef-8d43-bca50c530085.png"
#url = "Orignal.jpg"
if len(sys.argv)>1:
    url = sys.argv[1]
if re.compile("^\\w*\\:\\/\\/").search(url) is None:
  img = cv2.imread(url, cv2.IMREAD_COLOR)

else:
  resp = urllib.request.urlopen(url)
  img = np.asarray(bytearray(resp.read()), dtype = "uint8")
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)

# Allows window to be resized by user (in theory)
cv2.namedWindow('imgWnd', cv2.WINDOW_NORMAL)
height, width = img.shape[:2]    # Get image dimensions
widthMax = 1024                  # The screen width, in pixels

if (width > widthMax):
  cv2.resizeWindow('imgWnd', widthMax, int(height*widthMax/width))
#cv2.imshow("Edge", img)
#drawline(Edges,img, 10, 10)
Huff, point = hough(img)
Huff =255-Huff
#print(np.max(Huff))
cv2.imshow("imgWnd", Huff)
Line, Blank = Lines(img,(255-Huff), point)
cv2.imshow("Lines",cv2.resize(Line,(300,400),interpolation = cv2.INTER_CUBIC))
cv2.imshow("Blah",Blank)
cv2.imshow("canny", 255-cv2.Canny(img.copy(), 100, 200))
cv2.waitKey(0)
cv2.destroyAllWindows()
