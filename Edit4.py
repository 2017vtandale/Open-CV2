import numpy as np
import cv2, sys, re, math
#from matplotlib import pyplot as plt
import urllib.request

def sobeledge(image, threshold, gx, gy):
    blur = cv2.GaussianBlur(image.copy(),(5,5),0)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    Gx = cv2.filter2D(1.0*gray, -1, gx)
    #print(np.max(Gx))
    Gy = cv2.filter2D(1.0*gray, -1, gy)
    edgeimage = np.zeros((height,width,3), np.uint8)
    edgeimage[:,:] = (255,255,255)
    '''for row in range(image.shape[0]):
        for col in range(image.shape[1]):'''
    Edges = (Gx*Gx + Gy*Gy) > threshold
    #print(Edges)
    Edges = 255-(255*Edges).astype(np.uint8)

def getAngle(X, Y):
    if X == 0:
        return 90
    angle  = 1.0*math.atan2(Y,X)
    pi = math.pi
    if angle>=(-pi) and angle<(-7*pi/8):
        return 0
    elif angle>=(-7*pi/8) and angle<(-5*pi/8):
        return 45
    elif angle >=(-5*pi/8) and angle<= (-3*pi/8):
        return 90
    elif angle >= (-3*pi/8) and angle<=(-pi/8):
        return -45
    elif angle >= (-pi/8) and angle <= (pi/8):
        return 0
    elif angle >= (pi/8) and angle <=(3*pi/8):
        return 45
    elif angle>=(3*pi/8) and angle<=(5*pi/8):
        return 90
    elif angle>(5*pi/8) and  angle<=(7*pi/8):
        return -45
    elif angle>=(7*pi/8):
        return 0


def getNeigh(angle, row, col, rows, cols):
    Neighbors = []

    if angle == 0:
        if col-1 > 0:
            Neighbors.append([row, col-1])
        if col+1 <cols:
            Neighbors.append([row, col+1])

    elif angle == 90:
        if row-1>0:
            Neighbors.append([row-1, col])
        if row+1<rows:
            Neighbors.append([row+1, col])

    elif angle == 45:
        if col-1>0 and row-1>0:
            Neighbors.append([row-1, col-1])
        if row+1<rows and col+1<cols:
            Neighbors.append([row+1, col+1])

    else:
        if col+1< cols and row-1>0:
            Neighbors.append([row-1, col+1])
        if col-1>0 and row+1<rows:
            Neighbors.append([row+1, col-1])

    return Neighbors

def getPerpNeigh(angle, row, col, rows, cols):
    Neighbors = []

    if angle == 90:
        if col-1 > 0:
            Neighbors.append([row, col-1])
        if col+1 <cols:
            Neighbors.append([row, col+1])
    elif angle == 0:
        if row-1>0:
            Neighbors.append([row-1, col])
        if row+1<rows:
            Neighbors.append([row+1, col])
    elif angle == -45:
        if col-1>0 and row-1>0:
            Neighbors.append([row-1, col-1])
        if row+1<rows and col+1<cols:
            Neighbors.append([row+1, col+1])
    else:
        if col+1< cols and row-1>0:
            Neighbors.append([row-1, col+1])
        if col-1>0 and row+1<rows:
            Neighbors.append([row+1, col-1])

    return Neighbors

def isedge(neigh, row, col, gx, gy, angle, Edges):
    valgx = abs(gx[row,col][0])
    valgy = abs(gy[row, col][0])
    val = Edges[row,col][0]
    n01 = neigh[0][1]
    n00 = neigh[0][0]
    neigh1gx = abs(gx[n00, n01][0])
    neigh1gy = abs(gy[n00, n01][0])
    neigh1 = abs(Edges[n00, n01][0])
    if len(neigh)==2:
        n10 = neigh[1][0]
        n11 = neigh[1][1]
    else:
        n11 = neigh[0][1]
        n10 = neigh[0][0]
    neigh2gx = abs(gx[n10, n11][0])
    neigh2gy = abs(gy[n10, n11][0])
    neigh2 = abs(Edges[n10, n11][0])
    if angle == 0:
        return valgx>neigh1gx and valgx>neigh2gx
    elif angle == 90:
        return valgy>neigh1gy and valgy>neigh2gy
    return val>neigh1 and val>neigh2

def getallNeigh(image, row, col):
    rows = set([row+1, row-1, row]) - set(range(-10,0)) - set(range(image.shape[0], image.shape[0]+10))
    cols = set([col+1, col-1, col]) - set(range(-10,0)) - set(range(image.shape[1], image.shape[1]+10))
    neighs = []

    for x in rows:
        for z in cols:
            neighs.append([x, z])
    return neighs

def cannyedge(image, min, max, gx, gy):
    rows = img.shape[0]
    cols = img.shape[1]
    blur = cv2.GaussianBlur(image.copy(),(7,7),0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    Gx = cv2.filter2D(1.0*blur, -1, gx)
    Gy = cv2.filter2D(1.0*blur, -1, gy)
    Edges = (Gx*Gx + Gy*Gy)
    cEdges = np.copy(Edges)
    realedge = []
    notedge = []
    weak = []
    actual = []


    for row in range(rows):
        for col in range(cols):
            val = Edges[row, col][0]

            if val > max:
                angle = getAngle(Gx[row, col][0], Gy[row, col][0])
                Neighbors  = getNeigh(angle, row, col, rows, cols)
                #check = True

                if isedge(Neighbors, row, col, Gx, Gy, angle, Edges):
                    cEdges[row, col] = [255, 0, 0]
                    realedge.append([row, col])
                else:
                    cEdges[row, col] = [255, 255, 255]
            else:
                cEdges[row, col] = [255, 255, 255]

    c2Edges = np.copy(cEdges)
    actual = realedge
    checked = []

    while len(actual)>0:
        #checked.append(actual)
        #print(str(checked))
        for pix in actual:
            angle = getAngle(Gx[row, col][0], Gy[row, col][0])
            Neighbors  = getPerpNeigh(angle, pix[0], pix[1], rows, cols)
            for neigh in Neighbors:
                if (Edges[neigh[0], neigh[1]][0]>min) and (Edges[neigh[0], neigh[1]][0]<max) and (not [neigh[0], neigh[1]] in checked):
                    c2Edges[neigh[0], neigh[1]] = [255,0,0]
                    weak.append([neigh[0], neigh[1]])
            checked.append([neigh[0], neigh[1]])
        #print(str(len(weak)))
        actual = weak
        weak = []

    return (cEdges, c2Edges)



def avgpixel(img, prow, pcol):
    green = 0.0
    red = 0.0
    blue = 0.0
    posrow = set([prow-1, prow+1, prow, prow-2, prow+2]) - set(range(img.shape[0],img.shape[0]+10)) - set(range(-10, 0))
    poscol  = set([pcol, pcol-1, pcol+1, pcol-2, pcol+2]) - set(range(img.shape[1],img.shape[1]+10)) - set(range(-10, 0))
    pixel = 0.0

    for row in posrow:
        for col in poscol:
            colorcode = img[row,col]
            red+=colorcode[0]
            green+=colorcode[1]
            blue+=colorcode[2]
            pixel+=1

    return {"Red":(red/pixel), "Green":(green/pixel), "Blue":(blue/pixel)}

def blur(img):
    rows = img.shape[0]
    cols = img.shape[1]
    blurimg = img
    for row in range(rows):
        for col in range(cols):
            cc = avgpixel(img, row, col)
            blurimg[row, col] = [cc["Red"], cc["Green"], cc["Blue"]]
    return blurimg

def convert(img):
    #Edges = 255-img
    Edges = np.copy(img)
    Edges = cv2.cvtColor(Edges,cv2.COLOR_GRAY2RGB)
    #print(str(Edges))
    for row in range(Edges.shape[0]):
        for col in  range(Edges.shape[1]):
            if Edges[row, col][1] == 255:
                Edges[row, col] = [255, 0, 0]
            else:
                Edges[row, col] = [255,255,255]
    return Edges

def grayscale(img):
    cc = 0 #color code
    grayimg = img
    rows = img.shape[0]
    cols = img.shape[1]
    for row in range(rows):
        for col in range(cols):
            cc = img[row,col]
            newcc = .11*cc[0] + .59*cc[1] + .3*cc[2]
            grayimg[row, col] = [newcc, newcc, newcc]
    return grayimg

def addborder(image):
    for col in range(image.shape[1]):
        image[0,col] = 0
        image[image.shape[0]-1, col] = 0
    for row in range(image.shape[0]):
        image[row,0] = 0
        image[row, image.shape[1]-1] = 0
    return image


url = "http://www.thebackgammonstore.com/media/img/thechessstore/W1200-H600-Bffffff/wood_chess_set_packages/vnt/fk/fierce_knight_staunton_chess_sets_gr_bw_walnut_board_setup_bw_zoom_1200.jpg"
#url = ""
max = 2500
min = 1250
if len(sys.argv)>1:
    if len(sys.argv[1])>5:
        url = sys.argv[1]
        if len(sys.argv)>2:
            threshold = int(sys.argv[2])
    else:
        threshold = int(sys.argv[1])
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

cv2.imshow('imgWnd',img)
Currimg = img
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gradx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
grady = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
#gaus = np.array([])

Canny1, Canny2 = cannyedge(img, min, max, gradx, grady)
#Blah = convert(img)

Canny = addborder(convert(cv2.Canny(img, 100, 200)))
#print(str(Canny1))
cv2.imwrite('Canny1.jpg', Canny1)
cv2.imwrite('Canny2.jpg', Canny2 )
cv2.imshow("Canny1", Canny1)
cv2.imshow("Canny2", Canny2)
pressed = cv2.waitKey(0)

cv2.destroyAllWindows()
