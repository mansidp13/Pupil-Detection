import os
import cv2 as cv
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
import time
from multiprocessing import Pool
from PIL import Image
import upload
import statistics

def moving_avg(data_list, window_size):
    n = len(data_list)
    avg = []*n #creates and empty array
    for i in range (n):
        if i < window_size:
            avg.append(data_list[i]) ## keep the data as it before the avg window
        else:
            s = 0
            for p in range(window_size):
               s = data_list[i-p] + s   
            av = s / window_size  ## takes the average as the counter goes
            avg.append(av)
    return avg

## incorrect data elimination using the difference from previous data
def eliminate_outliers(data_list,difference):
    n = len(data_list)
    for i in range(n):
        while i > 0 and abs(data_list[i] - data_list[i-1]) > difference:
            data_list[i] = data_list[i-1]  ## if the difference is bigger than threshold, previous data is taken
    return data_list

## function to detect the circle using edge detection
def circle_detection(img_file):
    img = cv.imread(img_file) ## read the image file
    
    ## separate the file name from its path name
    m = img_file.split('/')[-1]
    m1 = m.split('.')[0]

    ## variable to get the timestamp from img number
    t1 = float(m1)
    t = float(t1/6) # the frame/second is 6 in this case
    h, w = img.shape[:2]

##  creating a mask around the eyes to help code not detect other circles
    ## adusting the radius and center for the mask around the eyes
##    img = cv.resize(img,(w+55,h), interpolation=cv.INTER_AREA)
##    radius = h/2
##    center = int(w/2),int(h/2)
##    print(center)
##    
####    if center is None: # use the middle of the image
####        center = (int(w/2), int(h/2))
####    if radius is None: # use the smallest distance between the center and image walls
####        radius = int(h/2)
##    Y, X = np.ogrid[:h, :w]
##    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
##    mask = dist_from_center <= radius
##   
##    masked_img = img.copy()
##    masked_img[~mask] = 0
##    img = masked_img

    
    img = cv.medianBlur(img,11)# blurring the image
    cimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #converts to grayscale
##    cimg = cv.equalizeHist(cimg) #equalizes contrast
    cimg = cv.convertScaleAbs(cimg, alpha= 1.1, beta= 15)

    circles = cv.HoughCircles(cimg,cv.HOUGH_GRADIENT,2 ,1000, param1=60,param2=120,minRadius=14,maxRadius=320)

    ## when the circle is detected
    if circles is not None:
        circles = np.int16(np.around(circles)) # converts values into integer
        for i in circles[0,:]:
        #    # draw the outer circle
            cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        #    # draw the center of the circle
            cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            r = circles[0][0][2] ## the radius
            x_pix = circles[0][0][0]  ## the x coordinate of the center
            y_pix = circles[0][0][1]  ## the y coordinate of the center
            
    ## when the function does not detect any circle i.e. eyes re closed       
    else:
        ## assigns the 0s that are adjusted later
        r = 0
        x_pix = 0
        y_pix = 0

    ## resizing the image to see the whole image without panning
##    cimg = cv.resize(cimg,(w//2,h//2), interpolation=cv.INTER_AREA)
##    cv.imshow(img_file,cimg)
##    cv.waitKey(0)

    ## returns the tuple of time, radius, center coordinates
    return t,r,x_pix,y_pix




if __name__ == '__main__':
    try:
        c=[] ## initializing the list
        w = 5   # window size for moving average
        s = 0    # initial sum value to get moving average calculation
        count = 1
        im =[]
        vid = input ("video path:") #path where the video is located
        base_time = input("Baseline time: ") #time for baseline data
        base_time = int(base_time)
        
        vidcap = cv.VideoCapture(vid)
        success, image = vidcap.read()
        fps = vidcap.get(cv.CAP_PROP_FPS)
        print(fps) # it is fixed for a certain camera
        path = input(" new path where to put images: ") #stores the images here
        os.mkdir(path)#make a new folder to store
        img_path = path + "/%d.png"
        while success:
          pat = img_path % count  
          cv.imwrite(pat, image)
          success, image = vidcap.read()
          print('Saved image ', count)
          count += 1
          
        for n in range(1,15,1):      
            b = f'{path}{ "/" }{n}{".png"}'
            im.append(b) ## creates the list of paths to images
        
        with Pool(5) as p:
           cv.destroyAllWindows()
           c = p.map(circle_detection, im)
           
    finally:
        print("Done detecting")
##    finish = time.perf_counter()

    x = [lis[0] for lis in c]
    y = [lis[1] for lis in c]
    x_pix = [lis[2] for lis in c]
    y_pix = [lis[3] for lis in c]
##     = list(np.array(radius)*np.array(radius)*math.pi)  ## to find the area
##    print(x,y,x_pix,y_pix)
    n = len(y)
    y_avg = y

    ## data processing
    y = eliminate_outliers(y,33)  # eliminating the outliers
    y_pix = eliminate_outliers(y_pix,10)
    x_pix = eliminate_outliers(x_pix,10)
    y_avg = moving_avg(y,w)

    ## normalization
    base_data = [y[i] for i in range (len(y)) if float(x[i]) < base_time]
    base_dia = statistics.mean(base_data)
    normal_data = [(y[i]/base_dia)*(100) for i in range (len(y))]
    np.savetxt("y_liane_1.csv",np.transpose([x,y,x_pix,y_pix,normal_data]),fmt="%f",
               header='Time(s),Radius(Pixel),center(x),center(y),normal_data(%)',delimiter=",")
            
##  print(x,y,y_avg,"x co-rdinates are:" ,x_pix,"y coordinates are: ", y_pix)            
##   print(y,y_avg)   
    
## Ploting details
    plt.ylim(10, 320)
    plt.ylabel("Pupil Diameter (Pixels)")
    plt.xlabel("time (t)")
    plt.grid()

    x1 = plt.plot(x,y,'g')
    x2 = plt.plot(x,normal_data,'red')
    plt.legend(["Pupil Diameter", "Normal Pupil Diameter"], loc ="lower right")
    
##    x3 = plt.plot(x,y_avg,'red')
    plt.show()
    
