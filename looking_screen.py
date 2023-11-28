import numpy as np
import cv2
import dlib
import sys

model = "shape_predictor_68_face_landmarks.dat"             #pretrained model for facial landmark detection
detector = dlib.get_frontal_face_detector()                 #returns a face detector object ( uses dlib's pretrained face-detector )
predictor = dlib.shape_predictor(model)                     #returns a predictor object ( basically a set of coordinates of the 68 landmarks detected)

left = [36, 37, 38, 39, 40, 41]                             #facial landmarks for left eye
right = [42, 43, 44, 45, 46, 47]                            #facial landmarks for right eye

vid_cap = cv2.VideoCapture(0)                               #capture video from webcam     
ret, frame = vid_cap.read()
thresh = frame.copy()

kernel = np.ones((9, 9), np.uint8)                          #9x9 array filled with 1s, for dilation

#defining functions to get the position of pupils 
def eye_on_mask(mask, side):
    points = [coords[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right = False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        return cx, cy
    except:
        return 0,0

#defining function to get threshold value from trackbar
def do_nothing():
    pass
def get_threshold():
    return 44

#defining fucntion to get distance between 2 points 

def get_dist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

# time_delay = 3
calibrated = False

ret, frame = vid_cap.read()

while(True):

    ret, frame = vid_cap.read()                             #read returns a bool (for success) and the frame read from the video capture
    thresh = frame.copy()                                   #copy the frame to thresh

    if ret == True:
        pass
    else:
        print("Error reading frame")
        sys.exit()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          #predictor works on grayscale images
    rects = detector(gray, 1)                               #detect faces in the grayscale image, 1 is the number of times to upsample the image (to detect faces at different distances)
    
    if(rects.__len__() == 0):
        cv2.putText(frame, "Not looking on screen", (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0,0,255), 2)
    
    for rect in rects:                                      #for each face detected

        # convert coordinates into a 2d numpy array 

        shape = predictor(gray, rect)                       #predictor returns a set of coordinates of the 68 landmarks detected, in the rect detected 
        coords = np.zeros((68, 2), dtype="int")             #create an array of zeros ((0,0), (0,0), (0,0).... 68 times))
        for i in range (0,68):
            coords[i] = (shape.part(i).x, shape.part(i).y)  #fill the array with the coordinates of the 68 landmarks detected   

        # draw facial landmarks on face and on pupil

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)    #gets height and width of frame to make grayscale copy of image
        mask = eye_on_mask(mask, left)                      #fills the left eye with white color in the mask
        mask = eye_on_mask(mask, right)                     #fills the right eye with white color in the mask
        mask = cv2.dilate(mask, kernel, 5)                  #dilate the mask to join the gaps in the thresholded image of eyes
        eyes = cv2.bitwise_and(frame, frame, mask=mask)     #bitwise and operation between the frame and the mask, to get the eyes only
        mask = (eyes == [0, 0, 0]).all(axis=2)              #check if all the pixels in the eyes are black
        eyes[mask] = [255,255,255]
        mid = (coords[42][0] + coords[39][0]) // 2          #mid point between the eyes
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = get_threshold()        #threshold value for the thresholded image of eyes
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)     #converts image into binary image based on a threshold value
        thresh = cv2.erode(thresh, None, iterations=2)      #erode the thresholded image to remove noise
        thresh = cv2.dilate(thresh, None, iterations=4)     #dilate the thresholded image to join the gaps in the thresholded image of eyes
        thresh = cv2.medianBlur(thresh, 3)                  #median blur the thresholded image to remove noise
        thresh = cv2.bitwise_not(thresh)                    #invert the thresholded image

        (leftx, lefty) = contouring(thresh[:, 0:mid], mid, frame)    #get the coordinates of the left pupil
        (rightx, righty) = contouring(thresh[:, mid:], mid, frame, True)

        cv2.circle(frame, (leftx, lefty), 4, (0,0,255), 2)           #draw a circle at the left pupil
        cv2.circle(frame, (rightx, righty), 4, (0,0,255), 2)         #draw a circle at the right pupil

        for (x,y) in coords:
            cv2.circle(frame, (x,y), 1, (0,255,0), -1)

        
        left_center_x = 0
        left_center_y = 0
        right_center_x = 0
        right_center_y = 0
        for j in left:
            left_center_x += coords[j][0]
        left_center_x /= 6
        for j in left:
            left_center_y += coords[j][1]
        left_center_y /= 6
        for j in right:
            right_center_x += coords[j][0]
        right_center_x /= 6
        for j in right:
            right_center_y += coords[j][1]
        right_center_y /= 6


        center_dist_left = get_dist(leftx, lefty, left_center_x, left_center_y)
        center_dist_right = get_dist(rightx, righty, right_center_x, right_center_y)

        # time_delay -= 1
        # if(time_delay <= 0):
        #     time_delay = 3
        #     print(get_dist(coords[37][0], coords[37][1], coords[41][0], coords[41][1]))
        #     print(get_dist(coords[38][0], coords[38][1], coords[40][0], coords[40][1]))
        #     print("----------------")

        #     print(get_dist(coords[43][0], coords[43][1], coords[47][0], coords[47][1]))
        #     print(get_dist(coords[44][0], coords[44][1], coords[46][0], coords[46][1]))
        #     print("----------------")
        #     print("----------------")
    
        # if( (min(get_dist(coords[37][0], coords[37][1], coords[41][0], coords[41][1]), get_dist(coords[38][0], coords[38][1], coords[40][0], coords[40][1])) <= 7) & (min(get_dist(coords[43][0], coords[43][1], coords[47][0], coords[47][1]), get_dist(coords[44][0], coords[44][1], coords[46][0], coords[46][1])) <= 7)):
        #     cv2.putText(frame, "Not looking on screen", (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0,0,255), 2q)   #eyes closed

        # else:

        if(max(center_dist_right, center_dist_left) <= 6.7):
            cv2.putText(frame, "Looking on screen", (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0,255,0), 2)
        else:
            cv2.putText(frame, "Not looking on screen", (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0,0,255), 2)


    # show everything on the window
    cv2.imshow("looking on screen",frame)

    if( cv2.waitKey(1) == ord('q') ):
        break

vid_cap.release()                                           #release the video capture
cv2.destroyAllWindows()                                     #destroy all windows