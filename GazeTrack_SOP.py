import numpy as np
import cv2
import dlib
import sys
import pygame
import time
import threading

coords = np.zeros((68, 2), dtype="int")                     #create an array of zeros ((0,0), (0,0), (0,0).... 68 times))

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

#defining function to get threshold value

def get_threshold():
    return 80

#defining fucntion to get distance between 2 points 

def get_dist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def calibration_phase():

    global top_left_x, top_left_y, top_right_x, top_right_y, bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y
    global interim_top_left, interim_top_right, interim_bottom_left, interim_bottom_right
    global final_x1, final_x2, final_y1, final_y2, width_for_pupil, height_for_pupil
    global leftx, lefty, rightx, righty
    global calibration_for_corner, calibration_complete
    global quit_flag
    
    time.sleep(10)                  # delay so that camera opens up first
    
    #initialise pygame window
    pygame.init()
    area = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)    #initialise pygame window
    x_screen, y_screen = area.get_size()       #get size of pygame window
    # fill with black color
    area.fill((0,0,0))

    #area on screen is roughly 1510 x 840
    # draw circles at 4 corners and centre of screen

    # top left

    pygame.draw.circle(area, (255, 255, 255), (20, 20), 20, 2)         
    area.blit(pygame.font.SysFont('Arial', 50).render('Look at the top left corner of the screen ', True, (255, 255, 255)), (440, 390))
    pygame.display.flip()

    time.sleep(1)
    calibration_for_corner = 0     #flag for top left corner
    time.sleep(4)
    calibration_for_corner = -1    #reset flag

    # top right

    area.fill((0,0,0))
    pygame.draw.circle(area, (255, 255, 255), (x_screen-20, 20), 20, 2)       
    area.blit(pygame.font.SysFont('Arial', 50).render('Look at the top right corner of the screen ', True, (255, 255, 255)), (440, 390))
    pygame.display.flip()

    time.sleep(1)
    calibration_for_corner = 1     #flag for top right corner
    time.sleep(4)
    calibration_for_corner = -1    #reset flag

    #bottom left

    area.fill((0,0,0))
    pygame.draw.circle(area, (255, 255, 255), (20, y_screen-20), 20, 2)         
    area.blit(pygame.font.SysFont('Arial', 50).render('Look at the bottom left corner of the screen ', True, (255, 255, 255)), (440, 390))
    pygame.display.flip()
    
    time.sleep(1)
    calibration_for_corner = 2     #flag for bottom left corner
    time.sleep(4)
    calibration_for_corner = -1    #reset flag
    
    #bottom right

    area.fill((0,0,0))
    pygame.draw.circle(area, (255, 255, 255), (x_screen-20, y_screen-20), 20, 2)
    area.blit(pygame.font.SysFont('Arial', 50).render('Look at the bottom right corner of the screen ', True, (255, 255, 255)), (440, 390))
    pygame.display.flip()

    time.sleep(1)
    calibration_for_corner = 3     # flag for bottom right corner
    time.sleep(4)
    calibration_for_corner = -1    # reset flag

    calibration_complete = True    

    time.sleep(1)                  # delay for calculating final x1,x2,y1,y2

    # running the part to display a circle on screen and move it according to the eye movement  

    if(calibration_complete == False):
        run = True
        averaging_list_x = [-1,-1,-1]
        averaging_list_y = [-1,-1,-1]
        averaging_list_x_index = 0
        averaging_list_y_index = 0
    else:
        print("gadbad")
    while run:
        pygame.time.delay(100)      # 0.1 seconds

        if(averaging_list_x[2] == -1 and averaging_list_y[2] == -1):
            x_to_display = (x_screen/(width_for_pupil*(1.5))) * (( abs(leftx - final_x1[0]) + abs(rightx - final_x1[1]) ) / 2 ) 
            y_to_display = (y_screen/(height_for_pupil*(3.5))) * (( abs(lefty - final_y1[0]) + abs(righty - final_y1[1]) ) / 2 )
            averaging_list_x[averaging_list_x_index] = x_to_display
            averaging_list_y[averaging_list_y_index] = y_to_display
            averaging_list_x_index += 1
            averaging_list_y_index += 1
            if(averaging_list_x_index == 3):
                averaging_list_x_index = 0
            if(averaging_list_y_index == 3):
                averaging_list_y_index = 0
        else:
            x_to_display = (averaging_list_x[0] + averaging_list_x[1] + averaging_list_x[2]) / 3
            y_to_display = (averaging_list_y[0] + averaging_list_y[1] + averaging_list_y[2]) / 3

            new_x = (x_screen/(width_for_pupil*(1.5))) * (( abs(leftx - final_x1[0]) + abs(rightx - final_x1[1]) ) / 2 )
            new_y = (y_screen/(height_for_pupil*(3.5))) * (( abs(lefty - final_y1[0]) + abs(righty - final_y1[1]) ) / 2 )

            for i in range (0,2):
                averaging_list_x[i] = averaging_list_x[i+1]
                averaging_list_y[i] = averaging_list_y[i+1]

            averaging_list_x[2] = new_x
            averaging_list_y[2] = new_y



        area.fill((0,0,0))           # fill the window with black color
        pygame.draw.circle(area, (255, 255, 255), (int(x_to_display), int(y_to_display)), 210)    # draw a circle at the final x1,y1
        pygame.display.flip()        # update the display

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                quit_flag = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quit_flag = True
                    run = False
    
    pygame.quit()                   # quit pygame



def main_stuff():

    global top_left_x, top_left_y, top_right_x, top_right_y, bottom_left_x, bottom_left_y, bottom_right_x, bottom_right_y
    global interim_top_left, interim_top_right, interim_bottom_left, interim_bottom_right
    global final_x1, final_x2, final_y1, final_y2, width_for_pupil, height_for_pupil
    global leftx, lefty, rightx, righty
    global calibration_for_corner, calibration_complete
    global quit_flag

    model = "shape_predictor_68_face_landmarks.dat"             #pretrained model for facial landmark detection
    detector = dlib.get_frontal_face_detector()                 #returns a face detector object ( uses dlib's pretrained face-detector )
    predictor = dlib.shape_predictor(model)                     #returns a predictor object ( basically a set of coordinates of the 68 landmarks detected)

    left = [36, 37, 38, 39, 40, 41]                             #facial landmarks for left eye
    right = [42, 43, 44, 45, 46, 47]                            #facial landmarks for right eye

    vid_cap = cv2.VideoCapture(0)                               #capture video from webcam     
    kernel = np.ones((9, 9), np.uint8)                          #9x9 array filled with 1s, for dilation

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
            threshold = get_threshold()                                     #threshold value for the thresholded image of eyes
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)   #converts image into binary image based on a threshold value
            thresh = cv2.erode(thresh, None, iterations=2)      #erode the thresholded image to remove noise
            thresh = cv2.dilate(thresh, None, iterations=4)     #dilate the thresholded image to join the gaps in the thresholded image of eyes
            thresh = cv2.medianBlur(thresh, 3)                  #median blur the thresholded image to remove noise
            thresh = cv2.bitwise_not(thresh)                    #invert the thresholded image

            (leftx, lefty) = contouring(thresh[:, 0:mid], mid, frame)     #get the coordinates of the left pupil
            (rightx, righty) = contouring(thresh[:, mid:], mid, frame, True)

            cv2.circle(frame, (leftx, lefty), 4, (0,0,255), 2)           #draw a circle at the left pupil
            cv2.circle(frame, (rightx, righty), 4, (0,0,255), 2)         #draw a circle at the right pupil

            for (x,y) in coords:
                cv2.circle(frame, (x,y), 1, (0,255,0), -1)

            if(calibration_for_corner == 0):
                top_left_x = np.append(top_left_x, np.array([[leftx, rightx]]), axis=0)
                top_left_y = np.append(top_left_y, np.array([[lefty, righty]]), axis=0)

            if(calibration_for_corner == 1):
                top_right_x = np.append(top_right_x, np.array([[leftx, rightx]]), axis=0)
                top_right_y = np.append(top_right_y, np.array([[lefty, righty]]), axis=0)

            if(calibration_for_corner == 2):
                bottom_left_x = np.append(bottom_left_x, np.array([[leftx, rightx]]), axis=0)
                bottom_left_y = np.append(bottom_left_y, np.array([[lefty, righty]]), axis=0)

            if(calibration_for_corner == 3):
                bottom_right_x = np.append(bottom_right_x, np.array([[leftx, rightx]]), axis=0)
                bottom_right_y = np.append(bottom_right_y, np.array([[lefty, righty]]), axis=0)

            if(calibration_complete == True):
                interim_top_left[0][0] = np.mean(top_left_x, axis=0)[0]       #left eye x
                interim_top_left[1][0] = np.mean(top_left_x, axis=0)[1]       #right eye x
                interim_top_left[0][1] = np.mean(top_left_y, axis=0)[0]       #left eye y
                interim_top_left[1][1] = np.mean(top_left_y, axis=0)[1]       #right eye y

                interim_top_right[0][0] = np.mean(top_right_x, axis=0)[0]
                interim_top_right[1][0] = np.mean(top_right_x, axis=0)[1]
                interim_top_right[0][1] = np.mean(top_right_y, axis=0)[0]
                interim_top_right[1][1] = np.mean(top_right_y, axis=0)[1]

                interim_bottom_left[0][0] = np.mean(bottom_left_x, axis=0)[0]
                interim_bottom_left[1][0] = np.mean(bottom_left_x, axis=0)[1]
                interim_bottom_left[0][1] = np.mean(bottom_left_y, axis=0)[0]
                interim_bottom_left[1][1] = np.mean(bottom_left_y, axis=0)[1]

                interim_bottom_right[0][0] = np.mean(bottom_right_x, axis=0)[0]
                interim_bottom_right[1][0] = np.mean(bottom_right_x, axis=0)[1]
                interim_bottom_right[0][1] = np.mean(bottom_right_y, axis=0)[0]
                interim_bottom_right[1][1] = np.mean(bottom_right_y, axis=0)[1]

                final_x1 = [(interim_top_left[0][0] + interim_bottom_left[0][0]) / 2, (interim_top_left[1][0] + interim_bottom_left[1][0]) / 2]
                final_x2 = [(interim_top_right[0][0] + interim_bottom_right[0][0]) / 2, (interim_top_right[1][0] + interim_bottom_right[1][0]) / 2]
                final_y1 = [(interim_top_left[0][1] + interim_top_right[0][1]) / 2, (interim_top_left[1][1] + interim_top_right[1][1]) / 2]
                final_y2 = [(interim_bottom_left[0][1] + interim_bottom_right[0][1]) / 2, (interim_bottom_left[1][1] + interim_bottom_right[1][1]) / 2]

                width_for_pupil = (abs(final_x2[0]-final_x1[0]) + abs(final_x2[1]-final_x1[1]))/2
                height_for_pupil = (abs(final_y2[0]-final_y1[0]) + abs(final_y2[1]-final_y1[1]))/2

                calibration_complete = False

        if( quit_flag == True ):                       #wait for q to be pressed to quit
            break

    vid_cap.release()                                           #release the video capture
    cv2.destroyAllWindows()                                     #destroy all windows



if __name__ == '__main__':
    
    p1 = threading.Thread(target = calibration_phase, args = ())
    p2 = threading.Thread(target = main_stuff, args = ())

    # defining arrays and floats to store coordinates for calibration;

    top_left_x = np.empty((0, 2), dtype=float)
    top_left_y = np.empty((0, 2), dtype=float)

    top_right_x = np.empty((0, 2), dtype=float)
    top_right_y = np.empty((0, 2), dtype=float)

    bottom_left_x = np.empty((0, 2), dtype=float)
    bottom_left_y = np.empty((0, 2), dtype=float)

    bottom_right_x = np.empty((0, 2), dtype=float)
    bottom_right_y = np.empty((0, 2), dtype=float)

    interim_top_left = [[0,0], [0,0]]         # 0 for left x,y , 1 for right x,y
    interim_top_right = [[0,0], [0,0]]
    interim_bottom_left = [[0,0], [0,0]]
    interim_bottom_right = [[0,0], [0,0]]

    final_x1 = [0,0]                          # 0 for left, 1 for right
    final_x2 = [0,0]
    final_y1 = [0,0]
    final_y2 = [0,0]
    width_for_pupil = 0
    height_for_pupil = 0

    leftx = 0
    lefty = 0
    rightx = 0
    righty = 0

    quit_flag = False                       # flag for quitting the program

    # defining flags to get coordinates for calibration

    calibration_for_corner = -1         # 0 = top left, 1 = top right, 2 = bottom left, 3 = bottom right
    calibration_complete = False        # flag for calibration complete


    p1.start()
    p2.start()

    p1.join()
    p2.join()
