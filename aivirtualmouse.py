	import cv2
import mediapipe as mp
import random
import util
import pyautogui
from pynput.mouse import  Button,Controller
mouse =Controller()

#to get the height and width of screen we use pyautogui
screen_width,screen_height = pyautogui.size()
mouse = Controller()
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)
#finding finger tip
def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark [mpHands.HandLandmark.INDEX_FINGER_TIP]

    return None
# mouse movement function
def move_mouse(index_fingure_tip):
    if index_fingure_tip is not None:
        x = int(index_fingure_tip.x * screen_width)
        y = int(index_fingure_tip.y * screen_height)
        pyautogui.moveTo(x,y)

#checking for Left click  i.e index is bent or not and thumb is open and middle finger is open
def is_left_click(landmarks_list,thumb_index_dist):
    return (util.get_angle(landmarks_list[5],landmarks_list[6],landmarks_list[8])<50 and
            util.get_angle(landmarks_list[9],landmarks_list[10],landmarks_list[12])>90 and
            thumb_index_dist > 50
            )
#checking for right click
def is_right_click(landmarks_list,thumb_index_dist):
    return (util.get_angle(landmarks_list[9],landmarks_list[10],landmarks_list[12])<50 and
            util.get_angle(landmarks_list[5],landmarks_list[6],landmarks_list[8])>90 and
            thumb_index_dist > 50
            )

#checking for double click
def is_double_click(landmarks_list,thumb_index_dist):
    return (util.get_angle(landmarks_list[5],landmarks_list[6],landmarks_list[8])<50 and
            util.get_angle(landmarks_list[9],landmarks_list[10],landmarks_list[12])<50 and
            thumb_index_dist > 50
            )

#checking for screenshotqq
def is_screenshot_click(landmarks_list,thumb_index_dist):
    return (util.get_angle(landmarks_list[5],landmarks_list[6],landmarks_list[8])<50 and
            util.get_angle(landmarks_list[9],landmarks_list[10],landmarks_list[12])<90 and
            thumb_index_dist < 50
            )
#this function detect the gesture in frame
def detect_gesture(frame,landmarks_list,processed):
    #checking for all landmarks if we get all landmarks then only proceed
    if len(landmarks_list)>=21:
        # we check index finger tip for moving the mouse
        index_fingerq_tip = find_finger_tip(processed)

        # checking thumb is closed or opend by checking the distance between tip of thumb[4] and base of index [5]
        thumb_index_dist = util.get_distance([landmarks_list[4],landmarks_list[5]])
        #<50 thumb is bent and index finger angle position(landmark 5,6,8,)
        if thumb_index_dist < 50 and  util.get_angle(landmarks_list[5],landmarks_list[6],landmarks_list[8])>90:

            move_mouse(index_fingerq_tip);

#Left click
        elif is_left_click(landmarks_list,thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#Right click
        elif is_right_click(landmarks_list,thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#double click
        elif is_double_click(landmarks_list,thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#screenshot
        elif is_screenshot_click(landmarks_list,thumb_index_dist):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)




#This function is for open a camera frame and capture a Video
def main():
    cap = cv2.VideoCapture(0) #here we have one camera so '0' it can range according to no.of camera
    draw = mp.solutions.drawing_utils  #mediapipe helps to draw landmarks

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break
            frame = cv2.flip(frame,1)
            frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB) # it returns the hand landmarks

            landmarks_list = [] # list to store the landmarks

            if processed.multi_hand_landmarks: # if takes landmarks from multi hands
                hand_landmarks = processed.multi_hand_landmarks[0] # then take only one of the hands landmakrs
                draw.draw_landmarks(frame,hand_landmarks, mpHands.HAND_CONNECTIONS) # it draw the hand marks so we can see it

            #loop for landmarks to add it in list
                for lm in hand_landmarks.landmark:
                   landmarks_list.append((lm.x,lm.y))

            detect_gesture(frame,landmarks_list,processed)

            cv2.imshow('Frame',frame)#shows the frame which is captured
            #after waiting for 1sec if we press q then it will close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

	Util Function for distance and angle measuring
import numpy as np

def get_angle(a,b,c):
    #calculating angle between finger
    radians = np.arctan2(c[1]-b[1],a[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(np.degrees(radians)) # converting radian into degree
    return angle

def get_distance(landmark_list):
    #if there are more than 2 landmark then only proceed otherwise return
    if len(landmark_list)<2:
        return

    (x1, y1),(x2,y2) = landmark_list[0],landmark_list[1]
    L = np.hypot(x2 - x1, y2 -y1)
    return  np.interp(L,[0,1],[0,1000])
