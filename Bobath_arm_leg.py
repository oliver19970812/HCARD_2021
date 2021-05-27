# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import threading
import matplotlib.pyplot as plt
import serial
import math
import pygame


def instruction():
    
    size = (1500,900)
    t_wait = 5000 #wait for 5 seconds
    
    ins_1 = cv2.imread('instruction/ins_1.png')
    ins_1 = cv2.resize(ins_1, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", ins_1)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()
    
    ins_2 = cv2.imread('instruction/ins_2.png')
    ins_2 = cv2.resize(ins_2, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", ins_2)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()

    ins_3 = cv2.imread('instruction/ins_3.png')
    ins_3 = cv2.resize(ins_3, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", ins_3)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()
    
    ins_4 = cv2.imread('instruction/ins_4.png')
    ins_4 = cv2.resize(ins_4, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", ins_4)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()
    
    ins_5 = cv2.imread('instruction/ins_5.png')
    ins_5 = cv2.resize(ins_5, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", ins_5)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()
    
    ins_6 = cv2.imread('instruction/ins_6.png')
    ins_6 = cv2.resize(ins_6, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", ins_6)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()

    ins_7 = cv2.imread('instruction/ins_7.png')
    ins_7 = cv2.resize(ins_7, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", ins_7)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()

    

def params(args):
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["hand"] = True

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item
    return params

def compare (x,y):
    a = x[1]
    b = y[1]
    if a > b:
        return x
    else:
        return y

def find_arm_min(a,b,c,d,e):
    l = []
    l.append(a)
    l.append(b)
    l.append(c)
    l.append(d)
    l.append(e)
    l.sort()
    return l[4]

def find_leg_min(a,b,c,d):
    l = []
    l.append(a)
    l.append(b)
    l.append(c)
    l.append(d)
    l.sort()
    return l[3]

def keypoints_extract(Body_Keypoints, Right_hand_keypoints, Left_hand_keypoints):
    Body_Keypoints = np.delete(Body_Keypoints, -1, axis=2)
    Right_hand_keypoints = np.delete(Right_hand_keypoints, -1, axis=2)
    Left_hand_keypoints = np.delete(Left_hand_keypoints, -1, axis=2)
    #print(Body_Keypoints)
    # print(Left_hand_keypoints)
    # print(Right_hand_keypoints)


    nose = list(Body_Keypoints[0,0])

    left_eye = Body_Keypoints[0, 16]
    right_eye = Body_Keypoints[0, 15]
    eye = list(compare(left_eye,right_eye))
    # print(eye)

    left_ear = Body_Keypoints[0, 18]
    right_ear = Body_Keypoints[0, 17]
    ear = list(compare(left_ear, right_ear))

    left_shoulder = Body_Keypoints[0, 5]
    right_shoulder = Body_Keypoints[0, 2]
    shoulder = list(compare(left_shoulder, right_shoulder))

    left_elbow = Body_Keypoints[0, 6]
    right_elbow = Body_Keypoints[0, 3]
    elbow = list(compare(left_elbow, right_elbow))

    left_wrist = Body_Keypoints[0, 7]
    right_wrist = Body_Keypoints[0, 4]
    wrist = list(compare(left_wrist, right_wrist))

    left_hand5 = Right_hand_keypoints[0, 5]
    right_hand5 = Right_hand_keypoints[0, 5]
    hand5 = list(compare(left_hand5, right_hand5))

    left_hand17 = Left_hand_keypoints[0, 17]
    right_hand17 = Right_hand_keypoints[0, 17]
    hand17 = list(compare(left_hand17, right_hand17))

    arm_keypoints = [nose, eye, ear, shoulder, elbow, wrist, hand5, hand17]
    #print(keypoints)
    return arm_keypoints

def keypoints_extract_leg(Body_Keypoints):
    Body_Keypoints = np.delete(Body_Keypoints, -1, axis=2)
    
    left_shoulder = Body_Keypoints[0, 5]
    right_shoulder = Body_Keypoints[0, 2]
    shoulder = list(compare(left_shoulder, right_shoulder))

    left_hip = Body_Keypoints[0, 12]
    right_hip = Body_Keypoints[0, 9]
    hip = list(compare(left_hip, right_hip))
    
    left_knee = Body_Keypoints[0, 13]
    right_knee = Body_Keypoints[0, 10]
    knee = list(compare(left_knee, right_knee))
    
    left_ankle = Body_Keypoints[0, 14]
    right_ankle = Body_Keypoints[0, 11]
    ankle = list(compare(left_ankle, right_ankle))
    
    left_little_toe = Body_Keypoints[0, 20]
    right_little_toe = Body_Keypoints[0, 23]
    little_toe = list(compare(left_little_toe, right_little_toe))
    
    left_heel = Body_Keypoints[0, 21]
    right_heel = Body_Keypoints[0, 24]
    heel = list(compare(left_heel,right_heel))
    
    leg_keypoints = [shoulder, hip, knee, ankle, little_toe, heel]
    
    return leg_keypoints


def eliminate_zero(keypoint1, keypoint2):
    keypoint_1 = np.array([i[0] for i in keypoint1])
    # print(keypoint1)
    keypoint_2 = np.array([i[0] for i in keypoint2])
    # print(keypoint2)
    index_1 = list(np.where(keypoint_1 == 0)[0]) # find which index = 0
    index_2 = list(np.where(keypoint_2 == 0)[0])
    # print(index_1)
    # print(index_2)
    index = sorted(set(index_1+index_2)) #排序+去重
    def zeros(index):
        nonzero_index = [0, 1, 2, 3, 4, 5, 6, 7]
        intersection = list(set(nonzero_index).intersection(set(index)))
        for i in intersection:
            if i in nonzero_index:
                nonzero_index.remove(i)
        return nonzero_index
    nonzero_index = zeros(index)
    nonzero_keypoints1 = []
    nonzero_keypoints2 = []
    for i in nonzero_index:
        nonzero_keypoints1.append(keypoint1[i])
        nonzero_keypoints2.append(keypoint2[i])
    #print('nonezero Keypoints:')
    #print(nonzero_keypoints1)
    #print(nonzero_keypoints2)
    return nonzero_keypoints1, nonzero_keypoints2

def eliminate_zero_leg(keypoint1, keypoint2):
    keypoint_1 = np.array([i[0] for i in keypoint1])
    # print(keypoint1)
    keypoint_2 = np.array([i[0] for i in keypoint2])
    # print(keypoint2)
    index_1 = list(np.where(keypoint_1 == 0)[0]) # find which index = 0
    index_2 = list(np.where(keypoint_2 == 0)[0])
    # print(index_1)
    # print(index_2)
    index = sorted(set(index_1+index_2)) #排序+去重
    def zeros(index):
        nonzero_index = [0, 1, 2, 3, 4, 5]
        intersection = list(set(nonzero_index).intersection(set(index)))
        for i in intersection:
            if i in nonzero_index:
                nonzero_index.remove(i)
        return nonzero_index
    nonzero_index = zeros(index)
    nonzero_keypoints1 = []
    nonzero_keypoints2 = []
    for i in nonzero_index:
        nonzero_keypoints1.append(keypoint1[i])
        nonzero_keypoints2.append(keypoint2[i])
    #print('nonezero Keypoints:')
    #print(nonzero_keypoints1)
    #print(nonzero_keypoints2)
    return nonzero_keypoints1, nonzero_keypoints2

def score(data):
    _range = np.max(data) - np.min(data)
    scores = 100-((data - np.min(data)) / _range)*60
    return np.mean(scores)

def arduino_control():
    global realtime_recg_j
    global datafromUser
    
    pygame.mixer.init()
    bad,good=0,0       
    recordSpeed=[]
    
    ########需要信号开始####
    ########################
    
    
    
    if datafromUser == '1':
        pygame.mixer.music.load("hotline_bling.mp3")
        print("Hotline Bling")
    elif datafromUser == '2':
        pygame.mixer.music.load("Nohands.mp3")
        print("Nohands")
    elif datafromUser == '3':
        pygame.mixer.music.load("bad guy.mp3")
        print("Bad Guy")
    elif datafromUser == '4':
        pygame.mixer.music.load("Attention.mp3")
        print("Attention")
    elif datafromUser == '5':
        pygame.mixer.music.load("DDU-DU DDU-DU.mp3")
        print("DDU-DU DDU-DU")
    elif datafromUser == '6':
        pygame.mixer.music.load("Fire.mp3")
        print("Fire")
    elif datafromUser == '7':
        pygame.mixer.music.load("River.mp3")
        print("River")
    elif datafromUser == '8':
        pygame.mixer.music.load("Unstoppable.mp3")
        print("Unstoppable")
    elif datafromUser == '9':
        pygame.mixer.music.load("YOUTH.mp3")
        print("YOUTH")
    elif datafromUser == '10':
        pygame.mixer.music.load("There For You.mp3")
        print("There For You")
    else:
        pygame.mixer.music.load("Lemon.mp3")
        print("Lemon")
    
    #cv2.waitKey(10000)
    
    ser = serial.Serial('COM6', 9600,timeout=1)
    time.sleep(2)
    # Read and record the data
    #i=0        
    
    
    
    pygame.mixer.music.play(5)#add a -1 like so: pygame.mixer.music.play(-1) and the music will repeat forever
        
    while(1):
        
        data =[]  
        b = ser.readline()         # read a byte string
        string_n = b.decode() # decode byte string into Unicode  
        string = string_n.rstrip() # remove \n and \r
        string= string.strip(' ')
        #print('times:',i)
        #print(string)
        
        if not string:
            continue  #print("The string is empty")
        else:
            for s in string.split(' '):
                s=float(s)
                data.append(s)  
                if len(data)%3==0:
                    #print(data)
                    recordSpeed.append(data)
                    if (abs(data[0])<5 and abs(data[1])<5 and abs(data[2])<5)\
                        or (abs(data[0])>100 or abs(data[1])>100 or abs(data[2])>100 ):
                        pygame.mixer.music.pause() #暂停
                        time.sleep(0.6)
                        pygame.mixer.music.unpause()
                        bad+=1
                    else:
                        good+=1
        if realtime_recg_j > 440:
            break
        # k = cv2.waitKey(2)
        # if (k & 0xff == ord('q')):
        #     break
        # if i > 50:
        #     break
        # i+=1
       
    ser.close()#close serial port
    
    
    pygame.mixer.music.stop()
    #calculate result
    record=np.array(recordSpeed)
    row,col=np.shape(record)
    
    for i in range(row):
        if (abs(record[i][0])<5 and abs(record[i][1])<5 and abs(record[i][2])<5)\
        or (abs(record[i][0])>100 or abs(record[i][1])>100 or abs(record[i][2])>100 ):
            bad+=1
        else:
            good+=1
            
    score=good/(good+bad)*100
    print('Your speed control score is:', score)


def Tutorial_videos():
    global flag
    #tutorial_a = cv2.VideoCapture('Bobath_Tutorial/Bobath-a.mp4')  
    print('Press q to quit')
    global realtime_recg_i
    global realtime_recg_j
    
    tutorial_a = cv2.VideoCapture('Bobath_Tutorial/Bobath_arm.mp4')  
    while(tutorial_a.isOpened()):  
        ret, frame = tutorial_a.read()  
        frame = cv2.resize(frame, (600, 1000))
        cv2.imshow('Tutorial_a', frame)  
        k = cv2.waitKey(1)  
        #q键退出
        realtime_recg_i+=1
        if realtime_recg_i > 450:
            break
        if realtime_recg_i == 300:
            text_gj = cv2.imread('instruction/text_gj.png')
            text_gj = cv2.resize(text_gj, (1500,900))
            cv2.namedWindow("GOOD JOB", 1)
            cv2.imshow("GOOD JOB", text_gj)
            cv2.waitKey(4000)
            cv2.destroyAllWindows()
        if (k & 0xff == ord('q')):  
            break  
    tutorial_a.release()  
    cv2.destroyAllWindows()

    text_transfer = cv2.imread('instruction/text_transfer.png')
    text_transfer = cv2.resize(text_transfer, (1500,900))
    cv2.namedWindow("Transfer", 1)
    cv2.imshow("Transfer", text_transfer)
    cv2.waitKey(30000)
    cv2.destroyAllWindows()
    
    flag = 1
    

    tutorial_c = cv2.VideoCapture('Bobath_Tutorial/Bobath_leg.mp4')
    while(tutorial_c.isOpened()):  
        ret, frame = tutorial_c.read()  
        frame = cv2.resize(frame, (600, 1000))
        cv2.imshow('Tutorial_c', frame)
        k = cv2.waitKey(1)
        #q键退出
        realtime_recg_j+=1
        if realtime_recg_j > 450:
            break
        if realtime_recg_j == 300:
            text_gj = cv2.imread('instruction/text_great.png')
            text_gj = cv2.resize(text_gj, (1500,900))
            cv2.namedWindow("GOOD JOB", 1)
            cv2.imshow("GOOD JOB", text_gj)
            cv2.waitKey(4000)
            cv2.destroyAllWindows()
        if (k & 0xff == ord('q')):  
            break
    tutorial_c.release()
    cv2.destroyAllWindows()
    


def error_cal(std_points, usr_points, F):
    
    ones = np.ones(len(std_points))
    std_1 = np.c_[std_points, ones.T]
    usr_1 = np.c_[usr_points, ones.T]
    i = 0
    error_sum = 0
    while i < len(std_points):
        single_e = np.abs(np.dot(np.dot(std_1[i:i+1], F), usr_1[i:i+1].T))
        error_sum += single_e
        i+=1
    
    return error_sum


def mean_square_error(std_points, usr_points, H):
    i = 0
    estimate_point = []
    ones = np.ones(len(std_points))
    std_1 = np.c_[std_points,ones.T]
    while i < len(std_points):
        estimate_point.append(np.dot(H, std_1[i].T))
        #single_e = np.abs(np.dot(np.dot(std_1[i:i+1], F), use_1[i:i+1].T))
        #error+=single_e
        i+=1
    estimate_point = np.delete(estimate_point, -1, axis=1)
    mse = ((np.float32(estimate_point) - np.float32(usr_points)) ** 2).mean(axis=0)
    return mse[0]*mse[1]

def get_degree(keypoints):
    # keypoints = [nose, eye, ear, shoulder, elbow, wrist, hand5, hand17]

    point_1 = keypoints[3] # shoulder
    point_2 = keypoints[4] # elbow
    point_3 = keypoints[5] # wrist
    point_4 = keypoints[2] # nose
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1])) # w-e
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1])) #s-w
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1])) #s-e
    d = math.sqrt(
        (point_4[0] - point_2[0]) * (point_4[0] - point_2[0]) + (point_4[1] - point_2[1]) * (point_4[1] - point_2[1])) #n-e
    e = math.sqrt(
        (point_1[0] - point_4[0]) * (point_1[0] - point_4[0]) + (point_1[1] - point_4[1]) * (point_1[1] - point_4[1])) #n-s
    if a==0 or c ==0:
        B = 0
        #print('Cannot detect the angle')
    else:
    #A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
        B = 180 - math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    if c==0 or e==0:
        D = 0
        #print('Cannot detect the angle')
    else:
    #C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
        D = 180 - math.degrees(math.acos((d * d - c * c - e * e) / (-2 * c * e)))
    #print(B, D)
    return B,D

def get_leg_degree(leg_keypoints):
    # leg_keypoints = [shoulder, hip, knee, ankle, little_toe, heel]

    point_1 = leg_keypoints[1] # hip
    point_2 = leg_keypoints[2] # knee
    point_3 = leg_keypoints[3] # ankle
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1])) #a-k
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1])) #h-a
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1])) #h-k

    if a==0 or c ==0:
        B = 0
        #print('Cannot detect the angle')
    else:
    #A = math.degrees(math.acos((a * a - b * b - c * c) / (-2 * b * c)))
        B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    return B

def set_threshold(t_list):
    
    r = 1
    t_max = max(t_list)
    t_min = min(t_list)
    t = t_min + (t_max - t_min)*r
    
    return t


def video_calib_video():
    
    global video_calib_i
    global video_calib_j
    
    tutorial_a = cv2.VideoCapture('Bobath_Tutorial/Bobath_arm.mp4')  
    print('Press q to quit')
    video_calib_i = 0
    while video_calib_i<=180:
        ret, frame = tutorial_a.read()  
        frame = cv2.resize(frame, (600, 1000))
        
        cv2.imshow('Tutorial Part 1 of 2', frame)
        
        video_calib_i+=1
        k = cv2.waitKey(2)

        if (k & 0xff == ord('q')):  
            break  
    tutorial_a.release()
    cv2.destroyAllWindows()
    
    
    tutorial_b = cv2.VideoCapture('Bobath_Tutorial/Bobath_leg.mp4')  
    print('Press q to quit')
    video_calib_j = 0
    while video_calib_j<=180:
        ret, frame = tutorial_b.read()  
        frame = cv2.resize(frame, (600, 1000))

        cv2.imshow('Tutorial Part 2 of 2', frame)
        video_calib_j+=1
        k = cv2.waitKey(2)

        if (k & 0xff == ord('q')):  
            break  
    tutorial_b.release()
    cv2.destroyAllWindows()
    return 0

def video_calib():
    
    global t_1
    global t_2
    global t_3
    global t_4
    global t_5
    
    global t_6
    global t_7
    global t_8
    global t_9
    
    global H_arm
    global H_leg
    
    global video_calib_i
    global video_calib_j
    
    arm_error_list_1 = []
    arm_error_list_2 = []
    arm_error_list_3 = []
    arm_error_list_4 = []
    arm_error_list_5 = []
    
    leg_error_list_1 = []
    leg_error_list_2 = []
    leg_error_list_3 = []
    leg_error_list_4 = []    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    
    params = dict()
    params["model_folder"] = "../../../models/"
    params["hand"] = True

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    datum_calib = op.Datum()
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while video_calib_i<=180:
        
        ret_1,frame_1=video.read()
        rows,cols,ch = frame_1.shape
        
        frame_1 = cv2.resize(frame_1, (1000, 600))
        datum_calib.cvInputData = frame_1
        opWrapper.emplaceAndPop([datum_calib])
        
        #cv2.imshow("Warm-up Part 1 of 2", datum_calib.cvOutputData)
        cv2.imshow("Warm-up Part 1 of 2", frame_1)
                
        calib_user_arm_keypoints = keypoints_extract(datum_calib.poseKeypoints, datum_calib.handKeypoints[1], datum_calib.handKeypoints[0])        
        Standard_keypoints_arm_1_counter, calib_user_arm_keypoints_counter_1 = eliminate_zero(Standard_keypoints_arm_1, calib_user_arm_keypoints) 
        Standard_keypoints_arm_2_counter, calib_user_arm_keypoints_counter_2 = eliminate_zero(Standard_keypoints_arm_2, calib_user_arm_keypoints)
        Standard_keypoints_arm_3_counter, calib_user_arm_keypoints_counter_3 = eliminate_zero(Standard_keypoints_arm_3, calib_user_arm_keypoints) 
        Standard_keypoints_arm_4_counter, calib_user_arm_keypoints_counter_4 = eliminate_zero(Standard_keypoints_arm_4, calib_user_arm_keypoints)
        Standard_keypoints_arm_5_counter, calib_user_arm_keypoints_counter_5 = eliminate_zero(Standard_keypoints_arm_5, calib_user_arm_keypoints)

        error_arm_1 = mean_square_error(Standard_keypoints_arm_1_counter, calib_user_arm_keypoints_counter_1, H_arm)
        error_arm_2 = mean_square_error(Standard_keypoints_arm_2_counter, calib_user_arm_keypoints_counter_2, H_arm)
        error_arm_3 = mean_square_error(Standard_keypoints_arm_3_counter, calib_user_arm_keypoints_counter_3, H_arm)
        error_arm_4 = mean_square_error(Standard_keypoints_arm_4_counter, calib_user_arm_keypoints_counter_4, H_arm)
        error_arm_5 = mean_square_error(Standard_keypoints_arm_5_counter, calib_user_arm_keypoints_counter_5, H_arm)

        arm_error_list_1.append(error_arm_1)
        arm_error_list_2.append(error_arm_2)
        arm_error_list_3.append(error_arm_3)
        arm_error_list_4.append(error_arm_4)
        arm_error_list_5.append(error_arm_5)
        
        k = cv2.waitKey(50)
        if (k & 0xff == ord('q')):  
            break
    
    while video_calib_j<=180:
        
        ret_1,frame_1=video.read()
        rows,cols,ch = frame_1.shape
        
        frame_1 = cv2.resize(frame_1, (1000, 600))
        datum_calib.cvInputData = frame_1
        opWrapper.emplaceAndPop([datum_calib])
        
        #cv2.imshow("Warm-up Part 2 of 2", datum_calib.cvOutputData)
        cv2.imshow("Warm-up Part 2 of 2", frame_1)
                
        calib_user_leg_keypoints = keypoints_extract_leg(datum_calib.poseKeypoints)        
        Standard_keypoints_leg_1_counter, calib_user_leg_keypoints_counter_1 = eliminate_zero_leg(Standard_keypoints_leg_1, calib_user_leg_keypoints) 
        Standard_keypoints_leg_2_counter, calib_user_leg_keypoints_counter_2 = eliminate_zero_leg(Standard_keypoints_leg_2, calib_user_leg_keypoints)
        Standard_keypoints_leg_3_counter, calib_user_leg_keypoints_counter_3 = eliminate_zero_leg(Standard_keypoints_leg_3, calib_user_leg_keypoints) 
        Standard_keypoints_leg_4_counter, calib_user_leg_keypoints_counter_4 = eliminate_zero_leg(Standard_keypoints_leg_4, calib_user_leg_keypoints)

        error_leg_1 = mean_square_error(Standard_keypoints_leg_1_counter, calib_user_leg_keypoints_counter_1, H_leg)
        error_leg_2 = mean_square_error(Standard_keypoints_leg_2_counter, calib_user_leg_keypoints_counter_2, H_leg)
        error_leg_3 = mean_square_error(Standard_keypoints_leg_3_counter, calib_user_leg_keypoints_counter_3, H_leg)
        error_leg_4 = mean_square_error(Standard_keypoints_leg_4_counter, calib_user_leg_keypoints_counter_4, H_leg)

        leg_error_list_1.append(error_leg_1)
        leg_error_list_2.append(error_leg_2)
        leg_error_list_3.append(error_leg_3)
        leg_error_list_4.append(error_leg_4)
        
        k = cv2.waitKey(50)
        if (k & 0xff == ord('q')):  
            break
    
    t_1 = set_threshold(arm_error_list_1)
    t_2 = set_threshold(arm_error_list_2)
    t_3 = set_threshold(arm_error_list_3)
    t_4 = set_threshold(arm_error_list_4)
    t_5 = set_threshold(arm_error_list_5)
    
    t_6 = set_threshold(leg_error_list_1)
    t_7 = set_threshold(leg_error_list_2)
    t_8 = set_threshold(leg_error_list_3)
    t_9 = set_threshold(leg_error_list_4)
    
    '''print("Threshold 1 : %d" %t_1)
    print("Threshold 2 : %d" %t_2)
    print("Threshold 3 : %d" %t_3)
    print("Threshold 4 : %d" %t_4)
    print("Threshold 5 : %d" %t_5)
    print("Threshold 6 : %d" %t_6)
    print("Threshold 7 : %d" %t_7)
    print("Threshold 8 : %d" %t_8)
    print("Threshold 9 : %d" %t_9)'''
    
    video.release()
    cv2.destroyAllWindows()

def realtime_recognition():
    
    global H_arm
    global H_leg
    global Standard_keypoints_1
    global Standard_keypoints_2
    global Standard_keypoints_3
    global Standard_keypoints_4
    global Standard_keypoints_5
    global Standard_keypoints_6
    
    global arm_error_list
    global leg_error_list
    
    global t_1
    global t_2
    global t_3
    global t_4
    global t_5
    
    global t_6
    global t_7
    global t_8
    global t_9
    
    global realtime_recg_i
    global realtime_recg_j
    
    global arm_error_list
    global leg_error_list
    
    global flag
    
    angle_1 = []
    angle_2 = []
    
    fault_count_arm = 0
    fault_count_leg = 0
    
    rate = 0.5
    t_arm = [t_1,t_2,t_3,t_4,t_5]
    t_leg = [t_6,t_7,t_8,t_9]
    #fault_arm = min(arm_error_list) + rate*(max(arm_error_list) - min (arm_error_list))
    #fault_leg = min(leg_error_list) + rate*(max(leg_error_list) - min (leg_error_list))  
    fault_arm = rate*min(t_arm)
    fault_leg = rate*min(t_leg)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    
    params = dict()
    params["model_folder"] = "../../../models/"
    params["hand"] = True

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    datum_realtime = op.Datum()
    video_realtime = cv2.VideoCapture(0,cv2.CAP_DSHOW)


    while(video_realtime.isOpened()):
        

        ret_1,frame_1=video_realtime.read()
        rows,cols,ch = frame_1.shape
        
        frame_1 = cv2.resize(frame_1, (1000, 600))
        datum_realtime.cvInputData = frame_1
        opWrapper.emplaceAndPop([datum_realtime])
        #cv2.imshow("Realtime_recognition", datum_realtime.cvOutputData)
        cv2.imshow("Realtime_recognition", frame_1)
        
        
        k = cv2.waitKey(50)
        #q键退出
        if (k & 0xff == ord('q')):  
            break
        
        realtime_user_arm_keypoints = keypoints_extract(datum_realtime.poseKeypoints, datum_realtime.handKeypoints[1], datum_realtime.handKeypoints[0])
        realtime_user_leg_keypoints = keypoints_extract_leg(datum_realtime.poseKeypoints)
       

        if k:
            #time = cv2.waitKey(50)       
            if realtime_recg_i < 450:
                angle_1.append(get_degree(realtime_user_arm_keypoints)[0])
                Standard_keypoints_arm_1_counter, realtime_user_keypoints_counter = eliminate_zero(Standard_keypoints_arm_1, realtime_user_arm_keypoints)                 
                error_arm_1 = mean_square_error(Standard_keypoints_arm_1_counter, realtime_user_keypoints_counter, H_arm)
                #print(error_start)
                Standard_keypoints_arm_2_counter, realtime_user_keypoints_counter = eliminate_zero(Standard_keypoints_arm_2, realtime_user_arm_keypoints)
                error_arm_2 = mean_square_error(Standard_keypoints_arm_2_counter, realtime_user_keypoints_counter, H_arm)
                #print(error_end)
                Standard_keypoints_arm_3_counter, realtime_user_keypoints_counter = eliminate_zero(Standard_keypoints_arm_3, realtime_user_arm_keypoints)                 
                error_arm_3 = mean_square_error(Standard_keypoints_arm_3_counter, realtime_user_keypoints_counter, H_arm)
                #print(error_start)
                Standard_keypoints_arm_4_counter, realtime_user_keypoints_counter = eliminate_zero(Standard_keypoints_arm_4, realtime_user_arm_keypoints)
                error_arm_4 = mean_square_error(Standard_keypoints_arm_4_counter, realtime_user_keypoints_counter, H_arm)
                #print(error_end)
                Standard_keypoints_arm_5_counter, realtime_user_keypoints_counter = eliminate_zero(Standard_keypoints_arm_5, realtime_user_arm_keypoints)                 
                error_arm_5 = mean_square_error(Standard_keypoints_arm_5_counter, realtime_user_keypoints_counter, H_arm)
                #print(error_start)
                arm_error = find_arm_min(error_arm_1, error_arm_2, error_arm_3, error_arm_4, error_arm_5)
                
                arm_error_list.append(arm_error)
                if arm_error > fault_arm:
                    cv2.imwrite('User_Bobath/user_bobath_arm_higherror_'+str(fault_count_arm)+'.jpg', frame_1)
                    fault_count_arm+=1
                   
            elif realtime_recg_j < 450:
                angle_2.append(get_leg_degree(realtime_user_leg_keypoints))
                Standard_keypoints_leg_1_counter, realtime_user_keypoints_counter = eliminate_zero_leg(Standard_keypoints_leg_1, realtime_user_leg_keypoints)                 
                error_leg_1 = mean_square_error(Standard_keypoints_leg_1_counter, realtime_user_keypoints_counter, H_leg)
                #print(error_start)
                Standard_keypoints_leg_2_counter, realtime_user_keypoints_counter = eliminate_zero_leg(Standard_keypoints_leg_2, realtime_user_leg_keypoints)
                error_leg_2 = mean_square_error(Standard_keypoints_leg_2_counter, realtime_user_keypoints_counter, H_leg)
                #print(error_end)
                Standard_keypoints_leg_3_counter, realtime_user_keypoints_counter = eliminate_zero_leg(Standard_keypoints_leg_3, realtime_user_leg_keypoints)                 
                error_leg_3 = mean_square_error(Standard_keypoints_leg_3_counter, realtime_user_keypoints_counter, H_leg)
                #print(error_start)
                Standard_keypoints_leg_4_counter, realtime_user_keypoints_counter = eliminate_zero_leg(Standard_keypoints_leg_4, realtime_user_leg_keypoints)
                error_leg_4 = mean_square_error(Standard_keypoints_leg_4_counter, realtime_user_keypoints_counter, H_leg)
                #print(error_start)
                
                leg_error = find_leg_min(error_leg_1, error_leg_2, error_leg_3, error_leg_4)
                
                leg_error_list.append(leg_error)
                if leg_error > fault_leg:
                    cv2.imwrite('User_Bobath/user_bobath_leg_higherror_'+str(fault_count_leg)+'.jpg', frame_1) 
                    fault_count_leg+=1
                    
            else:
                break

                
    video_realtime.release()
    
    print("Your quality of movement score for the arm tutorial is:")
    print(score(arm_error_list))
    print("Your quality of movement score for the leg tutorial is:")
    print(score(leg_error_list))
    print("Good job!")
    
    
    #rate = 0.05
    #line_arm = min(arm_error_list) + rate*(max(arm_error_list) - min (arm_error_list))
    #line_leg = min(leg_error_list) + rate*(max(leg_error_list) - min (leg_error_list))    
    
    plt.plot(arm_error_list,label='Arm Error')
    #plt.axhline(y=line_arm,label='succeed line', color='r')
    plt.xlabel("frame number")
    plt.ylabel("Mean square error")
    plt.title("Arm performance")
    plt.legend()
    plt.show()
    
    plt.plot(leg_error_list,label='Leg error')
    plt.xlabel("frame number")
    plt.ylabel("Mean square error")
    plt.title("Leg performance")
    #plt.axhline(y=line_leg,label='succeed line', color='r')
    plt.legend()
    plt.show()
    
    plt.plot(angle_1, label='Arm angles')
    plt.xlabel("frame number")
    plt.ylabel("Arm-Torso Angle (degrees)")
    plt.title("Graph plotting arm angle")
    plt.legend()
    plt.show()
    
    plt.plot(angle_2, label='Leg angles')
    plt.xlabel("frame number")
    plt.ylabel("Knee joint angle (degrees)")
    plt.title("Graph plotting knee joint angle")
    plt.legend()
    plt.show()
    
    
try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e
    
    t_wait = 7000
    size = (1500, 900)
    flag = 0
    
    pygame.mixer.init()
    pygame.mixer.music.load("PinkPartner.mp3")
    pygame.mixer.music.play(5)#add a -1 like so: pygame.mixer.music.play(-1) and the music will repeat forever
        
    
    instruction()
    
    print("Song list:")
    print("'1' for Hotline Bling")
    print("'2' for Nohands")
    print("'3' for Bad Guy")
    print("'4' for Attention")
    print("'5' for DDU-DU DDU-DU")
    print("'6' for Fire")
    print("'7' for River")
    print("'8' for Unstoppable")
    print("'9' for YOUTH")
    print("'10' for There For You") 
    print("any other number for Lemon")
    print(' ')
    print("Input number to choose your prefer song:")
    
    datafromUser=input()
    
    cv2.waitKey(15000)
    
    part1 = cv2.imread('instruction/text_part1.png')
    part1 = cv2.resize(part1, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", part1)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()
    
    Standard_keypoints_arm = []
    User_keypoints_arm = []
    
    Standard_keypoints_leg = []
    User_keypoints_leg = []    
    
    arm_error_list = []
    leg_error_list = []
    
    
    delay = 1000
    t_show = 1500
    # Posture-arm-1
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath_arm_1.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()

    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])


    # Display Image
    Standard_keypoints_arm_1 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    print('Please imitate the posture')
    Output = cv2.resize(datum.cvOutputData, (600, 1000))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", Output)
    cv2.waitKey(delay)
    time_left = 3
    while time_left > 0:
        print('Countdown for taking a photo:',time_left)
        time.sleep(1)
        time_left = time_left - 1
    cv2.destroyAllWindows()
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    ret,frame=video.read()
    cv2.imwrite('User_Bobath/user_bobath_arm_'+'1'+'.jpg', frame)
    video.release()
    user_bobath_arm_1 = cv2.imread('User_Bobath/user_bobath_arm_'+'1'+'.jpg')
    user_bobath_arm_1 = cv2.resize(user_bobath_arm_1, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath_arm_1)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_arm_1.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()
    
    
    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    
    
    User_keypoints_arm_1 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    
    Standard_keypoints_arm_1_nonzero, User_keypoints_arm_1_nonzero = eliminate_zero(Standard_keypoints_arm_1, User_keypoints_arm_1)
    
    for i in Standard_keypoints_arm_1_nonzero:
        Standard_keypoints_arm.append(i)
    for i in User_keypoints_arm_1_nonzero:
        User_keypoints_arm.append(i)
    
    # Posture-arm-2
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath_arm_2.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()
    
    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    
    
    # Display Image
    Standard_keypoints_arm_2 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    print('Please imitate the posture')
    Output = cv2.resize(datum.cvOutputData, (600, 1000))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", Output)
    cv2.waitKey(delay)
    time_left = 3
    while time_left > 0:
        print('Countdown for taking a photo:',time_left)
        time.sleep(1)
        time_left = time_left - 1
    cv2.destroyAllWindows()
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    ret,frame=video.read()
    cv2.imwrite('User_Bobath/user_bobath_arm_'+'2'+'.jpg', frame)
    video.release()
    user_bobath_arm_2 = cv2.imread('User_Bobath/user_bobath_arm_'+'2'+'.jpg')
    user_bobath_arm_2 = cv2.resize(user_bobath_arm_2, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath_arm_2)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_arm_2.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()
    
    
    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    
    
    User_keypoints_arm_2 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    
    Standard_keypoints_arm_2_nonzero, User_keypoints_arm_2_nonzero = eliminate_zero(Standard_keypoints_arm_2, User_keypoints_arm_2)
    
    for i in Standard_keypoints_arm_2_nonzero:
        Standard_keypoints_arm.append(i)
    for i in User_keypoints_arm_2_nonzero:
        User_keypoints_arm.append(i)    
    
    # Posture-arm-3
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath_arm_3.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()

    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])


    # Display Image
    Standard_keypoints_arm_3 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    print('Please imitate the posture')
    Output = cv2.resize(datum.cvOutputData, (600, 1000))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", Output)
    cv2.waitKey(delay)
    time_left = 3
    while time_left > 0:
        print('Countdown for taking a photo:',time_left)
        time.sleep(1)
        time_left = time_left - 1
    cv2.destroyAllWindows()
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    ret,frame=video.read()
    cv2.imwrite('User_Bobath/user_bobath_arm_'+'3'+'.jpg', frame)
    video.release()
    user_bobath_arm_3 = cv2.imread('User_Bobath/user_bobath_arm_'+'3'+'.jpg')
    user_bobath_arm_3 = cv2.resize(user_bobath_arm_3, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath_arm_3)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_arm_3.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()
    
    
    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    
    
    User_keypoints_arm_3 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    
    Standard_keypoints_arm_3_nonzero, User_keypoints_arm_3_nonzero = eliminate_zero(Standard_keypoints_arm_3, User_keypoints_arm_3)
    
    for i in Standard_keypoints_arm_3_nonzero:
        Standard_keypoints_arm.append(i)
    for i in User_keypoints_arm_3_nonzero:
        User_keypoints_arm.append(i)    
    
    # Posture-arm-4
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath_arm_4.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()

    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])


    # Display Image
    Standard_keypoints_arm_4 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    print('Please imitate the posture')
    Output = cv2.resize(datum.cvOutputData, (600, 1000))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", Output)
    cv2.waitKey(delay)
    time_left = 3
    while time_left > 0:
        print('Countdown for taking a photo:',time_left)
        time.sleep(1)
        time_left = time_left - 1
    cv2.destroyAllWindows()
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    ret,frame=video.read()
    cv2.imwrite('User_Bobath/user_bobath_arm_'+'4'+'.jpg', frame)
    video.release()
    user_bobath_arm_4 = cv2.imread('User_Bobath/user_bobath_arm_'+'4'+'.jpg')
    user_bobath_arm_4 = cv2.resize(user_bobath_arm_4, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath_arm_4)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_arm_4.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()
    
    
    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    
    
    User_keypoints_arm_4 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    
    Standard_keypoints_arm_4_nonzero, User_keypoints_arm_4_nonzero = eliminate_zero(Standard_keypoints_arm_4, User_keypoints_arm_4)
    
    for i in Standard_keypoints_arm_4_nonzero:
        Standard_keypoints_arm.append(i)
    for i in User_keypoints_arm_4_nonzero:
        User_keypoints_arm.append(i)    
        
    # Posture-arm-5
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath_arm_5.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()

    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])


    # Display Image
    Standard_keypoints_arm_5 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    print('Please imitate the posture')
    Output = cv2.resize(datum.cvOutputData, (600, 1000))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", Output)
    cv2.waitKey(delay)
    time_left = 3
    while time_left > 0:
        print('Countdown for taking a photo:',time_left)
        time.sleep(1)
        time_left = time_left - 1
    cv2.destroyAllWindows()
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    ret,frame=video.read()
    cv2.imwrite('User_Bobath/user_bobath_arm_'+'5'+'.jpg', frame)
    video.release()
    user_bobath_arm_5 = cv2.imread('User_Bobath/user_bobath_arm_'+'5'+'.jpg')
    user_bobath_arm_5 = cv2.resize(user_bobath_arm_5, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath_arm_5)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_arm_5.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()
    
    
    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    
    
    User_keypoints_arm_5 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    
    Standard_keypoints_arm_5_nonzero, User_keypoints_arm_5_nonzero = eliminate_zero(Standard_keypoints_arm_5, User_keypoints_arm_5)
    
    for i in Standard_keypoints_arm_5_nonzero:
        Standard_keypoints_arm.append(i)
    for i in User_keypoints_arm_5_nonzero:
        User_keypoints_arm.append(i)    
        
        


    '''print("All Standard Arm Keypoints:")
    print(Standard_keypoints_arm)
    print("All User Arm Keypoints:")
    print(User_keypoints_arm)'''
    
    Standard_keypoints_arm = np.int32(Standard_keypoints_arm)
    User_keypoints_arm = np.int32(User_keypoints_arm)
    #F, mask = cv2.findFundamentalMat(Standard_keypoints, User_keypoints, cv2.FM_LMEDS)
    H_arm, status = cv2.findHomography(Standard_keypoints_arm, User_keypoints_arm)
    #print("Arm homography matrix")
    #print(H_arm)
    
    
    # Posture-leg-1
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath_leg_1.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()

    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])


    # Display Image
    Standard_keypoints_leg_1 = keypoints_extract_leg(datum.poseKeypoints)
    print('Please imitate the posture')
    Output = cv2.resize(datum.cvOutputData, (600, 1000))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", Output)
    cv2.waitKey(delay)
    time_left = 3
    while time_left > 0:
        print('Countdown for taking a photo:',time_left)
        time.sleep(1)
        time_left = time_left - 1
    cv2.destroyAllWindows()
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    ret,frame=video.read()
    cv2.imwrite('User_Bobath/user_bobath_leg_'+'1'+'.jpg', frame)
    video.release()
    user_bobath_leg_1 = cv2.imread('User_Bobath/user_bobath_leg_'+'1'+'.jpg')
    user_bobath_leg_1 = cv2.resize(user_bobath_leg_1, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath_leg_1)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_leg_1.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()
    
    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    
    
    User_keypoints_leg_1 = keypoints_extract_leg(datum.poseKeypoints)
    
    Standard_keypoints_leg_1_nonzero, User_keypoints_leg_1_nonzero = eliminate_zero_leg(Standard_keypoints_leg_1, User_keypoints_leg_1)
    
    for i in Standard_keypoints_leg_1_nonzero:
        Standard_keypoints_leg.append(i)
    for i in User_keypoints_leg_1_nonzero:
        User_keypoints_leg.append(i)
    
    # Posture-leg-2
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath_leg_2.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()

    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])


    # Display Image
    Standard_keypoints_leg_2 = keypoints_extract_leg(datum.poseKeypoints)
    print('Please imitate the posture')
    Output = cv2.resize(datum.cvOutputData, (600, 1000))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", Output)
    cv2.waitKey(delay)
    time_left = 3
    while time_left > 0:
        print('Countdown for taking a photo:',time_left)
        time.sleep(1)
        time_left = time_left - 1
    cv2.destroyAllWindows()
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    ret,frame=video.read()
    cv2.imwrite('User_Bobath/user_bobath_leg_'+'2'+'.jpg', frame)
    video.release()
    user_bobath_leg_2 = cv2.imread('User_Bobath/user_bobath_leg_'+'2'+'.jpg')
    user_bobath_leg_2 = cv2.resize(user_bobath_leg_2, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath_leg_2)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_leg_2.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()
    
    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    
    
    User_keypoints_leg_2 = keypoints_extract_leg(datum.poseKeypoints)
    
    Standard_keypoints_leg_2_nonzero, User_keypoints_leg_2_nonzero = eliminate_zero_leg(Standard_keypoints_leg_2, User_keypoints_leg_2)
    
    for i in Standard_keypoints_leg_2_nonzero:
        Standard_keypoints_leg.append(i)
    for i in User_keypoints_leg_2_nonzero:
        User_keypoints_leg.append(i)    
    
    
    # Posture-leg-3
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath_leg_3.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()

    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])


    # Display Image
    Standard_keypoints_leg_3 = keypoints_extract_leg(datum.poseKeypoints)
    print('Please imitate the posture')
    Output = cv2.resize(datum.cvOutputData, (600, 1000))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", Output)
    cv2.waitKey(delay)
    time_left = 3
    while time_left > 0:
        print('Countdown for taking a photo:',time_left)
        time.sleep(1)
        time_left = time_left - 1
    cv2.destroyAllWindows()
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    ret,frame=video.read()
    cv2.imwrite('User_Bobath/user_bobath_leg_'+'3'+'.jpg', frame)
    video.release()
    user_bobath_leg_3 = cv2.imread('User_Bobath/user_bobath_leg_'+'3'+'.jpg')
    user_bobath_leg_3 = cv2.resize(user_bobath_leg_3, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath_leg_3)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_leg_3.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()
    
    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    
    
    User_keypoints_leg_3 = keypoints_extract_leg(datum.poseKeypoints)
    
    Standard_keypoints_leg_3_nonzero, User_keypoints_leg_3_nonzero = eliminate_zero_leg(Standard_keypoints_leg_3, User_keypoints_leg_3)
    
    for i in Standard_keypoints_leg_3_nonzero:
        Standard_keypoints_leg.append(i)
    for i in User_keypoints_leg_3_nonzero:
        User_keypoints_leg.append(i)    
        
        
    # Posture-leg-4
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath_leg_4.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()

    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])


    # Display Image
    Standard_keypoints_leg_4 = keypoints_extract_leg(datum.poseKeypoints)
    print('Please imitate the posture')
    Output = cv2.resize(datum.cvOutputData, (600, 1000))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", Output)
    cv2.waitKey(delay)
    time_left = 3
    while time_left > 0:
        print('Countdown for taking a photo:',time_left)
        time.sleep(1)
        time_left = time_left - 1
    cv2.destroyAllWindows()
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    ret,frame=video.read()
    cv2.imwrite('User_Bobath/user_bobath_leg_'+'4'+'.jpg', frame)
    video.release()
    user_bobath_leg_4 = cv2.imread('User_Bobath/user_bobath_leg_'+'4'+'.jpg')
    user_bobath_leg_4 = cv2.resize(user_bobath_leg_4, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath_leg_4)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_leg_4.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()
    
    params1 = params(args)
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params1)
    opWrapper.start()
    
    
    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    
    
    User_keypoints_leg_4 = keypoints_extract_leg(datum.poseKeypoints)
    
    Standard_keypoints_leg_4_nonzero, User_keypoints_leg_4_nonzero = eliminate_zero_leg(Standard_keypoints_leg_4, User_keypoints_leg_4)
    
    for i in Standard_keypoints_leg_4_nonzero:
        Standard_keypoints_leg.append(i)
    for i in User_keypoints_leg_4_nonzero:
        User_keypoints_leg.append(i)        
    
    '''print("All Standard Leg Keypoints:")
    print(Standard_keypoints_leg)
    print("All User Leg Keypoints:")
    print(User_keypoints_leg)'''
    
    Standard_keypoints_leg = np.int32(Standard_keypoints_leg)
    User_keypoints_leg = np.int32(User_keypoints_leg)
    #F, mask = cv2.findFundamentalMat(Standard_keypoints, User_keypoints, cv2.FM_LMEDS)
    H_leg, status = cv2.findHomography(Standard_keypoints_leg, User_keypoints_leg)
    #print("Leg homography matrix")
    #print(H_leg)
    
    
    
    part2 = cv2.imread('instruction/text_part2.png')
    part2 = cv2.resize(part2, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", part2)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()
    
    print("Try to follow the video for warm-up")
    t_1 = 0
    t_2 = 0
    t_3 = 0
    t_4 = 0
    t_5 = 0
    t_6 = 0
    t_7 = 0
    t_8 = 0
    t_9 = 0
    t_10 = 0
    
    video_calib_i = 0
    video_calib_j = 0
    
    
    t3 = threading.Thread(target=video_calib, name='control')
    t4 = threading.Thread(target=video_calib_video, name='control')
    
    t3.start()
    t4.start()
    
    t3.join()
    t4.join()
    
    part3 = cv2.imread('instruction/text_part3.png')
    part3 = cv2.resize(part3, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", part3)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()
    
    pygame.mixer.music.stop()
    print("Please follow the video tutorials")
    
    realtime_recg_i = 0
    realtime_recg_j = 0
    
    t1 = threading.Thread(target=Tutorial_videos, name='control')
    t2 = threading.Thread(target=realtime_recognition, name='control')
    t5 = threading.Thread(target=arduino_control, name='control')
    t5.start()
    t1.start()
    t2.start()

    
    
    t1.join()
    t2.join()
    t5.join()
    
    finish = cv2.imread('instruction/text_finish.png')
    finish = cv2.resize(finish, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", finish)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()
    
except Exception as e:
    print(e)
    sys.exit(-1)
