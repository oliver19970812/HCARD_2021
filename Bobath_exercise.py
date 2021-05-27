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

# 定义函数，第一个参数是缩放比例，第二个参数是需要显示的图片组成的元组或者列表
def instruction():
    
    size = (1500,900)
    t_wait = 1000
    
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
    
    ins_8 = cv2.imread('instruction/ins_8.png')
    ins_8 = cv2.resize(ins_8, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", ins_8)
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

def keypoints_extract(Body_Keypoints, Right_hand_keypoints, Left_hand_keypoints):
    Body_Keypoints = np.delete(Body_Keypoints, -1, axis=2)
    Right_hand_keypoints = np.delete(Right_hand_keypoints, -1, axis=2)
    Left_hand_keypoints = np.delete(Left_hand_keypoints, -1, axis=2)
    #print(Body_Keypoints)
    # print(Left_hand_keypoints)
    # print(Right_hand_keypoints)
    def compare (x,y):
        a = x[1]
        b = y[1]
        if a > b:
            return x
        else:
            return y

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

    keypoints = [nose, eye, ear, shoulder, elbow, wrist, hand5, hand17]
    #print(keypoints)
    return keypoints

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

def score(data):
    _range = np.max(data) - np.min(data)
    scores = 100-((data - np.min(data)) / _range)*40
    return np.mean(scores)    

def arduino_control():
    global realtime_recg_k
    print("Input number to choose your prefer song:")
    datafromUser=input()
    ser = serial.Serial('COM6', 9600,timeout=1)
    time.sleep(2)
    # Read and record the data
    i=0        
    bad,good=0,0       
    recordSpeed=[]
    
    if datafromUser == '0':
        ser.write(b'0')
        print("Canon D")
    elif datafromUser == '1':
        ser.write(b'1')
        print("Pink Partner")
    else:
        ser.write(b'2')
        print("Harry Potter")
    
        
    while(1):
        data =[]  
        b = ser.readline()         # read a byte string
        string_n = b.decode() # decode byte string into Unicode  
        string = string_n.rstrip() # remove \n and \r
        string= string.strip(' ')
        print('times:',i)
        print(string)
        
        if not string:
            print("The string is empty")
        else:
            for s in string.split(' '):
                s=float(s)
                data.append(s)  
                if len(data)%3==0:
                    print(data)
                    recordSpeed.append(data)
        if realtime_recg_k > 1200:
            break
            
        ###############歌曲再while里，需要信号跳出while
        ##if signal:
        ##break
        #############################################
    ser.close()#close serial port
    #calculate result
    record=np.array(recordSpeed)
    row,col=np.shape(record)
    
    for i in range(row):
        if (abs(record[i][0])<5 and abs(record[i][1])<5 and abs(record[i][2])<5)\
        or (abs(record[i][0])>200 or abs(record[i][1])>200 or abs(record[i][2])>200 ):
            bad+=1
        else:
            good+=1
            
    score=good/(good+bad)*100
    print('your score is:', score)


def Tutorial_videos():
    
    #tutorial_a = cv2.VideoCapture('Bobath_Tutorial/Bobath-a.mp4')  
    print('Press q to quit')
    global realtime_recg_i
    global realtime_recg_j
    global realtime_recg_k
    '''while(tutorial_a.isOpened()):  
        ret, frame = tutorial_a.read()  
        frame = cv2.resize(frame, (1000, 600))
        cv2.imshow('Tutorial_a', frame)
        k = cv2.waitKey(1)
        #q键退出
        realtime_recg_i+=1
        if realtime_recg_i > 1410:
            break
        if realtime_recg_i == 1000:
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
    
    tutorial_b = cv2.VideoCapture('Bobath_Tutorial/Bobath-b.mp4')
    while(tutorial_b.isOpened()):
        ret, frame = tutorial_b.read()
        frame = cv2.resize(frame, (1000, 600))
        cv2.imshow('Tutorial_b', frame)
        k = cv2.waitKey(1)  
        #q键退出
        realtime_recg_j+=1
        if realtime_recg_j > 1530:
            break
        if (k & 0xff == ord('q')):  
            break  
    tutorial_b.release()  
    cv2.destroyAllWindows()'''
    
    tutorial_c = cv2.VideoCapture('Bobath_Tutorial/Bobath-c.mp4')  
    while(tutorial_c.isOpened()):  
        ret, frame = tutorial_c.read()  
        frame = cv2.resize(frame, (1000, 600))
        cv2.imshow('Tutorial_c', frame)  
        k = cv2.waitKey(1)  
        #q键退出
        realtime_recg_k+=1
        if realtime_recg_k > 900:
            break
        if realtime_recg_k == 700:
            text_gj = cv2.imread('instruction/text_gj.png')
            text_gj = cv2.resize(text_gj, (1000,600))
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
        B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    if c==0 or e==0:
        D = 0
        #print('Cannot detect the angle')
    else:
    #C = math.degrees(math.acos((c * c - a * a - b * b) / (-2 * a * b)))
        D = math.degrees(math.acos((d * d - c * c - e * e) / (-2 * c * e)))
    #print(B, D)
    return B,D

def set_threshold(t_list):
    
    r = 0.05
    t_max = max(t_list)
    t_min = min(t_list)
    t = t_min + (t_max - t_min)*r
    
    return t


def video_calib_video():
    
    global video_calib_i
    global video_calib_j
    global video_calib_k
    
    '''tutorial_a = cv2.VideoCapture('Bobath_Tutorial/Bobath-a.mp4')  
    print('Press q to quit')
    video_calib_i = 0
    while video_calib_i<=450:
        ret, frame = tutorial_a.read()  
        frame = cv2.resize(frame, (1000, 600))
        
        cv2.imshow('Tutorial Part 1 of 3', frame)
        
        video_calib_i+=1
        k = cv2.waitKey(1)

        if (k & 0xff == ord('q')):  
            break  
    tutorial_a.release()
    cv2.destroyAllWindows()
    
    
    tutorial_b = cv2.VideoCapture('Bobath_Tutorial/Bobath-b.mp4')  
    print('Press q to quit')
    video_calib_j = 0
    while video_calib_j<=600:
        ret, frame = tutorial_b.read()  
        frame = cv2.resize(frame, (1000, 600))

        cv2.imshow('Tutorial Part 2 of 3', frame)
        
        video_calib_j+=1
        k = cv2.waitKey(1)

        if (k & 0xff == ord('q')):  
            break  
    tutorial_b.release()
    cv2.destroyAllWindows()'''
    
    tutorial_c = cv2.VideoCapture('Bobath_Tutorial/Bobath-c.mp4')  
    print('Press q to quit')
    video_calib_k = 0
    while video_calib_k<=480:
        ret, frame = tutorial_c.read()  
        frame = cv2.resize(frame, (1000, 600))

        cv2.imshow('Tutorial Part 3 of 3', frame)
        video_calib_k+=1
        k = cv2.waitKey(1)

        if (k & 0xff == ord('q')):  
            break  
    tutorial_c.release()
    return 0

def video_calib():
    
    global t_1
    global t_2
    global t_3
    global t_4
    global t_5
    global t_6
    
    global video_calib_i
    global video_calib_j
    global video_calib_k
    
    pos_error_list_1 = []
    pos_error_list_2 = []
    pos_error_list_3 = []
    pos_error_list_4 = []
    pos_error_list_5 = []
    pos_error_list_6 = []
    
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
    '''video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    while video_calib_i<=450:
        
        ret_1,frame_1=video.read()
        #if ret == True:
        rows,cols,ch = frame_1.shape
        datum_calib.cvInputData = frame_1
        opWrapper.emplaceAndPop([datum_calib])
        
        cv2.imshow("Warm-up Part 1 of 3", datum_calib.cvOutputData)
        
        calib_user_keypoints = keypoints_extract(datum_calib.poseKeypoints, datum_calib.handKeypoints[1], datum_calib.handKeypoints[0])        
        Standard_keypoints_1_counter, calib_user_keypoints_counter_1 = eliminate_zero(Standard_keypoints_1, calib_user_keypoints) 
        Standard_keypoints_2_counter, calib_user_keypoints_counter_2 = eliminate_zero(Standard_keypoints_2, calib_user_keypoints)
        
        error_start = mean_square_error(Standard_keypoints_1_counter, calib_user_keypoints_counter_1, H)
        error_end = mean_square_error(Standard_keypoints_2_counter, calib_user_keypoints_counter_2, H)
        
        pos_error_list_1.append(error_start)
        pos_error_list_2.append(error_end)
        
        #i+=1
        k = cv2.waitKey(50)
        if (k & 0xff == ord('q')):  
            break 
        
    video.release()
    cv2.destroyAllWindows()
    
    t_1 = set_threshold(pos_error_list_1)
    t_2 = set_threshold(pos_error_list_2)
    print("Threshold 1 : %d" %t_1)
    print("Threshold 2 : %d" %t_2)
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    
    while video_calib_j<=600:
        
        ret_1,frame_1=video.read()
        #if ret == True:
        rows,cols,ch = frame_1.shape
        datum_calib.cvInputData = frame_1
        opWrapper.emplaceAndPop([datum_calib])
        
        cv2.imshow("Warm-up Part 2 of 3", datum_calib.cvOutputData)
                
        calib_user_keypoints = keypoints_extract(datum_calib.poseKeypoints, datum_calib.handKeypoints[1], datum_calib.handKeypoints[0])        
        Standard_keypoints_3_counter, calib_user_keypoints_counter_3 = eliminate_zero(Standard_keypoints_3, calib_user_keypoints) 
        Standard_keypoints_4_counter, calib_user_keypoints_counter_4 = eliminate_zero(Standard_keypoints_4, calib_user_keypoints)
        
        error_start = mean_square_error(Standard_keypoints_3_counter, calib_user_keypoints_counter_3, H)
        error_end = mean_square_error(Standard_keypoints_4_counter, calib_user_keypoints_counter_3, H)
        
        pos_error_list_3.append(error_start)
        pos_error_list_4.append(error_end)
        
        
        k = cv2.waitKey(50)
        if (k & 0xff == ord('q')):  
            break 
    video.release()
    cv2.destroyAllWindows()
    
    t_3 = set_threshold(pos_error_list_3)
    t_4 = set_threshold(pos_error_list_4)
    print("Threshold 3 : %d" %t_3)
    print("Threshold 4 : %d" %t_4)'''
    
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while video_calib_k<=480:
        
        ret_1,frame_1=video.read()
        #if ret == True:
        rows,cols,ch = frame_1.shape
        datum_calib.cvInputData = frame_1
        opWrapper.emplaceAndPop([datum_calib])
        
        cv2.imshow("Warm-up Part 3 of 3", datum_calib.cvOutputData)
                
        calib_user_keypoints = keypoints_extract(datum_calib.poseKeypoints, datum_calib.handKeypoints[1], datum_calib.handKeypoints[0])        
        Standard_keypoints_5_counter, calib_user_keypoints_counter_5 = eliminate_zero(Standard_keypoints_5, calib_user_keypoints) 
        Standard_keypoints_6_counter, calib_user_keypoints_counter_6 = eliminate_zero(Standard_keypoints_6, calib_user_keypoints)
        
        error_start = mean_square_error(Standard_keypoints_5_counter, calib_user_keypoints_counter_5, H)
        error_end = mean_square_error(Standard_keypoints_6_counter, calib_user_keypoints_counter_6, H)
        
        pos_error_list_5.append(error_start)
        pos_error_list_6.append(error_end)
        
        k = cv2.waitKey(50)
        if (k & 0xff == ord('q')):  
            break 
    
    t_5 = set_threshold(pos_error_list_5)
    t_6 = set_threshold(pos_error_list_6)
    print("Threshold 5 : %d" %t_5)
    print("Threshold 6 : %d" %t_6)
    video.release()
    cv2.destroyAllWindows()

def realtime_recognition():
    
    global H
    global Standard_keypoints_1
    global Standard_keypoints_2
    global Standard_keypoints_3
    global Standard_keypoints_4
    global Standard_keypoints_5
    global Standard_keypoints_6
    
    global posture_1_error_list
    global posture_2_error_list
    global posture_3_error_list
    global posture_4_error_list
    global posture_5_error_list
    global posture_6_error_list
    
    global t_1
    global t_2
    global t_3
    global t_4
    global t_5
    global t_6
    
    global realtime_recg_i
    global realtime_recg_j
    global realtime_recg_k
    
    counter_start_1 = 0
    counter_end_1 = 0
    counter_start_2 = 0
    counter_end_2 = 0
    counter_start_3 = 0
    counter_end_3 = 0
    
    angle_1 = []
    angle_2 = []
    angle_3 = []
    
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

    posture_comp_1 = []

    while(video_realtime.isOpened()):
        

        ret_1,frame_1=video_realtime.read()
        #if ret == True:
        rows,cols,ch = frame_1.shape
        datum_realtime.cvInputData = frame_1
        opWrapper.emplaceAndPop([datum_realtime])
        #print("Left hand keypoints: \n" + str(datum_realtime.handKeypoints[0]))
        #print("Right hand keypoints: \n" + str(datum_realtime.handKeypoints[1]))
        #print("Body keypoints: \n" + str(datum_realtime.poseKeypoints))
        cv2.imshow("Realtime_recognition", datum_realtime.cvOutputData)
        
        k = cv2.waitKey(50)
        #q键退出
        if (k & 0xff == ord('q')):  
            break
        
        realtime_user_keypoints = keypoints_extract(datum_realtime.poseKeypoints, datum_realtime.handKeypoints[1], datum_realtime.handKeypoints[0])
        

        if k:
            #time = cv2.waitKey(50)
            '''if realtime_recg_i < 1410:
                angle_1.append(get_degree(realtime_user_keypoints))
                Standard_keypoints_1_counter, realtime_user_keypoints_counter = eliminate_zero(Standard_keypoints_1, realtime_user_keypoints) 
                error_start_1 = mean_square_error(Standard_keypoints_1_counter, realtime_user_keypoints_counter, H)
                #print(error_start)
                if error_start_1 < t_1:
                    counter_start_1 += 1
                Standard_keypoints_2_counter, realtime_user_keypoints_counter = eliminate_zero(Standard_keypoints_2, realtime_user_keypoints)
                error_end_1 = mean_square_error(Standard_keypoints_2_counter, realtime_user_keypoints_counter, H)
                #print(error_end)
                if error_end_1 < t_2:
                    counter_end_1 +=1
                if np.abs(counter_start_1 - counter_end_1) > 1:
                    counter = max(counter_start_1, counter_end_1)
                    if counter_start_1 > counter_end_1:
                        counter_start_1 = counter - 1
                    else:
                        counter_end_1 = counter - 1
                posture_1_error_list.append(error_start_1)
                posture_2_error_list.append(error_end_1)
                #print("You already did %d times" %counter_end_1)
            elif realtime_recg_j < 1530:
                angle_2.append(get_degree(realtime_user_keypoints))
                Standard_keypoints_3_counter, realtime_user_keypoints_counter = eliminate_zero(Standard_keypoints_3, realtime_user_keypoints)                 
                error_start_2 = mean_square_error(Standard_keypoints_3_counter, realtime_user_keypoints_counter, H)
                #print(error_start)
                if error_start_2 < t_3:
                    counter_start_2 += 1
                Standard_keypoints_4_counter, realtime_user_keypoints_counter = eliminate_zero(Standard_keypoints_4, realtime_user_keypoints)
                error_end_2 = mean_square_error(Standard_keypoints_4_counter, realtime_user_keypoints_counter, H)
                #print(error_end)
                if error_end_2 < t_4:
                    counter_end_2 +=1
                if np.abs(counter_start_2 - counter_end_2) > 1:
                    counter = max(counter_start_2, counter_end_2)
                    if counter_start_2 > counter_end_2:
                        counter_start_2 = counter - 1
                    else:
                        counter_end_2 = counter - 1
                posture_3_error_list.append(error_start_2)
                posture_4_error_list.append(error_end_2)  
                #print("You already did %d times" %counter_end_2)  '''        
            if realtime_recg_k < 900:
                angle_3.append(get_degree(realtime_user_keypoints)[0])
                Standard_keypoints_5_counter, realtime_user_keypoints_counter = eliminate_zero(Standard_keypoints_5, realtime_user_keypoints)                 
                error_start_3 = mean_square_error(Standard_keypoints_5_counter, realtime_user_keypoints_counter, H)
                #print(error_start)
                if error_start_3 < t_5:
                    counter_start_3 += 1
                Standard_keypoints_6_counter, realtime_user_keypoints_counter = eliminate_zero(Standard_keypoints_6, realtime_user_keypoints)
                error_end_3 = mean_square_error(Standard_keypoints_6_counter, realtime_user_keypoints_counter, H)
                #print(error_end)
                if error_end_3 < t_6:
                    counter_end_3 +=1
                if np.abs(counter_start_3 - counter_end_3) > 1:
                    counter = max(counter_start_3, counter_end_3)
                    if counter_start_3 > counter_end_3:
                        counter_start_3 = counter - 1
                    else:
                        counter_end_3 = counter - 1
                
                
                if error_end_3 > error_start_3:
                    posture_comp_1.append(error_start_3)
                else:
                    posture_comp_1.append(error_end_3)
                    
                posture_5_error_list.append(error_start_3)
                posture_6_error_list.append(error_end_3)
                #print("You already did %d times" %counter_end_3)     
            else:
                break

                
    video_realtime.release()
    
    '''print("Your score on tutorial 1 is:")
    print(score(posture_2_error_list))
    print("Your score on tutorial 2 is:")
    print(score(posture_4_error_list))'''
    print("Your score on tutorial 3 is:")
    print(score(posture_6_error_list))
    print("Good job!")
    
    
    #succeed_list_1 = np.zeros(len(posture_2_error_list))
    #succeed_list_2 = np.zeros(len(posture_4_error_list))
    #succeed_list_3 = np.zeros(len(posture_6_error_list))
    rate = 0.3
    #line1 = min(posture_2_error_list) + rate*(max(posture_2_error_list) - min (posture_2_error_list))
    #line2 = min(posture_4_error_list) + rate*(max(posture_4_error_list) - min (posture_4_error_list))
    #line3 = min(posture_6_error_list) + rate*(max(posture_6_error_list) - min (posture_6_error_list))
    line3 = min(posture_comp_1) + rate*(max(posture_comp_1) - min (posture_comp_1))
    
    '''for i in succeed_list_1:
        succeed_list_1[i] = line1
    
    for j in succeed_list_2:
        succeed_list_2[j] = line2
    
    for k in succeed_list_3:
        succeed_list_3[k] = line3
    line1'''
    
    
    '''plt.plot(posture_2_error_list,label='posture 1')
    #plt.plot(succeed_list_1,label='succeed line')
    plt.axhline(y=line1,label='succeed line')
    plt.legend()
    plt.show()
    
    plt.plot(posture_4_error_list,label='posture 2')
    #plt.plot(succeed_list_2,label='succeed line')
    plt.axhline(y=line2,label='succeed line')
    plt.legend()
    plt.show()'''
    
    plt.plot(posture_comp_1,label='posture 3')
    #plt.plot(succeed_list_3,label='succeed line')
    plt.axhline(y=line3,label='succeed line', color='r')
    plt.legend()
    plt.show()
    
    '''plt.plot(angle_1, label='posture 1 angles')
    plt.legend()
    plt.show()
    plt.plot(angle_2, label='posture 2 angles')
    plt.legend()
    plt.show()'''
    plt.plot(angle_3, label='posture 3 angles')
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
    size = (1000, 600)
    
    instruction()
    
    part1 = cv2.imread('instruction/text_part1.png')
    part1 = cv2.resize(part1, size)
    cv2.namedWindow("instructions", 1)
    cv2.imshow("instructions", part1)
    cv2.waitKey(t_wait)
    cv2.destroyAllWindows()
    
    Standard_keypoints = []
    User_keypoints = []
    
    posture_1_error_list = []
    posture_2_error_list = []
    posture_3_error_list = []
    posture_4_error_list = []
    posture_5_error_list = []
    posture_6_error_list = []
    delay = 1000
    t_show = 1500
    # Posture-1
    '''parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath-1.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    #print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    #print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
    Standard_keypoints_1 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    #datum.cvOutputData = cv2.resize(datum.cvOutputData, (500, 500))
    #video = cv2.resize(video, (500, 500))
    print('Please imitate the posture')
    #hmerge = np.hstack((datum.cvOutputData, video))
    Output = cv2.resize(datum.cvOutputData, (1000, 600))
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
    cv2.imwrite('User_Bobath/user_bobath_'+'a'+'.jpg', frame)
    video.release()
    user_bobath1 = cv2.imread('User_Bobath/user_bobath_'+'a'+'.jpg')
    user_bobath1 = cv2.resize(user_bobath1, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath1)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_a.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    
    User_keypoints_1 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    
    Standard_keypoints_1_nonzero, User_keypoints_1 = eliminate_zero(Standard_keypoints_1, User_keypoints_1)
    
    for i in Standard_keypoints_1_nonzero:
        Standard_keypoints.append(i)
    for i in User_keypoints_1:
        User_keypoints.append(i)
    
    # Posture-2
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath-2.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    #print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    #print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
    Standard_keypoints_2 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    #stackedimageh = ManyImgs(0.2, ([datum.cvOutputData, video]))
    #datum.cvOutputData = cv2.resize(datum.cvOutputData, (500, 500))
    #video = cv2.resize(video, (500, 500))
    print('Please imitate the posture')
    #hmerge = np.hstack((datum.cvOutputData, video))
    Output = cv2.resize(datum.cvOutputData, (1000, 600))
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
    cv2.imwrite('User_Bobath/user_bobath_'+'b'+'.jpg', frame)
    video.release()
    user_bobath2 = cv2.imread('User_Bobath/user_bobath_'+'b'+'.jpg')
    user_bobath2 = cv2.resize(user_bobath2, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath2)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_b.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    
    User_keypoints_2 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    
    Standard_keypoints_2_nonzero, User_keypoints_2 = eliminate_zero(Standard_keypoints_2, User_keypoints_2)
    
    for i in Standard_keypoints_2_nonzero:
        Standard_keypoints.append(i)
    for i in User_keypoints_2:
        User_keypoints.append(i)
    
    # Posture-3
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath-3.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    #print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    #print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
    Standard_keypoints_3 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    #stackedimageh = ManyImgs(0.2, ([datum.cvOutputData, video]))
    #datum.cvOutputData = cv2.resize(datum.cvOutputData, (500, 500))
    #video = cv2.resize(video, (500, 500))
    print('Please imitate the posture')
    #hmerge = np.hstack((datum.cvOutputData, video))
    Output = cv2.resize(datum.cvOutputData, (1000, 600))
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
    cv2.imwrite('User_Bobath/user_bobath_'+'c'+'.jpg', frame)
    video.release()
    user_bobath3 = cv2.imread('User_Bobath/user_bobath_'+'c'+'.jpg')
    user_bobath3 = cv2.resize(user_bobath3, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath3)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_c.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    
    User_keypoints_3 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    
    Standard_keypoints_3_nonzero, User_keypoints_3 = eliminate_zero(Standard_keypoints_3, User_keypoints_3)
    
    for i in Standard_keypoints_3_nonzero:
        Standard_keypoints.append(i)
    for i in User_keypoints_3:
        User_keypoints.append(i)
    
    
    # Posture-4
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath-4.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    #print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    #print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
    Standard_keypoints_4 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    #stackedimageh = ManyImgs(0.2, ([datum.cvOutputData, video]))
    #datum.cvOutputData = cv2.resize(datum.cvOutputData, (500, 500))
    #video = cv2.resize(video, (500, 500))
    print('Please imitate the posture')
    #hmerge = np.hstack((datum.cvOutputData, video))
    Output = cv2.resize(datum.cvOutputData, (1000, 600))
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
    cv2.imwrite('User_Bobath/user_bobath_'+'d'+'.jpg', frame)
    video.release()
    user_bobath4 = cv2.imread('User_Bobath/user_bobath_'+'d'+'.jpg')
    user_bobath4 = cv2.resize(user_bobath4, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath4)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_d.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    
    User_keypoints_4 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    
    Standard_keypoints_4_nonzero, User_keypoints_4 = eliminate_zero(Standard_keypoints_4, User_keypoints_4)
    
    for i in Standard_keypoints_4_nonzero:
        Standard_keypoints.append(i)
    for i in User_keypoints_4:
        User_keypoints.append(i)'''
    
    
    # Posture-5
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath-5.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    #print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    #print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
    Standard_keypoints_5 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    #stackedimageh = ManyImgs(0.2, ([datum.cvOutputData, video]))
    #datum.cvOutputData = cv2.resize(datum.cvOutputData, (500, 500))
    #video = cv2.resize(video, (500, 500))
    print('Please imitate the posture')
    #hmerge = np.hstack((datum.cvOutputData, video))
    Output = cv2.resize(datum.cvOutputData, (1000, 600))
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
    cv2.imwrite('User_Bobath/user_bobath_'+'e'+'.jpg', frame)
    video.release()
    user_bobath5 = cv2.imread('User_Bobath/user_bobath_'+'e'+'.jpg')
    user_bobath5 = cv2.resize(user_bobath5, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath5)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_e.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    
    
    User_keypoints_5 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    
    Standard_keypoints_5_nonzero, User_keypoints_5 = eliminate_zero(Standard_keypoints_5, User_keypoints_5)
    
    for i in Standard_keypoints_5_nonzero:
        Standard_keypoints.append(i)
    for i in User_keypoints_5:
        User_keypoints.append(i)
    
    
    
    # Posture-6
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="Standard_Bobath/Bobath-6.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    #print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    #print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
    Standard_keypoints_6 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    #stackedimageh = ManyImgs(0.2, ([datum.cvOutputData, video]))
    #datum.cvOutputData = cv2.resize(datum.cvOutputData, (500, 500))
    #video = cv2.resize(video, (500, 500))
    print('Please imitate the posture')
    #hmerge = np.hstack((datum.cvOutputData, video))
    Output = cv2.resize(datum.cvOutputData, (1000, 600))
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
    cv2.imwrite('User_Bobath/user_bobath_'+'f'+'.jpg', frame)
    video.release()
    user_bobath6 = cv2.imread('User_Bobath/user_bobath_'+'f'+'.jpg')
    user_bobath6 = cv2.resize(user_bobath6, (1000, 600))
    cv2.namedWindow("Bobath", 1)
    cv2.imshow("Bobath", user_bobath6)
    cv2.waitKey(t_show)
    cv2.destroyAllWindows()
    
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="User_Bobath/user_bobath_f.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
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
    
  
    User_keypoints_6 = keypoints_extract(datum.poseKeypoints, datum.handKeypoints[1], datum.handKeypoints[0])
    
    Standard_keypoints_6_nonzero, User_keypoints_6 = eliminate_zero(Standard_keypoints_6, User_keypoints_6)
    
    for i in Standard_keypoints_6_nonzero:
        Standard_keypoints.append(i)
    for i in User_keypoints_6:
        User_keypoints.append(i)
    
    print("All Standard Keypoints:")
    print(Standard_keypoints)
    print("All User Keypoints:")
    print(User_keypoints)
    
    Standard_keypoints = np.int32(Standard_keypoints)
    User_keypoints = np.int32(User_keypoints)
    #F, mask = cv2.findFundamentalMat(Standard_keypoints, User_keypoints, cv2.FM_LMEDS)
    H, status = cv2.findHomography(Standard_keypoints, User_keypoints)
    print(H)
    
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
    
    video_calib_i = 0
    video_calib_j = 0
    video_calib_k = 0
    
    
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
    
    print("Please follow the video tutorials")
    
    realtime_recg_i = 0
    realtime_recg_j = 0
    realtime_recg_k = 0
    
    t1 = threading.Thread(target=Tutorial_videos, name='control')
    t2 = threading.Thread(target=realtime_recognition, name='control')
    t5 = threading.Thread(target=arduino_control, name='control')
    t1.start()
    t2.start()
    t5.start()
    
    
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
