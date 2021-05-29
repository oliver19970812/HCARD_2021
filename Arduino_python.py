import serial
import time
import numpy as np
import cv2
########需要信号开始####
########################

print("Input number to choose your prefer song:")
datafromUser=input()
ser = serial.Serial('COM5', 9600,timeout=1)
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
    k = cv2.waitKey(2)
    if (k & 0xff == ord('q')):
        break
    if i > 50:
        break
    i+=1
        
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

