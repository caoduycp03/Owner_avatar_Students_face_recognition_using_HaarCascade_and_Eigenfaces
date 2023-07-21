# pip install opencv-python==4.5.2

import cv2 

video=cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count=0

while True:
    ret,frame=video.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        count=count+1
        cv2.imwrite(r"faces\\train\Duydeptraixy" + str(count)+ ".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    cv2.imshow("Frame",frame)


    key = cv2.waitKey(1)
    if(key == 27):
        break 
    if count>199:
        break

video.release()
cv2.destroyAllWindows()
print("Dataset Collection Done..................")