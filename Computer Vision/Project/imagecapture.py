import cv2, os
import imutils
import numpy
import urllib.request
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'images'  
sub_data = 'Upayan'     

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
# url='http://192.168.202.230:8080/shot.jpg'
count = 1
while count < 601:
    print(count)
    (_, im) = webcam.read()
    # imgPath = urllib.request.urlopen(url)
    # imgNp = numpy.array(bytearray(imgPath.read()), dtype=numpy.uint8)
    # im = cv2.imdecode(imgNp, -1)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('%s/%s.png' % (path,count), face_resize)
    count += 1
	
    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(10)
    if key == 27:
        break
print("Dataset obtained successfully")
webcam.release()
cv2.destroyAllWindows()