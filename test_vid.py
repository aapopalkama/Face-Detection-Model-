import cv2
import face_recognition
import os
import numpy as np
import socket
from goprocam import GoProCamera, constants



#   Connect to webcam or video

#   Use your gopro as webcam<
# WRITE = False
# gpCam = GoProCamera.GoPro()
# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# gpCam.livestream("start")
# gpCam.video_settings(res='1980', fps='30')
# gpCam.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)
# video_capture = cv2.VideoCapture("udp://10.5.5.9:8554", cv2.CAP_FFMPEG)
#>

#   If you want to use your computer camera
video_capture  = cv2.VideoCapture(0)

#   If you want use video from file 
#   video_capture = cv2.VideoCapture("filename")



#   Image path to known people
path = "images"
images = []
classNames = []
mylist = os.listdir(path)


for cl in mylist:
    #   Find names of known images
    Img = cv2.imread(f'{path}/{cl}')
    images.append(Img)
    #    extract the person name from the image path
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        #   convert the input frame from BGR to RGB 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList
encoded_face_train = findEncodings(images)

while True:
    #   grab the frame from the threaded video stream
    success, img = video_capture.read()
    imgS = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    
 
    #   loop over the facial embeddings incase
    #   we have multiple embeddings for multiple fcaes   
    for encode_face, faceloc in zip(encoded_faces,faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
       
        if matches[matchIndex]:
            #   Check known images
            name = classNames[matchIndex].upper().lower()
            y1,x2,y2,x1 = faceloc
            #   since we scaled down by 4 times
            y1, x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            #   Draw a box around the face
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            #   Draw a label with a name below the face
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            #   Put text to image
            cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),1)
          
    cv2.imshow('Video', img)
    #   Press q if you want to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#   Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()