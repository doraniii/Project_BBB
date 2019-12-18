import cv2

face_cascade = cv2.CascadeClassifier('C:\\data\\nw\\opencvWithPython-master\\haarcascade_frontalface_default.xml')
#eye_casecade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')

path = 'C:\\Users\\knit\\Desktop\\googleimg\\'

list = []
for i in range(100,200):
    list.append('%d.jpg' %i)

for s in list:
    #print(s)
    img = cv2.imread(path + s)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    
    for (x,y,w,h) in faces:
        cropped = img[y - int(h / 5):y + h + int(h / 5), x - int(w / 5):x + w + int(w / 5)]
        # 이미지를 저장
        cv2.imwrite(path + "face" + s, cropped)
    
    #cv2.imshow('Image view', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
