{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "\n",
    "face_cascade = cv2.CascadeClassifier('C:\\\\data\\\\nw\\\\opencvWithPython-master\\\\haarcascade_frontalface_default.xml')\n",
    "#eye_casecade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')\n",
    "\n",
    "path = 'C:\\\\Users\\\\knit\\\\Desktop\\\\googleimg\\\\'\n",
    "\n",
    "list = []\n",
    "for i in range(100,200):\n",
    "    list.append('%d.jpg' %i)\n",
    "\n",
    "for s in list:\n",
    "    #print(s)\n",
    "    img = cv2.imread(path + s)\n",
    "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "    faces = face_cascade.detectMultiScale(gray, 1.3,5)\n",
    "    \n",
    "    for (x,y,w,h) in faces:\n",
    "        cropped = img[y - int(h / 5):y + h + int(h / 5), x - int(w / 5):x + w + int(w / 5)]\n",
    "        # 이미지를 저장\n",
    "        cv2.imwrite(path + \"face\" + s, cropped)\n",
    "    \n",
    "    #cv2.imshow('Image view', img)\n",
    "    cv2.waitKey(0)\n",
    "    cv2.destroyAllWindows()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
