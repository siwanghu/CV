import cv2
import numpy
from cameraManager import CameraManager
from windowsManagers import WindowManager

class Camera(object):
    def __init__(self):
        self.__window=WindowManager("camera",self.keyPress)
        self.__camera=CameraManager(cv2.VideoCapture(0),self.__window,self.callback)

    def run(self):
        self.__window.createWindow()
        while(self.__window.isWindowCreated()):
            self.__camera.showFrame()
            self.__window.windowEvents()

    def callback(self,frame):
        classifier = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
        size = frame.shape[:2]
        image = numpy.zeros(size, dtype=numpy.float16)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.equalizeHist(image, image)
        divisor = 8
        h, w = size
        minSize = (w / divisor, h / divisor)
        faceRects = classifier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, minSize)
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect
                cv2.circle(frame, (x + w / 2, y + h / 2), min(w / 2, h / 2), (0, 0, 255))
        return frame

    def keyPress(self,keycode):
        if keycode == ord("s"):
            self.__camera.saveFrame()
        if keycode == ord("q"):
            self.__camera.release()
            self.__window.destoryWindow()
        if keycode == ord("a"):
            self.__camera.startRecord()
        if(keycode == ord("z")):
            self.__camera.endRecord()

if(__name__=="__main__"):
    Camera().run()