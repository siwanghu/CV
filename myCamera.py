import cv2
import numpy
from cameraManager import CameraManager
from windowsManagers import WindowManager
from scipy import ndimage

def findContours():
    frame = cv2.imread("/home/siwanghu-pc/Desktop/CV/my_camera/ham.jpg")
    ret, thre = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    img, contours, hier = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = numpy.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 3)
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(img, center, radius, (0, 255, 0), 2)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    cv2.namedWindow("test")
    cv2.imshow("test",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class Camera(object):
    def __init__(self):
        self.__window=WindowManager("camera",self.keyPress)
        self.__camera=CameraManager(cv2.VideoCapture(0),self.__window,self.canny)

    def run(self):
        self.__window.createWindow()
        while(self.__window.isWindowCreated()):
            self.__camera.showFrame()
            self.__window.windowEvents()

    def callback_face(self,frame):
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


    def high_pass_filter(self,frame):
        kernel_3X3=numpy.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        kernel_5X5=numpy.array([[-1,-1,-1,-1,-1],[-1,1,2,1,-1],[-1,2,4,2,-1],[-1,1,2,1,-1],[-1,-1,-1,-1,-1]])
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        k3=ndimage.convolve(frame,kernel_3X3)
        k5=ndimage.convolve(frame,kernel_5X5)
        blurred=cv2.GaussianBlur(frame,(11,11),0)
        g_hpf=frame-blurred
        return g_hpf

    def low_pass_filter(self,frame):
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(frame,(11,11),0)
        return blurred

    def strokeEdges(self,frame):
        temp=frame
        frame=cv2.medianBlur(frame,7)
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame=cv2.medianBlur(frame,7)
        cv2.Laplacian(frame,cv2.CV_8U,frame,5)
        normal=(1.0/255)*(255-frame)
        channels=cv2.split(temp)
        for channel in channels:
            channel[:]=channel*normal
        cv2.merge(channels,normal)
        return normal

    def filter2D(self,frame):
        kernel1=numpy.array([[-2,-1,0],[-1,1,1],[0,1,2]])
        kernel2=numpy.array([[-1,-1,-1],[-1,8,-1],[-1,-1,1]])
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.filter2D(frame,-1,kernel2,frame)
        return frame

    def canny(self,frame):
        temp = frame
        frame=cv2.Canny(frame,100,200)
        normal = (1.0 / 255) * (255 - frame)
        channels = cv2.split(temp)
        for channel in channels:
            channel[:] = channel * normal
        cv2.merge(channels, normal)
        return normal

    def threshold(self,frame):
        ret,frame=cv2.threshold(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),127,255,cv2.THRESH_BINARY)
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
    #findContours()













