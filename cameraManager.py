import cv2
import numpy

class CameraManager(object):
    def __init__(self,camera,windows,callback=None):
        self.__camera=camera
        self.__windows=windows
        self.__frame=None
        self.__record=False
        self.__video=None
        self.__fps=60
        self.__size=None
        self.__callback=callback

    def showFrame(self):
        self.__success,self.__frame=self.__camera.read()
        if self.__frame!=None:
            if self.__callback!=None:
                self.__frame=self.__callback(self.__frame)
            self.__windows.showFrame(self.__frame)
        if self.__record:
            self.__video.write(self.__frame)

    def saveFrame(self):
        cv2.imwrite("shot.jpg",self.__frame)

    def startRecord(self):
        self.__record=True
        self.__size=(int(self.__camera.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.__camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.__video=cv2.VideoWriter("output.avi",cv2.VideoWriter_fourcc("I","4","2","0"),self.__fps,self.__size)

    def endRecord(self):
        self.__record = False
        self.__size=None
        self.__video=None

    def release(self):
        self.__camera.release()