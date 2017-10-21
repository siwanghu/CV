import cv2
import numpy

class WindowManager(object):
    def __init__(self, name, callback=None):
        self.__name = name
        self.__callback = callback
        self.__isWindowCreated = False

    def isWindowCreated(self):
        return self.__isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self.__name)
        self.__isWindowCreated=True

    def showFrame(self,frame):
        if(self.__isWindowCreated==True):
            cv2.imshow(self.__name,frame)

    def destoryWindow(self):
        cv2.destroyAllWindows()
        self.__isWindowCreated=False

    def windowEvents(self):
        keycode=cv2.waitKey(1)
        if(self.__callback!=None and keycode!=-1):
            keycode=keycode&0xFF
            self.__callback(keycode)


