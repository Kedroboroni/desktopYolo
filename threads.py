""" Файл в котором описаны потоки и процессы
    (Работы Yolo8, окна приложения, отдельных функций) """

from PySide6.QtCore import QThread, Signal, Qt, Slot, QTimer
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np
from ultralytics import YOLO
from events import preparationArray, ndarray2qpixmap, postVideo
from qimage2ndarray import array2qimage





class videoThread(QThread):
    pixmapSignal = Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._runFlag = True

    def run(self, parent, nameWeights, fileName):
        model = YOLO(nameWeights) #В дальнейшем модель Yolo, должна запускаться до видео (boxSettings)
        cap = cv2.VideoCapture(fileName)
        if not cap.isOpened():
            print("Ошибка открытия видео файла")
            return
        timerFps = QTimer()
        timerFps.start(30)
        timerFps.timeout.connect(lambda: postVideo(parent, model, cap, self))
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._runFlag = False
        self.wait()