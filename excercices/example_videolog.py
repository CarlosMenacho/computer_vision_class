import cv2
from utils import cvlog

cvlog.start("ejemplo")

for frame in cvlog.frames(0):

    print(frame.shape)
    print(type(frame))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cvlog.image("image/frame",frame)
    cvlog.image("image/gris", gray)

    

