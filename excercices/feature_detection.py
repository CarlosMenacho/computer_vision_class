import cv2
from utils import cvlog

cvlog.start("ejemplo")

ref = cv2.imread("gatito.jpeg", cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(ref, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck= True)

for frame in cvlog.frames(0):

    print(frame.shape)
    print(type(frame))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray, None)


    cvlog.image("image",frame)
    cvlog.image("image/gris", gray)

    cvlog.keypoints("image/keypoints", kp2)

    if des2 is not None:
        matches = bf.match(des1, des2)
        cvlog.matches("matches", ref, kp1, gray, kp2, matches, max_matches=50 )



    

