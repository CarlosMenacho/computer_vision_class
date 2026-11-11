import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()

    if not ok:
        print(f"error en la lectura de imagen")
        break
    ### desde aqui
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    pts = cv2.goodFeaturesToTrack(gray, 
                                  maxCorners=100, 
                                  qualityLevel=0.01, 
                                  minDistance=10)

    if pts is not None:
        for x,y in pts.reshape(-1, 2).astype(int):
            cv2.circle(frame, (x,y), 4, (0,255,0), -1)

    cv2.imshow("camara", frame)
    ### hasta aca - podemos realizar todos los pre-procesamientos que vimos
    if cv2.waitKey(1) & 0xFF == 27:
        # Si presionamos una tecla "esc" cerramos el ciclo
        break
cap.release()
cv2.destroyAllWindows()

