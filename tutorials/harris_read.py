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

    # Harris 
    R = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    frame[R > 0.01 * R.max()] = (0,0,255) # al color rojo 

    for y, x in np.argwhere(R > 0.01 * R.max()):
        cv2.circle(frame, (int(x), int(y)), 4, (0,0,255), 1)

    cv2.imshow("camara", frame)
    # cv2.imshow("imagen en gris", gray)
    ### hasta aca - podemos realizar todos los pre-procesamientos que vimos
    if cv2.waitKey(1) & 0xFF == 27:
        # Si presionamos una tecla "esc" cerramos el ciclo
        break
cap.release()
cv2.destroyAllWindows()

