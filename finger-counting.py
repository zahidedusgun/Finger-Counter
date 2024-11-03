import cv2
import mediapipe as mp

print("MediaPipe successfully imported!")

print("Package Imported")

cap = cv2.VideoCapture(0)  # Kamerayı başlatır
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('finger-counting.mp4', fourcc, 20.0, (640, 480))
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]

while cap.isOpened():
    
    success, img = cap.read()
    if not success:
        break  # Eğer görüntü alınamazsa döngüden çık

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            # 0 ID'ye lacivert bir nokta ekle
            if len(lmList) > 0:
                cv2.circle(img, (lmList[0][1], lmList[0][2]), 10, (128, 0, 128), cv2.FILLED)  # Lacivert

    if len(lmList) != 0:
        fingers = []
        
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else: 
                fingers.append(0)
                
        totalF = fingers.count(1)
        
        # Yazıyı kırmızı yap
        cv2.putText(img, str(totalF), (525, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)

    out.write(img)  # Videoya kareyi yaz

    cv2.imshow("Finger Counting", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
