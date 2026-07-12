import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# create dataset file
file = open("gesture_data.csv", "a", newline="")
writer = csv.writer(file)

label = input("Enter gesture label (e.g. 1, 2, thumbs_up): ")

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            data = []

            # extract 21 landmarks
            for lm in handLms.landmark:
                data.append(lm.x)
                data.append(lm.y)
                data.append(lm.z)

            # save label + data
            data.insert(0, label)
            writer.writerow(data)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) == 27:  # ESC to stop
        break

cap.release()
file.close()
cv2.destroyAllWindows()