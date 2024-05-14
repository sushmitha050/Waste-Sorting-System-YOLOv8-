import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import tkinter as tk
from tkinter import filedialog

model = YOLO('best_.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

def process_image():
    image_path = filedialog.askopenfilename()
    if image_path:
        frame = cv2.imread(image_path)

        my_file = open("coco1.txt", "r")
        data = my_file.read()
        class_list = data.split("\n")

        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            conf = row[4]
            d = int(row[5])
            c = class_list[d]
            if 'plastic' in c:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cvzone.putTextRect(frame, f'{c} {conf:.2f}', (x1, y1), 1, 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cvzone.putTextRect(frame, f'{c} {conf:.2f}', (x1, y1), 1, 1)

        cv2.imshow("RGB", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def webcam_interface():
    cap = cv2.VideoCapture(0)
    window_name = "Webcam"
    cv2.namedWindow(window_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        my_file = open("coco1.txt", "r")
        data = my_file.read()
        class_list = data.split("\n")

        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            conf = row[4]
            d = int(row[5])
            c = class_list[d]
            if 'plastic' in c:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cvzone.putTextRect(frame, f'{c} {conf:.2f}', (x1, y1), 1, 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cvzone.putTextRect(frame, f'{c} {conf:.2f}', (x1, y1), 1, 1)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyWindow(window_name)

def quit_webcam():
    cv2.destroyAllWindows()

# Create the Tkinter GUI
root = tk.Tk()
root.title("Object Detection")

# Set up styles
button_style = {'font': ('Arial', 12), 'bg': '#4CAF50', 'fg': 'white', 'padx': 10, 'pady': 5, 'borderwidth': 2, 'relief': 'raised'}
label_style = {'font': ('Arial', 12), 'bg': '#F0F0F0', 'fg': '#333333', 'padx': 10, 'pady': 5, 'borderwidth': 2, 'relief': 'ridge'}

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=process_image, **button_style)
select_button.pack(pady=10)

# Create a button to start webcam interface
webcam_button = tk.Button(root, text="Webcam Interface", command=webcam_interface, **button_style)
webcam_button.pack(pady=10)

# Create a label for instructions
instructions_label = tk.Label(root, text="Hover over the image to see RGB values.", **label_style)
instructions_label.pack()

# Set up OpenCV mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

root.mainloop()
