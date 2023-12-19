import cv2
import numpy as np
import sqlite3

# Initialize the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Connect to the SQLite database
conn = sqlite3.connect('face_database.db')
c = conn.cursor()

# Create the "People" table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS People (ID INTEGER PRIMARY KEY, NAME TEXT, IMAGE TEXT)''')

# Load the preloaded database of images
dataSet = "image-detection/dataSet"
list_of_files = os.listdir(dataSet)

# Training the face recognizer
for file in list_of_files:
    img = cv2.imread(os.path.join(dataSet, file))
    img = cv2.resize(img, (550, 550), interpolation = cv2.INTER_AREA)
    face_cascade.detectMultiScale(img, scaleFactor = 1.3, faceCascade = face)
        
    for (x, y, w, h) in face:
        gray = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (99, 99), 35)
        labels, confidence = recognizer.predict(gray)
        
        # Save the person's data in the database
        c.execute("INSERT INTO People (NAME, IMAGE) VALUES (?, ?)", (file, str(confidence)))
        conn.commit()

# Function to detect the face
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3, faceCascade = face)
    return gray, faces

# Function to draw the rectangle around the face and print the name
def draw_rectangle(img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        id, confidence = recognizer.predict(cv2.resize(img[y:y+h, x:x+w], (550, 550)))
        
        # Get the person's name from the database
        c.execute("SELECT NAME FROM People WHERE ID = ?", (id,))
        name = c.fetchone()[0]
        
        # Print the name on the image
        cv2.putText(img, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray, faces = detect_face(img)
    draw_rectangle(img, faces)
    cv2.imshow('Face Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
conn.close()