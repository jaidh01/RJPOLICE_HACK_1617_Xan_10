import cv2
import face_recognition
import numpy as np

def face_data():
        # Load known face 1
    known_image1 = face_recognition.load_image_file("jee.jpeg") 
    known_encoding1 = face_recognition.face_encodings(known_image1)[0]
    known_faces.append(known_encoding1)
    known_face_names.append("Person 1")

    # Load known face 2
    known_image2 = face_recognition.load_image_file("jai.jpeg")
    known_encoding2 = face_recognition.face_encodings(known_image2)[0]
    known_faces.append(known_encoding2)
    known_face_names.append("Person 2")

   

known_faces = []
known_face_names = []
roi_coordinates = (100, 100, 300, 300)  # Adjust these values according to your requirement
x, y, w, h = roi_coordinates

face_data()

font = cv2.FONT_HERSHEY_DUPLEX

video_capture = cv2.VideoCapture(0)  # Assuming your camera index is 0

while True:
    ret, frame = video_capture.read()

    # Extract the ROI from the frame
    roi_frame = frame[y:y+h, x:x+w]
    roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
    try:
        # Find face locations within the ROI
        face_locations = face_recognition.face_locations(roi_frame, number_of_times_to_upsample=1)

        if face_locations:
            # If faces are detected, perform face encodings
            face_encodings = face_recognition.face_encodings(roi_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Adjust the coordinates based on the ROI
                top += y
                right += x
                bottom += y
                left += x

                # Compare the current face with all known faces
                matches = face_recognition.compare_faces(known_faces, face_encoding)

                name = "Unknown"

                # If a match is found, use the name of the first matching known face
                if True in matches:
                    first_match_index = matches.index(True)

                    # Ensure index is within the bounds of known_face_names
                    if 0 <= first_match_index < len(known_face_names):
                        name = known_face_names[first_match_index]
                        print(f"Face detected: {name}")
                    else:
                        print("Error: Index out of bounds in known_face_names")

                # Draw a rectangle around the face and display the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    except Exception as e:
        print(f"Error during face recognition: {e}")

    cv2.imshow('Video', roi_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()