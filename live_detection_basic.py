import cv2
import numpy as np
import pytesseract
import easyocr
import imutils

# Function to perform number plate detection on a frame
def detect_number_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edge = cv2.Canny(gray, 170, 200)
    cnts, new = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCount = None
    for i in cnts:
        perimeter = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
        if len(approx) == 4:
            NumberPlateCount = approx
            x, y, w, h = cv2.boundingRect(i)
            crp_img = frame[y:y + h, x:x + w]
            cv2.imwrite('temp.png', crp_img)
            break
    return NumberPlateCount

# Function to perform OCR on the extracted number plate region
def perform_ocr(img_path):
    reader = easyocr.Reader(['ch_sim', 'en'])
    text = reader.readtext(img_path, detail=0)
    return text

# Start capturing frames from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to improve processing speed
    frame = imutils.resize(frame, width=500)

    # Perform number plate detection
    number_plate_coords = detect_number_plate(frame)

    if number_plate_coords is not None:
        # Draw bounding rectangle around the detected number plate
        cv2.drawContours(frame, [number_plate_coords], -1, (0, 255, 0), 3)

        # Perform OCR on the extracted number plate region
        detected_text = perform_ocr('temp.png')

        # Convert detected text to string
        detected_text = " ".join(detected_text)

        # Display the detected text on the frame
        cv2.putText(frame, str(detected_text), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Number Plate Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
