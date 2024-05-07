import cv2
import easyocr
import numpy as np
from pymongo import MongoClient

plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 500

# MongoDB configuration
client = MongoClient("mongodb://localhost:27017/")
db = client["vehicle_database"]
collection = db["registered_vehicles"]

def check_vehicle_registration(vehicle_number):
    query = {"vehicle_number": vehicle_number}
    result = collection.find_one(query)
    return result is not None

def save_plate(img, roi, count):
    cv2.imwrite(f"New/sample.jpg", roi)
    cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, "Plate Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow("Result", img)
    cv2.waitKey(500)

def extract_plate_number(img):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    plate_text = ''
    for detection in result:
        plate_text += detection[1] + ' '
    return plate_text.strip()
img_path = 'Dataset/7.jpeg' 

def detect_number_plate_image(sampleimage):
    img = cv2.imread(sampleimage)
    if img is None:
        print("Could not read the image file")
        exit()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)
    count = 0
    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "NumberPlate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y+h, x:x+w]
            plate_text = extract_plate_number(imgRoi)
            print("Detected Plate Text:", plate_text)
            save_plate(img, imgRoi, count)
            count += 1
            if check_vehicle_registration(plate_text):
                print(f"You are allowed. Your Vehicle NO: {plate_text}")
                result = f"You are allowed. Your Vehicle NO: {plate_text}"
            else:
                print(f"You are not allowed. Your Vehicle NO: {plate_text}")
                result = f"You are not allowed. Your Vehicle NO: {plate_text}"
    return result

detect_number_plate_image(img_path)

# cv2.imshow("Result", img)
# cv2.waitKey(0)
