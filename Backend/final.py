from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi import FastAPI, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.errors import DuplicateKeyError
from passlib.context import CryptContext
from pydantic import BaseModel
from pymongo import MongoClient
import numpy as np
import imutils
import easyocr
import winsound
import threading
import shutil
import cv2
import os

app = FastAPI()
router = APIRouter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

client = MongoClient("mongodb://localhost:27017/")
db = client["vehicle_database"]
collection = db["registered_vehicles"]
login_db = db["login_details"]

######################### login #############################################
# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(BaseModel):
    username: str
    password: str


# Route for user registration
@app.post("/register")
async def register(user: User):
    hashed_password = pwd_context.hash(user.password)
    try:
        login_db.insert_one(
            {"username": user.username, "password": hashed_password}
        )
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Username already exists")
    return {"message": "User registered successfully"}


# Route for user login
@app.post("/login")
async def login(user: User):
    stored_user = login_db.find_one({"username": user.username})
    if stored_user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not pwd_context.verify(user.password, stored_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"message": "Login successful"}


############################# DataBase #####################################

def insert_vehicle_details(owner_name: str, vehicle_number: str, phone_number: str):
    print("vehicle_number: ", vehicle_number)
    vehicle_number_insert = vehicle_number.replace(" ", "")
    print("vehicle_number_insert: ", vehicle_number_insert)
    vehicle_data = {
        "owner_name": owner_name,
        "vehicle_number": vehicle_number_insert,
        "phone_number": phone_number
    }
    result = collection.insert_one(vehicle_data)
    print("Vehicle details inserted successfully.")
    return result

def check_vehicle_registration(vehicle_number):
    # print("vehicle_number: ",vehicle_number)
    vehicle_register_number = vehicle_number.replace(" ", "")
    print("vehicle_register_number: ", vehicle_register_number)
    query = {"vehicle_number": vehicle_register_number}
    result = collection.find_one(query)
    return result is not None

############ Register From Image ###############
crop_folder = "cropped_images"

def perform_ocr(img_path):
    reader = easyocr.Reader(['ch_sim', 'en'])
    text = reader.readtext(img_path, detail=0)
    print("Detected Number: ", text)
    return text

def register_from_image(newimage):
    image = cv2.imread(newimage)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edge = cv2.Canny(gray, 170, 200)
    cnts, new = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image1 = image.copy()
    cv2.drawContours(image1, cnts, -1, (0, 225, 0), 3)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCount = None
    image2 = image.copy()
    cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
    count = 0
    name = 1
    for i in cnts:
        perimeter = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
        if len(approx) == 4:
            NumberPlateCount = approx
            x, y, w, h = cv2.boundingRect(i)
            crp_img = image[y:y + h, x:x + w]
            crp_img_loc = os.path.join(crop_folder, f"{name}.png")
            cv2.imwrite(crp_img_loc, crp_img)
            name += 1
            extracted_number = perform_ocr(crp_img_loc)
            print("Extracted_number: ", extracted_number)
            return extracted_number

  
####################### Live Detection ######################
is_running = False
latest_messages = []

def detect_number_plate(frame):
    plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in plates:
        plate_img = frame[y:y+h, x:x+w]
        cv2.imwrite('temp.png', plate_img)
        return [(x, y, w, h)]  
    return None


def capture_frames():
    global is_running, latest_messages
    cap = cv2.VideoCapture(0)  # camera value 0 is default camera. 1 is external camera.
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for the video writer
    out = cv2.VideoWriter('recorded_videos/output.mp4', fourcc, 20.0, (640, 480))
    while is_running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to receive frame.")
            break
        frame = imutils.resize(frame, width=500)
        number_plate_coords = detect_number_plate(frame)
        if number_plate_coords is not None:
            for (x, y, w, h) in number_plate_coords:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            detected_text = perform_ocr('temp.png')
            detected_text = " ".join(detected_text)

            # Check if the detected number is not empty
            if detected_text:
                if check_vehicle_registration(detected_text):
                    print("Your vehicle is allowed")
                    message = "Your vehicle is allowed"
                else:
                    message = "You are not allowed"
                    print("You are not allowed")
                    winsound.Beep(1000, 500)
            else:
                message = "Number plate not detected"
                print("Number plate not detected")
                winsound.Beep(1000, 500)

            latest_messages.append(message)
            cv2.putText(frame, str(detected_text), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Number Plate Detection', frame)
        out.write(frame)  # Write the frame to the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()  # Release the video writer
    cv2.destroyAllWindows()


def start_capture():
    global is_running, latest_messages
    is_running = True
    latest_messages.clear()
    threading.Thread(target=capture_frames).start()

def stop_capture():
    global is_running
    is_running = False

############### Image Detection ####################
plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 500

def save_plate(img, roi, count):
    cv2.imwrite(f"New/sample.jpg", roi)
    cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, "Plate Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    # cv2.imshow("Result", img)
    # cv2.waitKey(0)

def extract_plate_number(img):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img)
    plate_text = ''
    for detection in result:
        plate_text += detection[1] + ' '
    return plate_text.strip()

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
            if not plate_text:
                print("Number plate not detected")
                return "Number plate not detected"
            save_plate(img, imgRoi, count)
            count += 1
            if check_vehicle_registration(plate_text):
                print(f"You are allowed. Your Vehicle NO: {plate_text}")
                return f"You are allowed. Your Vehicle NO: {plate_text}"
            else:
                print(f"You are not allowed. Your Vehicle NO: {plate_text}")
                return f"You are not allowed. Your Vehicle NO: {plate_text}"
    print("Number plate not detected")
    return "Number plate not detected"

################### API Routes ##############################
class VehicleRegistrationRequest(BaseModel):
    owner_name: str
    vehicle_number: str
    phone_number: str

@app.post("/register-vehicle/")
async def register_vehicle(vehicle_data: VehicleRegistrationRequest):
    owner_name = vehicle_data.owner_name
    vehicle_number = vehicle_data.vehicle_number
    phone_number = vehicle_data.phone_number

    result = insert_vehicle_details(owner_name, vehicle_number, phone_number)
    
    if result.acknowledged:
        return JSONResponse(content={"message": "Vehicle registered successfully"})
    else:
        return JSONResponse(content={"message": "Failed to register vehicle"})
    
@app.post("/register-vehicle-from-image/")
async def register_image(owner_name: str = Form(...), phone_number: str = Form(...), image: UploadFile = File(...)):
    try:
        upload_folder = "uploaded_images"
        os.makedirs(upload_folder, exist_ok=True)
        file_location = os.path.join(upload_folder, image.filename)
        with open(file_location, "wb") as file_object:
            shutil.copyfileobj(image.file, file_object)
        register_number = register_from_image(file_location)
        insert_vehicle_details(owner_name, register_number[0], phone_number)
        return JSONResponse(content={"message": "Registered Successfully"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "Failed to register vehicle, Number is not clear", "detail": str(e)})

@app.post("/detect-number-plate/")
async def detect_number_plate_from_image(file: UploadFile = File(...)):
    with open(f"uploaded_images/{file.filename}", "wb") as buffer:
        buffer.write(file.file.read())
    results = detect_number_plate_image(f"uploaded_images/{file.filename}")
    print("Result: ",results)
    return {"messages": results}


@app.post("/start")
async def start_detection(background_tasks: BackgroundTasks):
    start_capture()
    return JSONResponse(content={"message": "Detection started"}, status_code=200)

@app.post("/stop")
async def stop_detection():
    stop_capture()
    return JSONResponse(content={"message": "Detection stopped"}, status_code=200)

@app.get("/latest-messages")
async def get_latest_messages():
    global latest_messages
    return {"messages": latest_messages}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)