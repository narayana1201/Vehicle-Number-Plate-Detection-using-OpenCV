# Vehicle-Number-Plate-Detection-using-OpenCV
Automatic-Number-Plate-Detection Project helps to Find and allow the Only register Vehicles inside your Campus or Your Private Place.

# Working: 
This Python project employs OpenCV and Haarcascade for automatic number plate detection. Initially, vehicle registration is facilitated through a registration page, storing pertinent information in MongoDB. Subsequently, a live stream monitoring page displays real-time vehicle license plate detections. Each detection triggers a comparison with the database; if a match is found, access is granted; otherwise, a beep alarm is activated.

**OpenCV** serves as the cornerstone for computer vision tasks, enabling image, video, and object detection. Haarcascade, a pre-trained algorithm within OpenCV, specializes in identifying number plates on vehicles. Following detection, 

**EasyOCR** steps in, utilizing a deep learning model to accurately extract alphanumeric characters from the license plate image.

**MongoDB** serves as the repository for all vehicle information, ensuring efficient data management and retrieval throughout the system's operation.

# =>Pre-Install Softwares:
1. Python 3.12
2. cmake
3. vscode
4. node
5. MongoDB

# =>Installation Steps:

Open command Prompt. (windows)
1. create virtualenv
   
command: virtualenv numberplate_detection

2. Activate virtualenv
   
command: numberplate_detection\Scripts\Activate

3. Install Requerements
   
command: pip install -r requerements.txt

4. Run python script
   
command: python main.py

=> It Will start the backend API.

# Start Frontend:

1. open code in vscode
2. npm i
3. npm start

=> It will start the frontend


Any Doubts:
Reach Me: narayananhm123@gmail.com
