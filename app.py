import cv2
import imutils
import time
import logging
import traceback  # Import traceback module
from facial_detections import detectFace
from blink_detection import isBlinking
from mouth_tracking import mouthTrack
from object_detection import detectObject
from eye_tracker import gazeDetection
from head_pose_estimation import head_pose_detection 
import winsound
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)
logger = logging.getLogger(__name__)


global data_record
data_record = []

# For Beeping
frequency = 2500
duration = 1000

# OpenCV videocapture for the webcam
try:
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam.open()
    if not cam.isOpened():
        raise RuntimeError("Failed to open camera")
    logger.info("Camera initialized successfully")
except Exception as e:
    logger.error(f"Camera initialization failed: {str(e)}")
    raise


# Face Count If-else conditions
def faceCount_detection(faceCount):
    if faceCount > 1:
        time.sleep(5)
        remark = "Multiple faces have been detected."
        winsound.Beep(frequency, duration)
    elif faceCount == 0:
        remark = "No face has been detected."
        time.sleep(3)
        winsound.Beep(frequency, duration)
    else:
        remark = "Face detecting properly."
    return remark


# Main function 
def proctoringAlgo():
    blinkCount = 0
    
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                logger.error("Failed to capture frame from camera")
                break

            # frame = imutils.resize(frame, width=450)

            record = []

            # Reading the Current time
            current_time = datetime.now().strftime("%H:%M:%S.%f")
            print("Current Time is:", current_time)
            record.append(current_time)

            # Returns the face count and will detect the face.
            faceCount, faces = detectFace(frame)
            print(faceCount_detection(faceCount))
            record.append(faceCount_detection(faceCount))
            # print(faceCount)

            if faceCount == 1:

                # Blink Detection
                blinkStatus = isBlinking(faces, frame)
                print(blinkStatus[2])

                if blinkStatus[2] == "Blink":
                    blinkCount += 1
                    record.append(blinkStatus[2] + " count: " + str(blinkCount))
                else:
                    record.append(blinkStatus[2])


                # Gaze Detection
                eyeStatus = (gazeDetection(faces, frame))
                print(eyeStatus)
                record.append(eyeStatus)

                # Mouth Position Detection
                print(mouthTrack(faces, frame))
                record.append(mouthTrack(faces, frame))

                # Object detection using YOLO
                objectName = detectObject(frame)
                print(objectName)
                record.append(objectName)

                if len(objectName) > 1:
                    time.sleep(4)
                    winsound.Beep(frequency, duration)
                    continue

                # Head Pose estimation
                print(head_pose_detection(faces, frame))
                record.append(head_pose_detection(faces, frame))
            
            else:
                data_record.append(record)
                continue

            data_record.append(record)
            # eyeStatus = gazeDetection(faces, frame)
            # print(eyeStatus)
            # print(objectName) 

            try:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested to quit")
                    break
            except Exception as e:
                logger.error(f"Error displaying frame: {str(e)}")
                logger.error(traceback.format_exc())
                break
    
    except Exception as e:
        logger.error(f"Error in proctoring algorithm: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        try:
            cam.release()
            cv2.destroyAllWindows()
            logger.info("Camera resources released")
        except Exception as e:
            logger.error(f"Error releasing camera resources: {str(e)}")


if __name__ == '__main__':
    try:
        proctoringAlgo()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())

    # Convert the list to a string with each element on a new line
    activityVal = "\n".join(map(str, data_record))
    # print(activityVal)

    with open('activity.txt', 'w') as file:
        file.write(str(activityVal))