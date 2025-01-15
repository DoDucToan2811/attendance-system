import cv2
import numpy as np
import pandas as pd
import insightface
from insightface.utils import face_align
import os
import onnxruntime as ort
from picamera2 import Picamera2
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import time
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

checked = set()

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = "smart-445016-ead681048f50.json"  # Replace with your downloaded JSON key file
SHEET_NAME = "VoBichHien"  # Replace with your Google Sheet name
MODEL_DIR = "./resources/anti_spoof_models"  # Path to liveness models
DEVICE_ID = 0  # Use 0 for CPU

# Authenticate and connect to Google Sheets
credentials = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, SCOPE)
client = gspread.authorize(credentials)
sheet = client.open(SHEET_NAME).sheet1  # Open the first sheet

def check_and_switch_sheet(client, workbook_name, room_id, current_datetime):
    """
    Check Room_ID, Date, and Hour Range in Sheet1 and switch to the corresponding sheet.
    """
    try:
        # Open the workbook and Sheet1
        workbook = client.open(workbook_name)
        sheet1 = workbook.sheet1  # Open the first sheet (Sheet1)

        # Fetch all data from Sheet1
        data = sheet1.get_all_records()

        # Format the current date as DD/MM/YYYY
        current_date = current_datetime.strftime("%d/%m/%Y")
        # Get the current hour and minute as HH:MM
        current_time = current_datetime.strftime("%H:%M")

        # Iterate through the rows in Sheet1
        for row in data:
            if str(row["Room_ID"]) == str(room_id) and str(row["Date"]) == current_date:
                # Check if the current time falls within the Hour Range
                hour_range = row["Hour Range"]
                start_time, end_time = hour_range.split("-")
                if start_time <= current_time <= end_time:
                    # If match found, get the Subject
                    subject_sheet_name = row["Subject"]
                    try:
                        # Switch to the sheet with the name of the Subject
                        subject_sheet = workbook.worksheet(subject_sheet_name)
                        print(f"Switched to sheet: {subject_sheet_name}")
                        return subject_sheet
                    except gspread.exceptions.WorksheetNotFound:
                        print(f"Worksheet '{subject_sheet_name}' not found.")
                        return None

        # No matching Room_ID, Date, or Hour Range found
        print("No matching Room_ID, Date, and Hour Range found in Sheet1.")
        return None

    except Exception as e:
        print(f"Error while switching sheet: {e}")
        return None

# Get the real-time current date in DD/MM/YYYY format
# Get the real-time current datetime
current_datetime = datetime.now()
room_id_input = "108"  # Replace with the Room_ID you want to check

# Call the function to select the correct sheet
subject_sheet = check_and_switch_sheet(client, SHEET_NAME, room_id_input, current_datetime)

if not subject_sheet:
    print("Exiting due to no matching sheet.")
    exit()

def update_attendance(sheet, student_id):
    """Update attendance for a student in Google Sheets based on the current date."""
    current_date = datetime.now().strftime("%d/%m/%Y")  # Format: DD/MM/YYYY
    try:
        # Get all rows and headers from the sheet
        records = sheet.get_all_records()
        headers = sheet.row_values(1)  # First row as headers
        session_column = None

        # Find or create the session column for the current date
        for idx, header in enumerate(headers):
            if current_date in header:  # Match the current date in column headers
                session_column = idx + 1  # Column index in Google Sheets
                break

        # If no column exists for the current date, create a new one
        if not session_column:
            next_column = len(headers) + 1
            sheet.update_cell(1, next_column, f"SS{len(headers) - 3}: {current_date}")  # Add new column
            session_column = next_column

        # Update attendance for the student
        row_found = False
        for idx, record in enumerate(records, start=2):  # Start=2 (skip header)
            sheet_student_id = str(record.get("Student ID")).strip()
            if sheet_student_id == str(student_id).strip():
                cell_value = sheet.cell(idx, session_column).value
                if cell_value != "✅":  # Avoid duplicate marking
                    sheet.update_cell(idx, session_column, "✅")
                    print(f"Attendance updated for {student_id} on {current_date}")
                #else:
                    #print(f"Attendance already marked for {student_id} on {current_date}")
                row_found = True
                break

        if not row_found:
            print(f"Student ID {student_id} not found in Google Sheets.")

    except Exception as e:
        print(f"Error updating Google Sheets: {e}")
        
# Load embeddings from CSV
def load_embeddings(csv_path):
    if not os.path.exists(csv_path):

        return {}, []

    df = pd.read_csv(csv_path)
    names = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values.astype(np.float32)
    return names, embeddings

# Compare embeddings with cosine similarity
def compare_embeddings(embedding, saved_embeddings, threshold=0.6):
    similarities = np.dot(saved_embeddings, embedding) / (
        np.linalg.norm(saved_embeddings, axis=1) * np.linalg.norm(embedding) + 1e-6
    )
    max_index = np.argmax(similarities)
    max_similarity = similarities[max_index]

    if max_similarity > threshold:
        return max_index, max_similarity
    return -1, max_similarity

liveness_session = ort.InferenceSession("liveness.onnx", providers=["CPUExecutionProvider"])

def detect_liveness(frame, model_test, image_cropper, model_dir):
    image_bbox = model_test.get_bbox(frame)
    if image_bbox is None:
        return None, "No face detected", (0, 0, 255)

    prediction = np.zeros((1, 3))
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))

    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    return image_bbox, f"Real Face: {value:.2f}" if label == 1 else f"Fake Face: {value:.2f}", (0, 255, 0) if label == 1 else (0, 0, 255)


def run_realtime_face_recognition(csv_path):
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"}))  # (640, 480) or (1280, 720)
    picam2.start()
    picam2.set_controls({"AfMode": 2})

    model_test = AntiSpoofPredict(DEVICE_ID)
    image_cropper = CropImage()

    # Initialize InsightFace detection and recognition models
    detector = insightface.model_zoo.get_model('buffalo_sc/det_500m.onnx')
    detector.prepare(ctx_id=0, input_size=(640, 640))

    recognizer = insightface.model_zoo.get_model('buffalo_m/w600k_r50.onnx')
    recognizer.prepare(ctx_id=0)

    # Load known embeddings
    names, saved_embeddings = load_embeddings(csv_path)

    # Initialize MediaPipe Face Detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    prev_time = 0
    frame_count = 0
    while True:
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = time.time()
        frame_count += 1

        # Use MediaPipe for face detection
        results = face_detection.process(frame_rgb)
		
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1, y1, w, h = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih),
                )
                x2, y2 = x1 + w, y1 + h

                # Draw bounding box using MediaPipe
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Crop the face for InsightFace processing
                cropped_face = frame[y1:y2, x1:x2]

                # Perform InsightFace face detection and recognition
                det, kpss = detector.detect(frame, input_size=(640, 640))
                if det.shape[0] > 0:
                    for i in range(det.shape[0]):
                        bbox = det[i, :4].astype(int)
                        score = det[i, 4]
                        aligned_face = face_align.norm_crop(frame, landmark=kpss[i])

                        embedding = recognizer.get_feat(aligned_face).flatten()

                        # Compare with saved embeddings
                        match_index, similarity = compare_embeddings(embedding, saved_embeddings)

                        if match_index != -1:
                            label = f"ID: {names[match_index]} ({similarity:.2f})"
                            color = (0, 255, 0)
                            if names[match_index] not in checked:
                                image_bbox, liveness_result, liveness_color = detect_liveness(
                                    frame, model_test, image_cropper, MODEL_DIR
                                )
                                if liveness_result.startswith("Real"):
                                    # Mark attendance and add to checked list
                                    update_attendance(subject_sheet, names[match_index])
                                    checked.add(names[match_index])
                        else:
                            label = f"Unknown ({similarity:.2f})"
                            color = (0, 0, 255)

                        # Display the label
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        fps = 1 / (current_time - prev_time) if prev_time else 0
        prev_time = current_time
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Show the frame
        cv2.imshow("Face Recognition with MediaPipe", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    face_detection.close()
    
if __name__ == "__main__":
    csv_path = "embeddings.csv"  # Path to your CSV file
    run_realtime_face_recognition(csv_path)
