import os
import cv2
import numpy as np
import pandas as pd
import insightface
from insightface.utils import face_align
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import gspread
import time
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = "smart-445016-ead681048f50.json"  # Replace with your downloaded JSON key file
SHEET_NAME = "VoBichHien"  # Replace with your Google Sheet name

# Authenticate and connect to Google Sheets
credentials = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, SCOPE)
client = gspread.authorize(credentials)
sheet = client.open(SHEET_NAME).sheet1  # Open the first sheet

def choose_google_sheet(client, workbook_name):
    """Prompt the user to choose a sheet (class) from the Google Sheets file."""
    try:
        # Open the workbook
        workbook = client.open(workbook_name)
        sheets = workbook.worksheets()  # List all sheets in the workbook
        
        # Display available sheet names
        print("Available sheets (classes):")
        for idx, sheet in enumerate(sheets):
            print(f"{idx + 1}. {sheet.title}")
        
        # Prompt the user to choose a sheet
        while True:
            try:
                choice = int(input("Enter the number of the sheet to use: "))
                if 1 <= choice <= len(sheets):
                    chosen_sheet = sheets[choice - 1]
                    print(f"Selected sheet: {chosen_sheet.title}")
                    return chosen_sheet
                else:
                    print("Invalid choice. Please select a valid sheet number.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    except Exception as e:
        print(f"Error selecting sheet: {e}")
        return None

# Call the choose_google_sheet function to select the sheet
sheet = choose_google_sheet(client, SHEET_NAME)
if not sheet:
    print("Exiting due to sheet selection error.")
    exit()


def update_attendance(student_id):
    """Update attendance for a student in Google Sheets based on the current date."""
    current_date = datetime.now().strftime("%d/%m/%Y")  # Format: 18/12/2024
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
                else:
                    print(f"Attendance already marked for {student_id} on {current_date}")
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
def compare_embeddings(embedding, saved_embeddings, threshold=0.5):
    similarities = np.dot(saved_embeddings, embedding) / (
        np.linalg.norm(saved_embeddings, axis=1) * np.linalg.norm(embedding) + 1e-6
    )
    max_index = np.argmax(similarities)
    max_similarity = similarities[max_index]

    if max_similarity > threshold:
        return max_index, max_similarity
    return -1, max_similarity

# Liveness detection setup
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
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))

    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        return image_bbox, f"Real Face: {value:.2f}", (0, 255, 0)
    else:
        return image_bbox, f"Fake Face: {value:.2f}", (0, 0, 255)


def run_realtime_face_recognition(csv_path, model_dir, device_id):
    # Initialize InsightFace detection and recognition models
    detector = insightface.model_zoo.get_model('buffalo_m/det_2.5g.onnx')
    detector.prepare(ctx_id=0, input_size=(640, 640))

    recognizer = insightface.model_zoo.get_model('buffalo_m/w600k_r50.onnx')
    recognizer.prepare(ctx_id=0)

    # Initialize liveness detection models
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    # Load known embeddings
    names, saved_embeddings = load_embeddings(csv_path)
    if len(names) == 0:
        print("No embeddings found. Please generate embeddings first.")
        return

    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Cannot access the webcam.")
        return

    print("Press 'q' to quit. Press 's' to save new embeddings.")

    recognized_start_time = None
    recognized_name = None
    success_message_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Perform face detection
        det, kpss = detector.detect(frame, input_size=(640, 640))

        for i in range(det.shape[0]):
            bbox = det[i, :4].astype(int)
            score = det[i, 4]
            x1, y1, x2, y2 = bbox
            cropped_face = frame[y1:y2, x1:x2]

            # Align and get face embedding
            if kpss is not None:
                aligned_face = face_align.norm_crop(frame, landmark=kpss[i])
                embedding = recognizer.get_feat(aligned_face).flatten()

                # Compare with saved embeddings
                match_index, similarity = compare_embeddings(embedding, saved_embeddings)

                # Perform liveness detection
                image_bbox, liveness_result, liveness_color = detect_liveness(frame, model_test, image_cropper, model_dir)

                if match_index != -1 and liveness_result.startswith("Real"):
                    # Match found, valid real face
                    label = f"ID: {names[match_index]} ({similarity:.2f}) - {liveness_result}"
                    color = (0, 255, 0)

                    # Start timing for continuous recognition
                    if recognized_name == names[match_index]:
                        if time.time() - recognized_start_time >= 3:
                            # Update attendance if recognized for 3 seconds
                            update_attendance(names[match_index])
                            recognized_start_time = None
                            recognized_name = None
                            success_message_time = time.time()
                            print(f"Attendance updated for {names[match_index]}.")
                    else:
                        recognized_name = names[match_index]
                        recognized_start_time = time.time()
                else:
                    # Reset if recognition is interrupted
                    recognized_name = None
                    recognized_start_time = None
                    label = f"Unknown ({similarity:.2f}) - {liveness_result}"
                    color = (0, 0, 255)

                # Display the label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw a rectangle around the detected face
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Show success message if attendance was updated
        if success_message_time and time.time() - success_message_time <= 2:
            cv2.putText(frame, "Recognize Success", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Show the frame
        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # Save new embedding on 's'
        if key == ord('s'):
            if det.shape[0] > 0:
                print("Saving new embedding...")
                new_id = input("Enter the ID for this face: ")
                with open(csv_path, 'a') as f:
                    f.write(f"{new_id}," + ','.join(map(str, embedding)) + '\n')
                names, saved_embeddings = load_embeddings(csv_path)
                print("Embedding saved!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    csv_path = "embeddings.csv"  # Path to your CSV file
    model_dir = "./resources/anti_spoof_models"  # Path to liveness models
    device_id = 0  # Use 0 for CPU
    run_realtime_face_recognition(csv_path, model_dir, device_id)
