# Face Recognition and Attendance System

This project implements a **real-time face recognition and attendance tracking system** with liveness detection. It ensures accurate identification of individuals and prevents spoofing attacks.

## Features

- **Real-time face detection and recognition**
- **Liveness detection** to distinguish real faces from spoofed ones
- Integration with **Google Sheets** to update attendance
- Support for adding new face embeddings dynamically
- CSV-based storage of face embeddings
- Continuous recognition with a timer to prevent duplicate attendance entries

## Prerequisites

### Software Requirements
- Python 3.7+
- Google Cloud service account credentials for accessing Google Sheets

### Python Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Required Files
- `requirements.txt`: List of required Python dependencies.
- `smart-445016-ead681048f50.json`: Google Cloud service account credentials.
- `embeddings.csv`: File to store face embeddings and IDs.
- Models for liveness detection (e.g., `2.7_80x80_MiniFASNetV2.pth`, `4_0_0_80x80_MiniFASNetV1SE.pth`).
- InsightFace model files:
  - `1k3d68.onnx` and `w600k_r50.onnx`: Download from the [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo) (in case of not already in `buffalo_m` folder) and place them in the `buffalo_m/` folder.
  - Additional REST API resources: [InsightFace-REST](https://github.com/SthPhoenix/InsightFace-REST).
  - Alternative download link for `1k3d68.onnx`: [Hugging Face](https://huggingface.co/DIAMONIK7777/antelopev2/blob/main/1k3d68.onnx).

**Note**: If you encounter issues downloading files due to Git LFS bandwidth limitations, please use the alternative links provided above.

## Project Structure

- **`main.py`**: The main script for real-time face recognition and attendance tracking.
- **`resources/anti_spoof_models`**: Directory containing models for liveness detection.
- **`embeddings.csv`**: CSV file storing face embeddings and IDs.
- **`smart-445016-ead681048f50.json`**: Google Cloud service account credentials (replace with your file).

## Setup

### 1. Configure Google Sheets

1. **Create a New Google Sheet**:
   - Open Google Sheets and create a new spreadsheet.
   - Set up the headers in the first row with the following structure:
     - Column A: `Room_ID` - Represents the ID of the room (e.g., 108).
     - Column B: `Date` - Specifies the session's date in the format `DD/MM/YYYY`.
     - Column C: `Hour Range` - Indicates the session's start and end time (e.g., `08:00-12:00`).
     - Column D: `Subject` - Specifies the subject or instructor name (e.g., `Prof.Hien`, `Prof.Son`).

   Example layout:
   ![Sheet1 Example](Sheet1.png)

2. **Add Additional Sheets for Attendance**:
   - Create another sheet (e.g., `Attendance`) with the following structure:
     - Column A: `Student ID` - The unique identifier for each student.
     - Column B: `Major` - The field of study (e.g., `CSE`, `ECE`).
     - Column C: `Intake Year` - The year the student joined (e.g., 2021, 2023).
     - Column D: `Image` - A marker (e.g., `x`) indicating the availability of the student's photo.
     - Columns E onward: Dates in the format `DD/MM/YYYY` to track attendance, where a ✅ indicates the student was present.

   Example layout:
   ![Attendance Example](Result_sheet.png)

3. **Set Up Google Sheets API**:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/).
   - Create a new project (or select an existing one).
   - Navigate to **APIs & Services > Library** and enable the **Google Sheets API** and **Google Drive API**.
   - Go to **APIs & Services > Credentials**, then click **Create Credentials > Service Account**.
   - Fill in the required details and create a service account.
   - Once the service account is created, go to the **Keys** tab and generate a new JSON key file.
   - Download the JSON key file and rename it to `smart-445016-ead681048f50.json` (or update the name in your script).
   - Share your Google Sheet with the **service account email** (found in the JSON key file) and grant **Editor** permissions.

4. **Update the Script**:
   - Replace the `SHEET_NAME` variable in `main.py` with the name of your Google Sheet:
     ```python
     SHEET_NAME = "YourGoogleSheetName"
     ```
   - Replace the `JSON_CREDENTIALS` variable with the path to your JSON key file:
     ```python
     JSON_CREDENTIALS = "path/to/your/json/file.json"
     ```
   - Verify the connection by running the script. It will prompt for selecting the correct sheet if multiple exist.

5. **Verify Formatting**:
   - Ensure each sheet matches the specified format. Attendance marking (✅ for present, blank for absent) should align with the attendance sheet's layout.
   - Ensure the `Room_ID`, `Date`, and `Hour Range` columns in `Sheet1` align with the session schedule.

### 2. Install Python Dependencies
Install the required Python packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 3. Configure Face Models
1. Download InsightFace models and place them in a folder (`buffalo_m/`):
   - **`det_2.5.onnx`**: Face detection model.
   - **`w600k_r50.onnx`**: Face recognition model.
   - Download from the [InsightFace Model Zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo).
   - Additional REST API resources: [InsightFace-REST](https://github.com/SthPhoenix/InsightFace-REST).
   - Alternative download link for `1k3d68.onnx`: [Hugging Face](https://huggingface.co/DIAMONIK7777/antelopev2/blob/main/1k3d68.onnx).
2. Ensure the `buffalo_m/` folder is in the same directory as `main.py`.

### 4. Run the Script
Run the script to start real-time face recognition:
```bash
python main.py
```

## How It Works

1. **Face Detection**:
   - Detects faces using the InsightFace detection model.
2. **Face Recognition**:
   - Matches detected faces with saved embeddings in `embeddings.csv`.
3. **Liveness Detection**:
   - Liveness detection prevents spoofing attacks by analyzing facial features to ensure the face is real and not a printed photo or video replay. Models such as MiniFASNet are used to process the cropped face image and predict whether the face is genuine. This ensures higher accuracy in attendance tracking.
4. **Attendance Update**:
   - Marks attendance for recognized individuals in Google Sheets if they are continuously recognized for 3 seconds.

## Key Functions

### `choose_google_sheet(client, workbook_name)`
Prompts the user to select a sheet (class) from a Google Sheets workbook.

### `update_attendance(student_id)`
Updates attendance for a specific `student_id` based on the current date in Google Sheets.

### `load_embeddings(csv_path)`
Loads face embeddings and corresponding IDs from the CSV file.

### `compare_embeddings(embedding, saved_embeddings, threshold=0.5)`
Compares a face embedding with saved embeddings using cosine similarity.

### `detect_liveness(frame, model_test, image_cropper, model_dir)`
Performs liveness detection on a detected face using anti-spoofing models.

### `run_realtime_face_recognition(csv_path, model_dir, device_id)`
Main function to run real-time face recognition and attendance tracking.

## Usage

- **Press `q`**: Exit the application.
- **Press `s`**: Save new face embeddings dynamically by entering an ID.

## Beginner's Manual

### Step-by-Step Guide

1. **Install Python**
   - Ensure Python 3.7+ is installed. Download it from [Python's official site](https://www.python.org/downloads/).

2. **Set Up the Environment**
   - Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Prepare the Google Sheet**
   - Create a Google Sheet.
   - Share it with the service account email provided in `smart-445016-ead681048f50.json`.
   - Update the `SHEET_NAME` variable in `main.py` to match your Google Sheet's name.

4. **Download the Models**
   - Place all required models in the appropriate folders.

5. **Run the Script**
   - Start the program:
     ```bash
     python main.py
     ```
   - Follow on-screen instructions to manage embeddings or update attendance.

6. **Test the System**
   - Use a webcam to recognize faces and update attendance.

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface): Open-source 2D and 3D deep face analysis toolbox.
- [Google Sheets API](https://developers.google.com/sheets/api): For attendance tracking.
- [OpenCV](https://opencv.org/): For real-time video processing.
- [InsightFace-REST](https://github.com/SthPhoenix/InsightFace-REST): REST API for efficient face recognition and analysis.
- [Hugging Face](https://huggingface.co/DIAMONIK7777/antelopev2/blob/main/1k3d68.onnx): Alternative source for downloading `1k3d68.onnx`.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

