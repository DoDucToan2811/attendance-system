import cv2
import numpy as np
import pandas as pd
import os
import onnxruntime as ort
import insightface
from insightface.utils import face_align
import os
import onnxruntime as ort
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

liveness_session = ort.InferenceSession("liveness.onnx", providers=["CPUExecutionProvider"])
def preprocess_for_liveness_opencv(image, target_size=(224, 224)):
    if image is None:
        print("Image is empty. Cannot preprocess.")
        return None
    try:
        resized = cv2.resize(image, target_size)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_image / 255.0

        # Apply mean subtraction and standard deviation normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std

        # Change dimensions to match the model input (C, H, W)
        preprocessed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        return np.expand_dims(preprocessed, axis=0).astype(np.float32)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None

def detect_liveness(face_image):
    preprocessed = preprocess_for_liveness_opencv(face_image)
    if preprocessed is None:
        print("Skipping liveness detection due to preprocessing error.")
        return 0.0  # Default liveness score for invalid input
    try:
        ort_inputs = {liveness_session.get_inputs()[0].name: preprocessed}
        liveness_score = liveness_session.run(None, ort_inputs)[0][0]
        return np.mean(liveness_score)
    except Exception as e:
        print(f"Error during liveness detection: {e}")
        return 0.0  # Default liveness score for detection failure

def run_realtime_face_recognition(csv_path):
    # Initialize InsightFace detection and recognition models
    detector = insightface.model_zoo.get_model('buffalo_m/det_2.5g.onnx')
    detector.prepare(ctx_id=0, input_size=(640, 640))

    recognizer = insightface.model_zoo.get_model('buffalo_m/w600k_r50.onnx')
    recognizer.prepare(ctx_id=0)

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
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Align and get face embedding
            if kpss is not None:
                aligned_face = face_align.norm_crop(frame, landmark=kpss[i])

                embedding = recognizer.get_feat(aligned_face).flatten()

                # Compare with saved embeddings
                match_index, similarity = compare_embeddings(embedding, saved_embeddings)

                if match_index != -1:
                    label = f"ID: {names[match_index]} ({similarity:.2f})"
                    color = (0, 255, 0)
                    liveness_score = detect_liveness(cropped_face)

                    if liveness_score <= 0.8:
                        # Fake face detected
                        cv2.putText(frame, "Liveness Failed!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                else:
                    label = f"Unknown ({similarity:.2f})"
                    color = (0, 0, 255)

                # Display the label
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


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
    run_realtime_face_recognition(csv_path)
