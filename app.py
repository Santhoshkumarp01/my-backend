import os
import subprocess
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed_videos'

# Load the trained model
model = tf.keras.models.load_model('hammer_throw_model.h5')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define the face landmark indices (0 to 10)
face_landmarks_indices = set(range(0, 11))

# Function to generate detailed feedback based on predicted phases
def generate_feedback(predictions):
    feedback = []
    annotations = []

    for i, phase in enumerate(predictions):
        if phase == 0:
            feedback.append(f"Starting Position (Turn {i + 1}):")
            feedback.append("1. Adjust your starting position for better balance.")
            feedback.append("2. Maintain a low center of gravity at the start.")
            feedback.append("3. Ensure your feet are shoulder-width apart.")
            annotations.append(("Circle", (320, 240), 50))
            annotations.append(("Text", (50, 50), "Keep balance"))

        elif phase == 1:
            feedback.append(f"Wind-Up Phase (Turn {i + 1}):")
            feedback.append("1. Improve arm extension during the wind-up.")
            feedback.append("2. Focus on controlling the hammer's rotation.")
            feedback.append("3. Ensure your shoulders are aligned with the hammer.")
            annotations.append(("Arrow", (300, 200), (350, 250)))
            annotations.append(("Line", (200, 150), (400, 150)))

        elif phase == 2:
            feedback.append(f"Turn Phase (Turn {i + 1}):")
            feedback.append("1. Maintain stability during the turns.")
            feedback.append("2. Pivot smoothly on your feet to avoid imbalance.")
            feedback.append("3. Keep your head steady and focus forward.")
            annotations.append(("Line", (200, 300), (400, 300)))
            annotations.append(("Rectangle", (150, 200), (300, 250)))

        elif phase == 3:
            feedback.append(f"Release Phase (Turn {i + 1}):")
            feedback.append("1. Align your body with the hammer's trajectory.")
            feedback.append("2. Extend your arms fully during the release.")
            feedback.append("3. Rotate your hips to add power to the release.")
            annotations.append(("Rectangle", (100, 150), (300, 350)))
            annotations.append(("Circle", (200, 300), 30))

        elif phase == 4:
            feedback.append(f"Post-Release (Turn {i + 1}):")
            feedback.append("1. Focus on the trajectory and landing.")
            feedback.append("2. Keep your body balanced after the release.")
            feedback.append("3. Prepare for a smooth follow-through.")
            annotations.append(("Text", (50, 50), f"Good Job Turn {i + 1}"))
            annotations.append(("Arrow", (150, 150), (200, 200)))

    return feedback, annotations

# Function to extract and preprocess frames from the video
def extract_and_preprocess_frames(video_path, frame_size=(224, 224), sequence_length=50):
    features = []
    cap = cv2.VideoCapture(video_path)

    # Use a pre-trained model for feature extraction (e.g., MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling='avg')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame and normalize
        frame = cv2.resize(frame, frame_size)
        frame = frame / 255.0

        # Extract features using the base model
        frame_features = base_model.predict(np.expand_dims(frame, axis=0))
        features.append(frame_features.flatten())

    cap.release()

    if len(features) < sequence_length:
        raise ValueError("Not enough frames in the video to match the sequence length.")

    features = features[:sequence_length]
    features = np.array(features)

    reshaped_features = features.reshape(1, sequence_length, -1)

    feature_dim = reshaped_features.shape[-1]
    if feature_dim > 99:
        reshaped_features = reshaped_features[:, :, :99]
    elif feature_dim < 99:
        padding = np.zeros((1, sequence_length, 99 - feature_dim))
        reshaped_features = np.concatenate((reshaped_features, padding), axis=-1)

    return reshaped_features

# Function to overlay annotations, feedback, and skeleton on video frames
def overlay_feedback_on_video(video_path, feedback, annotations, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    feedback_index = 0
    annotation_index = 0
    feedback_interval = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / len(feedback)))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            for idx, landmark in enumerate(result.pose_landmarks.landmark):
                if idx not in face_landmarks_indices:
                    mp_drawing.draw_landmarks(
                        frame, 
                        result.pose_landmarks, 
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )

        # Show feedback based on the frame count
        if feedback_index < len(feedback):
            cv2.putText(frame, feedback[feedback_index], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        if annotation_index < len(annotations):
            annotation = annotations[annotation_index]
            if annotation[0] == "Circle":
                cv2.circle(frame, annotation[1], annotation[2], (0, 255, 0), 3)
            elif annotation[0] == "Arrow":
                cv2.arrowedLine(frame, annotation[1], annotation[2], (0, 0, 255), 2, tipLength=0.3)
            elif annotation[0] == "Line":
                cv2.line(frame, annotation[1], annotation[2], (255, 0, 0), 2)
            elif annotation[0] == "Rectangle":
                cv2.rectangle(frame, annotation[1], annotation[2], (255, 255, 0), 3)
            elif annotation[0] == "Text":
                cv2.putText(frame, annotation[2], annotation[1], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_count += 1

        # Show feedback and annotations based on feedback interval
        if frame_count % feedback_interval == 0:
            feedback_index += 1
            annotation_index += 1

    cap.release()
    out.release()

# Function to find the next available filename with incremented numbers
def get_next_filename(folder, base_name, extension):
    i = 1
    while True:
        filename = f"{base_name}{i}.{extension}"
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            return file_path, filename
        i += 1

# Function to convert AVI to MP4
def convert_to_mp4(avi_path, mp4_path):
    command = f"ffmpeg -i {avi_path} -vcodec h264 -acodec aac {mp4_path}"
    subprocess.call(command, shell=True)

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        avi_output_path, avi_filename = get_next_filename(app.config['PROCESSED_FOLDER'], 'output', 'avi')
        mp4_output_path = avi_output_path.replace('.avi', '.mp4')

        # Extract and preprocess features from the video
        video_features = extract_and_preprocess_frames(file_path, frame_size=(224, 224), sequence_length=50)

        # Predict using the model
        predictions = model.predict(video_features)
        predicted_phases = np.argmax(predictions, axis=1)

        # Generate feedback and annotations
        feedback, annotations = generate_feedback(predicted_phases)

        # Overlay feedback, annotations, and skeleton on video
        overlay_feedback_on_video(file_path, feedback, annotations, avi_output_path)

        # Convert AVI to MP4
        convert_to_mp4(avi_output_path, mp4_output_path)

        # Remove the temporary AVI file
        os.remove(avi_output_path)

        return redirect(url_for('processed_video', filename=os.path.basename(mp4_output_path)))

    return render_template('upload.html')

@app.route('/processed/<filename>')
def processed_video(filename):
    return render_template('processed_video.html', video_url=url_for('static', filename=f'processed_videos/{filename}'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])
    app.run(debug=True)
