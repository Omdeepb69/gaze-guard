import cv2
import numpy as np
import os
import time

class GazeDetector:
    """
    Handles face and eye detection using Haar Cascades.
    Estimates gaze direction based on eye detection.
    """
    LOOKING_FORWARD = "LOOKING_FORWARD"
    LOOKING_AWAY = "LOOKING_AWAY"
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    FACE_DETECTION_ERROR = "FACE_DETECTION_ERROR"
    EYE_DETECTION_ERROR = "EYE_DETECTION_ERROR"

    def __init__(self, face_cascade_path=None, eye_cascade_path=None):
        """
        Initializes the face and eye detectors.

        Args:
            face_cascade_path (str, optional): Path to the face Haar cascade XML file.
                                               Defaults to OpenCV's default path.
            eye_cascade_path (str, optional): Path to the eye Haar cascade XML file.
                                              Defaults to OpenCV's default path.
        """
        if face_cascade_path is None:
            face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if eye_cascade_path is None:
            eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')

        if not os.path.exists(face_cascade_path):
            raise FileNotFoundError(f"Face cascade file not found at {face_cascade_path}")
        if not os.path.exists(eye_cascade_path):
            raise FileNotFoundError(f"Eye cascade file not found at {eye_cascade_path}")

        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

        if self.face_cascade.empty():
            raise IOError(f"Failed to load face cascade from {face_cascade_path}")
        if self.eye_cascade.empty():
            raise IOError(f"Failed to load eye cascade from {eye_cascade_path}")

        self.last_gaze_state = self.NO_FACE_DETECTED
        self.away_start_time = None
        self.alert_triggered = False

    def detect_features(self, frame: np.ndarray):
        """
        Detects faces and eyes in a given frame.

        Args:
            frame (np.ndarray): The input image frame (BGR).

        Returns:
            tuple: (list of face bounding boxes, list of lists of eye bounding boxes per face)
                   Returns (None, None) if face detection fails.
                   Eye bounding boxes are relative to the full frame.
        """
        if frame is None or frame.size == 0:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray) # Improve contrast

        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        except cv2.error as e:
            print(f"Error during face detection: {e}")
            return self.FACE_DETECTION_ERROR, None, None

        if len(faces) == 0:
            return [], [] # No faces detected

        all_eyes = []
        valid_faces = []

        for (x, y, w, h) in faces:
            face_roi_gray = gray[y:y+h, x:x+w]
            if face_roi_gray.size == 0:
                continue # Skip if ROI is invalid

            try:
                eyes = self.eye_cascade.detectMultiScale(
                    face_roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(20, 20) # Adjust min size relative to face size if needed
                )
            except cv2.error as e:
                print(f"Error during eye detection: {e}")
                # Treat as if no eyes were found for this face
                eyes = []


            # Convert eye coordinates from face ROI to full frame coordinates
            eyes_global_coords = [(ex + x, ey + y, ew, eh) for (ex, ey, ew, eh) in eyes]
            all_eyes.append(eyes_global_coords)
            valid_faces.append((x, y, w, h))

        return valid_faces, all_eyes

    def estimate_gaze_direction(self, faces: list, eyes_per_face: list):
        """
        Estimates gaze direction based on detected features.
        Focuses on the largest detected face.

        Args:
            faces (list): List of face bounding boxes [(x, y, w, h), ...].
            eyes_per_face (list): List of lists of eye bounding boxes [[(ex, ey, ew, eh), ...], ...].

        Returns:
            str: Gaze state (LOOKING_FORWARD, LOOKING_AWAY, NO_FACE_DETECTED).
        """
        if faces is None or not faces:
            return self.NO_FACE_DETECTED

        # Find the largest face (assuming it's the user)
        largest_face_idx = -1
        max_area = 0
        for i, (x, y, w, h) in enumerate(faces):
            area = w * h
            if area > max_area:
                max_area = area
                largest_face_idx = i

        if largest_face_idx == -1:
             return self.NO_FACE_DETECTED # Should not happen if faces is not empty, but safety check

        eyes_in_largest_face = eyes_per_face[largest_face_idx]

        # Simplistic gaze estimation: Check if at least two eyes are detected
        # More sophisticated methods could analyze eye position, pupil, etc.
        if len(eyes_in_largest_face) >= 2:
            # Basic check: ensure eyes are roughly horizontally aligned and within upper face half
            (fx, fy, fw, fh) = faces[largest_face_idx]
            eye_centers_y = [ey + eh / 2 for (ex, ey, ew, eh) in eyes_in_largest_face]
            
            # Check if eyes are in the upper ~60% of the face height
            valid_eyes = [e for e_idx, (ex, ey, ew, eh) in enumerate(eyes_in_largest_face) if (ey + eh/2) < (fy + fh * 0.6)]

            if len(valid_eyes) >= 2:
                 # Could add more checks here (e.g., distance between eyes)
                 return self.LOOKING_FORWARD
            else:
                 return self.LOOKING_AWAY # Eyes detected but not in expected position/count
        elif len(eyes_in_largest_face) == 1:
             # Only one eye detected, likely looking away or partial occlusion
             return self.LOOKING_AWAY
        else:
            # No eyes detected in the largest face ROI
            return self.LOOKING_AWAY


    def update_gaze_state(self, frame: np.ndarray, away_threshold_seconds: float):
        """
        Processes a frame, updates the gaze state, and checks the away timer.

        Args:
            frame (np.ndarray): The input video frame.
            away_threshold_seconds (float): Maximum allowed duration (in seconds)
                                            for looking away before triggering an alert.

        Returns:
            tuple: (current_gaze_state: str, alert_active: bool, faces: list, eyes_per_face: list)
                   alert_active is True if the away timer has expired.
        """
        faces, eyes_per_face = self.detect_features(frame)

        if faces == self.FACE_DETECTION_ERROR or faces == self.EYE_DETECTION_ERROR:
             # Handle cascade errors - perhaps maintain previous state or specific error state
             current_gaze_state = self.last_gaze_state # Maintain last known state on error
             alert_active = self.alert_triggered
             return current_gaze_state, alert_active, [], []


        current_gaze_state = self.estimate_gaze_direction(faces, eyes_per_face)

        alert_active = False
        now = time.time()

        if current_gaze_state == self.LOOKING_FORWARD:
            self.away_start_time = None # Reset timer when looking forward
            self.alert_triggered = False
        else: # LOOKING_AWAY or NO_FACE_DETECTED
            if self.away_start_time is None:
                # Start timer only if the previous state wasn't already away/no_face
                if self.last_gaze_state == self.LOOKING_FORWARD:
                    self.away_start_time = now
            elif not self.alert_triggered:
                # Timer is running, check if threshold exceeded
                elapsed_away_time = now - self.away_start_time
                if elapsed_away_time > away_threshold_seconds:
                    alert_active = True
                    self.alert_triggered = True # Keep alert active until looking forward again

        # If alert was already triggered, keep it active until looking forward
        if self.alert_triggered and current_gaze_state != self.LOOKING_FORWARD:
            alert_active = True

        self.last_gaze_state = current_gaze_state
        return current_gaze_state, alert_active, faces, eyes_per_face

# Example Usage (can be run standalone for basic testing)
if __name__ == '__main__':
    print("Initializing Gaze Detector...")
    try:
        detector = GazeDetector()
        print("Gaze Detector Initialized.")
    except (FileNotFoundError, IOError) as e:
        print(f"Error initializing detector: {e}")
        print("Please ensure Haar Cascade XML files are accessible.")
        print("Default location is within the OpenCV installation data folder.")
        exit()

    print("Attempting to open webcam...")
    cap = cv2.VideoCapture(0) # Use 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Webcam opened successfully. Starting detection loop (press 'q' to quit)...")

    away_duration_threshold = 5.0 # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)

        # Process frame
        gaze_state, alert, detected_faces, detected_eyes_per_face = detector.update_gaze_state(
            frame, away_duration_threshold
        )

        # --- Visualization ---
        # Draw rectangles around detected faces and eyes
        if detected_faces:
             # Highlight the largest face (used for gaze estimation)
            largest_face_idx = -1
            max_area = 0
            for i, (x, y, w, h) in enumerate(detected_faces):
                area = w * h
                if area > max_area:
                    max_area = area
                    largest_face_idx = i

            for i, (x, y, w, h) in enumerate(detected_faces):
                color = (0, 255, 0) if i == largest_face_idx else (255, 0, 0) # Green for largest, Blue otherwise
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                if i < len(detected_eyes_per_face):
                    eyes_in_this_face = detected_eyes_per_face[i]
                    for (ex, ey, ew, eh) in eyes_in_this_face:
                         cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 1) # Red for eyes

        # Display gaze state and alert status
        status_text = f"Gaze: {gaze_state}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if alert:
            alert_text = "ALERT: Look at the screen!"
            text_size, _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = frame.shape[0] - 30
            cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            # Optional: Add sound alert here using playsound or another library
            # from playsound import playsound
            # try:
            #     playsound('alert.wav', block=False) # Requires 'alert.wav' file
            # except Exception as e:
            #     print(f"Could not play sound: {e}")


        # Show the frame
        cv2.imshow('Gaze Guard - Model Test', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released.")