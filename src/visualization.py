import cv2
import numpy as np
import time
import argparse
import sys
import os

# Constants
DEFAULT_AWAY_TIME_LIMIT = 5.0  # seconds
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FACE_CASCADE = 'haarcascade_frontalface_default.xml'
DEFAULT_EYE_CASCADE = 'haarcascade_eye.xml'

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)

GAZE_STATUS_FORWARD = "LOOKING FORWARD"
GAZE_STATUS_AWAY = "LOOKING AWAY"
GAZE_STATUS_NO_FACE = "NO FACE DETECTED"
GAZE_STATUS_NO_EYES = "EYES NOT CLEAR"

ALERT_MESSAGE = "ALERT: Look Forward!"

# --- ROI Selection Handling ---
roi_selecting = False
roi_start_point = (-1, -1)
roi_end_point = (-1, -1)
selected_roi = None

def select_roi_callback(event, x, y, flags, param):
    global roi_selecting, roi_start_point, roi_end_point, selected_roi
    frame_copy = param['frame'].copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_selecting = True
        roi_start_point = (x, y)
        roi_end_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if roi_selecting:
            roi_end_point = (x, y)
            cv2.rectangle(frame_copy, roi_start_point, roi_end_point, COLOR_YELLOW, 2)
            cv2.imshow("Select ROI", frame_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_selecting = False
        roi_end_point = (x, y)
        # Ensure x1 < x2 and y1 < y2
        x1, y1 = roi_start_point
        x2, y2 = roi_end_point
        start_x, end_x = min(x1, x2), max(x1, x2)
        start_y, end_y = min(y1, y2), max(y1, y2)
        # Ensure ROI has non-zero width and height
        if end_x > start_x and end_y > start_y:
            selected_roi = (start_x, start_y, end_x - start_x, end_y - start_y)
            print(f"ROI selected: {selected_roi}")
        else:
            print("Invalid ROI selection (zero width or height).")
            selected_roi = None # Reset if invalid
        # Keep the window open briefly to show final ROI or close immediately
        # cv2.waitKey(500) # Optional delay
        cv2.destroyWindow("Select ROI")

def get_roi_from_user(capture):
    global selected_roi, roi_start_point, roi_end_point
    selected_roi = None
    roi_start_point = (-1, -1)
    roi_end_point = (-1, -1)

    print("\n--- ROI Selection ---")
    print("Draw a rectangle on the window to define the Region of Interest for eye detection.")
    print("Press any key after drawing to confirm.")

    window_name = "Select ROI"
    cv2.namedWindow(window_name)

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to capture frame for ROI selection.")
            cv2.destroyWindow(window_name)
            return None

        frame_display = frame.copy()
        if roi_start_point != (-1, -1) and roi_end_point != (-1, -1) and not roi_selecting:
             if selected_roi: # Draw confirmed ROI
                 x, y, w, h = selected_roi
                 cv2.rectangle(frame_display, (x, y), (x + w, y + h), COLOR_GREEN, 2)
                 cv2.putText(frame_display, "ROI Selected. Press any key.", (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)

        cv2.imshow(window_name, frame_display)
        cv2.setMouseCallback(window_name, select_roi_callback, {'frame': frame})

        key = cv2.waitKey(1) & 0xFF
        if selected_roi is not None and key != 255: # Key pressed after selection
             break
        if key == ord('q'): # Allow quitting ROI selection
            print("ROI selection cancelled.")
            selected_roi = None
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: # Window closed
            print("ROI selection window closed.")
            if not selected_roi: # If closed without confirming a valid ROI
                selected_roi = None
            break


    cv2.destroyWindow(window_name)
    # Re-initialize mouse callback state for safety if needed elsewhere
    # reset_roi_selection_state()
    return selected_roi

# --- Gaze Guard Core Logic ---
class GazeGuardVisualizer:
    def __init__(self, camera_index, away_time_limit, face_cascade_path, eye_cascade_path):
        self.camera_index = camera_index
        self.away_time_limit = away_time_limit
        self.face_cascade_path = face_cascade_path
        self.eye_cascade_path = eye_cascade_path

        if not os.path.exists(self.face_cascade_path):
            print(f"Error: Face cascade file not found at {self.face_cascade_path}")
            sys.exit(1)
        if not os.path.exists(self.eye_cascade_path):
            print(f"Error: Eye cascade file not found at {self.eye_cascade_path}")
            sys.exit(1)

        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade_path)

        if self.face_cascade.empty():
            print(f"Error: Failed to load face cascade from {self.face_cascade_path}")
            sys.exit(1)
        if self.eye_cascade.empty():
            print(f"Error: Failed to load eye cascade from {self.eye_cascade_path}")
            sys.exit(1)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            sys.exit(1)

        self.roi = None
        self.gaze_status = GAZE_STATUS_NO_FACE
        self.away_timer_start = None
        self.alert_active = False

    def detect_eyes(self, gray_frame, frame_to_draw_on):
        scale_factor = 1.1
        min_neighbors_face = 5
        min_neighbors_eyes = 5 # Adjusted for potentially better filtering
        min_size_face = (50, 50)
        min_size_eyes = (20, 20) # Adjusted min eye size

        target_frame = gray_frame
        offset = (0, 0)

        if self.roi:
            x, y, w, h = self.roi
            target_frame = gray_frame[y:y+h, x:x+w]
            offset = (x, y)
            # Draw ROI boundary on the display frame
            cv2.rectangle(frame_to_draw_on, (x, y), (x+w, y+h), COLOR_BLUE, 1)
            cv2.putText(frame_to_draw_on, "ROI Active", (x, y - 10 if y > 10 else y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 1)


        faces = self.face_cascade.detectMultiScale(
            target_frame,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors_face,
            minSize=min_size_face
        )

        detected_eyes_coords = []
        detected_face_coords = None

        if len(faces) == 0:
            self.gaze_status = GAZE_STATUS_NO_FACE
            return detected_eyes_coords, detected_face_coords

        # Assume the largest detected face is the user
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        face_roi_gray = target_frame[fy:fy+fh, fx:fx+fw]

        # Adjust face coordinates back to original frame space if ROI is used
        orig_fx, orig_fy = fx + offset[0], fy + offset[1]
        detected_face_coords = (orig_fx, orig_fy, fw, fh)

        # Draw face rectangle
        cv2.rectangle(frame_to_draw_on, (orig_fx, orig_fy), (orig_fx+fw, orig_fy+fh), COLOR_GREEN, 2)

        eyes = self.eye_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors_eyes,
            minSize=min_size_eyes
        )

        # Filter eyes based on position within the face ROI (e.g., upper half)
        filtered_eyes = []
        for (ex, ey, ew, eh) in eyes:
             if ey < fh / 2: # Simple filter: eyes should be in the upper half of the face box
                 filtered_eyes.append((ex, ey, ew, eh))


        if len(filtered_eyes) >= 2:
            # Simplistic: If 2+ eyes detected (in upper face), assume looking forward
            self.gaze_status = GAZE_STATUS_FORWARD
            # Sort eyes by x-coordinate to potentially distinguish left/right
            filtered_eyes.sort(key=lambda e: e[0])
            for (ex, ey, ew, eh) in filtered_eyes[:2]: # Draw only the first two detected
                eye_center_x = orig_fx + ex + ew // 2
                eye_center_y = orig_fy + ey + eh // 2
                radius = (ew + eh) // 4
                cv2.circle(frame_to_draw_on, (eye_center_x, eye_center_y), radius, COLOR_GREEN, 2)
                detected_eyes_coords.append((orig_fx + ex, orig_fy + ey, ew, eh))
        elif len(filtered_eyes) == 1:
             # If only one eye detected, could be looking sideways or partial occlusion
             self.gaze_status = GAZE_STATUS_AWAY # Treat as looking away
             (ex, ey, ew, eh) = filtered_eyes[0]
             eye_center_x = orig_fx + ex + ew // 2
             eye_center_y = orig_fy + ey + eh // 2
             radius = (ew + eh) // 4
             cv2.circle(frame_to_draw_on, (eye_center_x, eye_center_y), radius, COLOR_RED, 2)
             detected_eyes_coords.append((orig_fx + ex, orig_fy + ey, ew, eh))
        else:
            # No eyes detected within the face
            self.gaze_status = GAZE_STATUS_NO_EYES # Or GAZE_STATUS_AWAY

        return detected_eyes_coords, detected_face_coords


    def update_timer_and_alert(self):
        now = time.time()
        is_looking_away = self.gaze_status in [GAZE_STATUS_AWAY, GAZE_STATUS_NO_EYES, GAZE_STATUS_NO_FACE]

        if is_looking_away:
            if self.away_timer_start is None:
                self.away_timer_start = now
            elapsed_time = now - self.away_timer_start
            if elapsed_time >= self.away_time_limit:
                self.alert_active = True
            else:
                self.alert_active = False # Reset alert if time limit not yet reached
        else: # Looking forward
            self.away_timer_start = None
            self.alert_active = False

    def draw_visuals(self, frame):
        # Display Gaze Status
        status_color = COLOR_GREEN if self.gaze_status == GAZE_STATUS_FORWARD else COLOR_RED
        cv2.putText(frame, f"Status: {self.gaze_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Display Timer
        timer_text = "Timer: OK"
        timer_color = COLOR_GREEN
        if self.away_timer_start is not None:
            elapsed = time.time() - self.away_timer_start
            remaining = max(0, self.away_time_limit - elapsed)
            timer_text = f"Away Timer: {remaining:.1f}s / {self.away_time_limit:.1f}s"
            timer_color = COLOR_YELLOW if not self.alert_active else COLOR_RED
        cv2.putText(frame, timer_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, timer_color, 2)

        # Display Alert
        if self.alert_active:
            # Red border
            h, w = frame.shape[:2]
            border_size = 10
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_RED, border_size * 2) # Thicker border

            # Alert text
            text_size, _ = cv2.getTextSize(ALERT_MESSAGE, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
            text_x = (w - text_size[0]) // 2
            text_y = h - 40 # Position near bottom
            cv2.putText(frame, ALERT_MESSAGE, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_RED, 3, cv2.LINE_AA)

        # Display Instructions
        cv2.putText(frame, "Press 'q' to quit, 'c' to calibrate ROI", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)


    def run(self):
        print("Starting Gaze Guard...")
        print(f"Away time limit: {self.away_time_limit} seconds")
        print("Press 'c' to select Region of Interest (ROI).")
        print("Press 'q' to quit.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame_display = frame.copy() # Work on a copy for drawing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # --- Processing Steps ---
            eyes, face = self.detect_eyes(gray, frame_display)
            self.update_timer_and_alert()
            self.draw_visuals(frame_display)

            # --- Display ---
            cv2.imshow('Gaze Guard', frame_display)

            # --- User Input ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('c'):
                print("Entering ROI selection mode...")
                # Pause processing and select ROI
                cv2.destroyWindow('Gaze Guard') # Close main window temporarily
                self.roi = get_roi_from_user(self.cap)
                if self.roi:
                    print(f"ROI set to: {self.roi}")
                else:
                    print("ROI selection cancelled or failed. Using full frame.")
                # Re-create main window implicitly in the next loop iteration
                # or explicitly: cv2.namedWindow('Gaze Guard')

        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("Resources released.")

def main():
    parser = argparse.ArgumentParser(description="Gaze Guard - Eye Tracking and Alert System")
    parser.add_argument("-c", "--camera", type=int, default=DEFAULT_CAMERA_INDEX,
                        help="Camera index")
    parser.add_argument("-t", "--time_limit", type=float, default=DEFAULT_AWAY_TIME_LIMIT,
                        help="Maximum allowed time (seconds) to look away")
    parser.add_argument("--face_cascade", type=str, default=DEFAULT_FACE_CASCADE,
                        help="Path to face detection Haar cascade XML file")
    parser.add_argument("--eye_cascade", type=str, default=DEFAULT_EYE_CASCADE,
                        help="Path to eye detection Haar cascade XML file")
    args = parser.parse_args()

    # Check if cascade files exist (basic check)
    if not os.path.isfile(args.face_cascade):
        print(f"Error: Face cascade file not found: {args.face_cascade}")
        print("Please download it (e.g., from OpenCV's GitHub repository) or provide the correct path.")
        sys.exit(1)
    if not os.path.isfile(args.eye_cascade):
        print(f"Error: Eye cascade file not found: {args.eye_cascade}")
        print("Please download it (e.g., from OpenCV's GitHub repository) or provide the correct path.")
        sys.exit(1)


    visualizer = GazeGuardVisualizer(
        camera_index=args.camera,
        away_time_limit=args.time_limit,
        face_cascade_path=args.face_cascade,
        eye_cascade_path=args.eye_cascade
    )
    visualizer.run()

if __name__ == "__main__":
    main()