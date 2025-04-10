import cv2
import numpy as np
import time
import argparse
import os
import threading
import sys

class GazeGuard:
    def __init__(self, away_time_threshold=3, calibration_time=5, alert_sound=True):
        """
        Initialize the GazeGuard application
        
        Parameters:
        -----------
        away_time_threshold : int
            Maximum time in seconds allowed to look away before alert
        calibration_time : int
            Time in seconds for initial calibration
        alert_sound : bool
            Whether to use sound alerts
        """
        # Configuration
        self.away_time_threshold = away_time_threshold
        self.calibration_time = calibration_time
        self.alert_sound = alert_sound
        
        # Load OpenCV's pre-trained classifiers
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # State variables
        self.is_looking_away = False
        self.looking_away_start_time = 0
        self.alert_active = False
        self.calibration_active = False
        self.calibration_start_time = 0
        self.roi = None  # Region of interest
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Display settings
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Alert thread
        self.alert_thread = None
        self.running = True

    def play_alert(self):
        """Play alert sound and display visual alert"""
        try:
            # Try to use the playsound library if available
            import playsound
            alert_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alert.wav')
            
            # If alert sound file doesn't exist, use system beep
            if os.path.exists(alert_file):
                playsound.playsound(alert_file)
            else:
                print('\a')  # System beep
                
        except ImportError:
            # If playsound is not available, use system beep
            print('\a')
    
    def alert_loop(self):
        """Alert thread function - periodically triggers alerts while active"""
        while self.running:
            if self.alert_active:
                if self.alert_sound:
                    self.play_alert()
                time.sleep(2)  # Alert interval
            else:
                time.sleep(0.5)  # Check interval when not alerting
    
    def detect_eyes(self, frame, face=None):
        """
        Detect eyes in the frame or within a specific face area
        
        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame
        face : tuple, optional
            Face coordinates (x, y, w, h)
            
        Returns:
        --------
        list
            List of detected eyes as (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if face is not None:
            # Extract the face region
            x, y, w, h = face
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes within the face
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            
            # Adjust eye coordinates to be relative to the original frame
            return [(x + ex, y + ey, ew, eh) for (ex, ey, ew, eh) in eyes]
        else:
            # Detect in the entire frame
            return self.eye_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
    
    def detect_faces(self, frame):
        """
        Detect faces in the frame
        
        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame
            
        Returns:
        --------
        list
            List of detected faces as (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        return faces
    
    def is_looking_at_screen(self, frame):
        """
        Determine if the user is looking at the screen
        
        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame
            
        Returns:
        --------
        bool
            True if user appears to be looking at screen, False otherwise
        """
        faces = self.detect_faces(frame)
        
        if len(faces) == 0:
            return False  # No face detected
        
        # For now, we'll use the first (presumably largest/closest) face
        face = faces[0]
        
        # Detect eyes within the face
        eyes = self.detect_eyes(frame, face)
        
        # Simple heuristic: 
        # If we can detect at least two eyes, the person is likely looking at the screen
        return len(eyes) >= 2
    
    def run_calibration(self, frame):
        """
        Run the calibration process to set the ROI
        
        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame
        """
        elapsed = time.time() - self.calibration_start_time
        
        if elapsed < self.calibration_time:
            # Draw calibration instructions
            cv2.putText(frame, f"Calibrating... Look at the screen ({int(self.calibration_time - elapsed)}s)", 
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Detect face during calibration
            faces = self.detect_faces(frame)
            if len(faces) > 0:
                # Draw rectangle around the detected face
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:
            # Calibration finished
            self.calibration_active = False
            faces = self.detect_faces(frame)
            if len(faces) > 0:
                self.roi = faces[0]  # Save the face region as ROI
                cv2.putText(frame, "Calibration complete!", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Calibration failed! No face detected.", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Restart calibration
                self.calibration_start_time = time.time()
    
    def start_calibration(self):
        """Start the calibration process"""
        self.calibration_active = True
        self.calibration_start_time = time.time()
    
    def draw_status(self, frame):
        """
        Draw status information on the frame
        
        Parameters:
        -----------
        frame : numpy.ndarray
            The video frame to draw on
        """
        # Draw border based on current state
        border_color = (0, 255, 0)  # Green for normal state
        border_thickness = 2
        
        if self.alert_active:
            # Red pulsing border when alert is active
            intensity = int(128 + 127 * np.sin(time.time() * 5))
            border_color = (0, 0, intensity + 128)  # Pulsing red
            border_thickness = 5
        elif self.is_looking_away:
            # Yellow border when looking away but not alerting yet
            border_color = (0, 255, 255)
            border_thickness = 3
            
        # Draw border
        cv2.rectangle(frame, (0, 0), (self.frame_width, self.frame_height), 
                     border_color, border_thickness)
        
        # Draw status text
        if self.alert_active:
            cv2.putText(frame, "LOOK AT THE SCREEN!", 
                       (int(self.frame_width/2) - 150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif self.is_looking_away:
            remaining = max(0, self.away_time_threshold - (time.time() - self.looking_away_start_time))
            cv2.putText(frame, f"Looking away: {remaining:.1f}s remaining", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Focused", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw controls info
        cv2.putText(frame, "Press 'c' to calibrate, '+'/'-' to adjust threshold, 'q' to quit", 
                   (20, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Away threshold: {self.away_time_threshold}s", 
                   (20, self.frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
    def run(self):
        """Main application loop"""
        # Start alert thread
        self.alert_thread = threading.Thread(target=self.alert_loop)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        
        # Initial calibration
        self.start_calibration()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                    
                # Flip frame horizontally for a more natural mirror view
                frame = cv2.flip(frame, 1)
                
                if self.calibration_active:
                    self.run_calibration(frame)
                else:
                    # Check if looking at screen
                    looking_at_screen = self.is_looking_at_screen(frame)
                    
                    # Update state
                    if looking_at_screen:
                        if self.is_looking_away:
                            self.is_looking_away = False
                            self.alert_active = False
                    else:
                        if not self.is_looking_away:
                            self.is_looking_away = True
                            self.looking_away_start_time = time.time()
                        elif not self.alert_active and time.time() - self.looking_away_start_time > self.away_time_threshold:
                            self.alert_active = True
                    
                    # Draw faces and eyes for visual feedback
                    faces = self.detect_faces(frame)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        eyes = self.detect_eyes(frame, (x, y, w, h))
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Draw status information
                self.draw_status(frame)
                
                # Display the frame
                cv2.imshow('Gaze Guard', frame)
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.start_calibration()
                elif key == ord('+') or key == ord('='):
                    self.away_time_threshold += 1
                elif key == ord('-') or key == ord('_'):
                    self.away_time_threshold = max(1, self.away_time_threshold - 1)
                
        finally:
            # Clean up
            self.running = False
            if self.alert_thread and self.alert_thread.is_alive():
                self.alert_thread.join(timeout=1)
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    """Parse arguments and start the application"""
    parser = argparse.ArgumentParser(description='Gaze Guard - Monitor your focus')
    parser.add_argument('--threshold', type=int, default=3,
                        help='Maximum seconds allowed to look away (default: 3)')
    parser.add_argument('--calibration', type=int, default=5,
                        help='Calibration time in seconds (default: 5)')
    parser.add_argument('--mute', action='store_true',
                        help='Mute audio alerts')
    
    args = parser.parse_args()
    
    print("=== Gaze Guard ===")
    print(f"Look-away threshold: {args.threshold} seconds")
    print(f"Initial calibration: {args.calibration} seconds")
    print(f"Audio alerts: {'Disabled' if args.mute else 'Enabled'}")
    print("\nControls:")
    print("  c - Recalibrate")
    print("  + - Increase look-away threshold")
    print("  - - Decrease look-away threshold")
    print("  q - Quit")
    print("\nStarting application...")
    
    app = GazeGuard(
        away_time_threshold=args.threshold,
        calibration_time=args.calibration,
        alert_sound=not args.mute
    )
    app.run()


if __name__ == "__main__":
    main()
