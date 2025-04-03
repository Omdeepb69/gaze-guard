```python
import cv2
import json
import os
import time
import numpy as np
try:
    import simpleaudio as sa
except ImportError:
    print("Warning: simpleaudio library not found. Audible alerts will be disabled.")
    print("Install it using: pip install simpleaudio")
    sa = None
import wave
import struct


DEFAULT_CONFIG_PATH = 'config.json'
DEFAULT_SOUND_PATH = 'alert.wav'


def load_config(config_path=DEFAULT_CONFIG_PATH):
    default_config = {
        "max_away_time_seconds": 5,
        "eye_cascade_path": cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml', # Better for glasses
        "face_cascade_path": cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        "alert_sound_enabled": True,
        "alert_sound_path": DEFAULT_SOUND_PATH,
        "use_roi": False,
        "roi": None,
        "camera_index": 0,
        "min_eye_size": (25, 25),
        "scale_factor_eye": 1.1,
        "min_neighbors_eye": 7, # Increased for potentially more false positives with glasses cascade
        "scale_factor_face": 1.1,
        "min_neighbors_face": 5,
        "flip_frame_horizontal": False,
        "gaze_detection_threshold": 0.6 # Ratio of detected eyes needed to be considered 'looking'
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge user config with defaults, ensuring all keys exist
                config = default_config.copy()
                config.update(user_config)
                # Ensure cascade paths are resolved correctly
                config['eye_cascade_path'] = get_cascade_path(config['eye_cascade_path'])
                config['face_cascade_path'] = get_cascade_path(config['face_cascade_path'])
                return config
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading config file {config_path}: {e}. Using default configuration.")
            # Resolve default paths as well
            default_config['eye_cascade_path'] = get_cascade_path(default_config['eye_cascade_path'])
            default_config['face_cascade_path'] = get_cascade_path(default_config['face_cascade_path'])
            return default_config
    else:
        print(f"Config file {config_path} not found. Using default configuration.")
        # Resolve default paths
        default_config['eye_cascade_path'] = get_cascade_path(default_config['eye_cascade_path'])
        default_config['face_cascade_path'] = get_cascade_path(default_config['face_cascade_path'])
        # Optionally save the default config here if desired
        # save_config(default_config, config_path)
        return default_config

def save_config(config_data, config_path=DEFAULT_CONFIG_PATH):
    try:
        # Avoid saving resolved absolute paths if they were originally relative/default
        config_to_save = config_data.copy()
        if config_to_save['eye_cascade_path'] == cv2.data.haarcascades + os.path.basename(config_to_save['eye_cascade_path']):
             config_to_save['eye_cascade_path'] = os.path.basename(config_to_save['eye_cascade_path'])
        if config_to_save['face_cascade_path'] == cv2.data.haarcascades + os.path.basename(config_to_save['face_cascade_path']):
             config_to_save['face_cascade_path'] = os.path.basename(config_to_save['face_cascade_path'])

        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)
        print(f"Configuration saved to {config_path}")
    except IOError as e:
        print(f"Error saving config file {config_path}: {e}")


def check_file_exists(file_path):
    return file_path is not None and os.path.exists(file_path)

def get_cascade_path(cascade_filename_or_path):
    if cascade_filename_or_path is None:
        return None

    # 1. Check if it's an absolute path
    if os.path.isabs(cascade_filename_or_path) and os.path.exists(cascade_filename_or_path):
        return cascade_filename_or_path

    # 2. Check if it exists relative to the current working directory
    if os.path.exists(cascade_filename_or_path):
        return os.path.abspath(cascade_filename_or_path)

    # 3. Check in standard OpenCV cascade location
    try:
        cv_cascade_dir = cv2.data.haarcascades
        if cv_cascade_dir: # Check if the path is valid
             cv_cascade_path = os.path.join(cv_cascade_dir, os.path.basename(cascade_filename_or_path))
             if os.path.exists(cv_cascade_path):
                 return cv_cascade_path
    except AttributeError:
         print("Warning: cv2.data seems unavailable. Cannot find default cascades automatically.")


    # 4. If not found anywhere, return None or raise an error
    print(f"Error: Cascade file '{cascade_filename_or_path}' not found.")
    return None


def draw_roi(frame, roi, color=(255, 0, 0), thickness=2):
    if roi:
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(frame, 'ROI', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

def draw_detections(frame, detections, color=(0, 255, 0), thickness=2):
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

def draw_status(frame, status_text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, status_text, position, font, scale, color, thickness, cv2.LINE_AA)

def draw_timer(frame, current_time, max_time, is_away, position=(10, 60), font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, color_ok=(0, 255, 255), color_away=(0, 165, 255), thickness=2):
    timer_text = f"Away: {current_time:.1f}s / {max_time:.1f}s"
    color = color_away if is_away else color_ok
    cv2.putText(frame, timer_text, position, font, scale, color, thickness, cv2.LINE_AA)

def display_alert(frame, message="LOOK AT SCREEN!", color=(0, 0, 255), thickness=3):
    h, w = frame.shape[:2]
    font_scale = min(w / 400, 1.5) # Scale font based on frame width
    font = cv2.FONT_HERSHEY_TRIPLEX
    (text_width, text_height), baseline = cv2.getTextSize(message, font, font_scale, thickness)
    text_x = max(0, (w - text_width) // 2)
    text_y = max(text_height, (h + text_height) // 2)

    bg_x1 = max(0, text_x - 10)
    bg_y1 = max(0, text_y - text_height - 10 - baseline)
    bg_x2 = min(w, text_x + text_width + 10)
    bg_y2 = min(h, text_y + baseline + 10)

    sub_frame = frame[bg_y1:bg_y2, bg_x1:bg_x2]
    black_rect = np.zeros(sub_frame.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_frame, 0.5, black_rect, 0.5, 1.0) # Semi-transparent background
    frame[bg_y1:bg_y2, bg_x1:bg_x2] = res

    cv2.putText(frame, message, (text_x, text_y - baseline // 2), font, font_scale, color, thickness, cv2.LINE_AA)


_alert_sound_obj = None
_alert_wave_obj = None

def _load_alert_sound(sound_file_path):
    global _alert_wave_obj
    if sa is None or not check_file_exists(sound_file_path):
        _alert_wave_obj = None
        return False
    try:
        _alert_wave_obj = sa.WaveObject.from_wave_file(sound_file_path)
        return True
    except Exception as e:
        print(f"Error loading sound file {sound_file_path}: {e}")
        _alert_wave_obj = None
        return False

def play_alert_sound(sound_file_path):
    global _alert_sound_obj, _alert_wave_obj
    if sa is None:
        # print("Audible alert skipped: simpleaudio not available.")
        return

    if _alert_wave_obj is None:
        if not _load_alert_sound(sound_file_path):
            print(f"Warning: Alert sound file not found or invalid: {sound_file_path}")
            return

    try:
        if _alert_sound_obj and _alert_sound_obj.is_playing():
            return # Sound is already playing

        if _alert_wave_obj:
            _alert_sound_obj = _alert_wave_obj.play()
    except Exception as e:
        print(f"Error playing sound {sound_file_path}: {e}")
        _alert_sound_obj = None
        _alert_wave