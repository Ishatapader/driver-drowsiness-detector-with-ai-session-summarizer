import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
import csv
import os
import json
from datetime import datetime
from collections import deque

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False


class DrowsinessDetector:
    def __init__(self, config_path="config.json"):

        #  LOAD CONFIGURATION
        try:
            with open(config_path, 'r') as file:
                config = json.load(file)
        except FileNotFoundError:
            print(f"[ERROR] Configuration file '{config_path}' missing!")
            exit(1)

        # Dynamically pulling values from config.json
        self.EAR_THRESH = config['thresholds']['ear_drowsiness_limit']
        self.EAR_CONSEC_FRAMES = config['thresholds']['ear_consec_frames']

        self.MAR_THRESH = config['thresholds']['mar_yawn_limit']
        self.MAR_CONSEC_FRAMES = config['thresholds']['mar_consec_frames']

        self.PITCH_DELTA_THRESH = config['thresholds']['pitch_nod_limit']
        self.PITCH_CONSEC_FRAMES = config['thresholds']['pitch_consec_frames']

        self.CALIBRATION_FRAMES = config['system']['calibration_frames']
        self.CSV_FILE = config['system']['csv_log_file']

        # Audio config 
        audio_cfg = config.get('audio', {})
        self.AUDIO_SAMPLE_RATE = audio_cfg.get('sample_rate', 44100)
        self.AUDIO_FREQ_HZ = audio_cfg.get('freq_hz', 1000)
        self.AUDIO_DURATION_SEC = audio_cfg.get('duration_sec', 0.3)
        self.AUDIO_REPEAT_INTERVAL_SEC = audio_cfg.get('repeat_interval_sec', 1.5)

        # State Tracking Variables
        self.calibration_pitches = []
        self.pitch_baseline = None
        self.ear_counter = 0
        self.mar_counter = 0
        self.pitch_counter = 0

        # Episode tracking 
        self.active_episode = None       # currently active threat type, or None
        self.episode_start_time = None   # when current episode began
        self.last_beep_time = 0.0        # for repeat-while-active beeping

        # Audio & Threading Variables
        self.is_muted = False
        self.alarm_lock = threading.Lock()
        self.audio_queue = queue.Queue()
        self.audio_ok = SOUNDDEVICE_AVAILABLE   
        self.audio_status_msg = ""

        self.tone_array = self._generate_tone()
        self.stream = None
        if SOUNDDEVICE_AVAILABLE:
            self.stream = self._open_stream()
        else:
            self.audio_ok = False
            self.audio_status_msg = "sounddevice not installed"
            print("[WARNING] sounddevice not available — audio alerts disabled. "
                  "Install with: pip install sounddevice")

        # Background audio worker (writes to the persistent stream, never blocks main loop)
        threading.Thread(target=self._audio_worker, daemon=True).start()
        # Watchdog: periodically verifies the output stream is still alive
        threading.Thread(target=self._audio_watchdog, daemon=True).start()

        self.PITCH_SMOOTH_N = 5
        self.pitch_buffer = deque(maxlen=self.PITCH_SMOOTH_N)

        # Initialize the CSV File
        self._init_csv()

        #  LANDMARK INDICES & 3D MODEL

        self.LEFT_EYE  = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH     = [78, 81, 13, 311, 308, 402, 14, 178]
        self.POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

        self.MODEL_POINTS = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0,-150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype=np.float32)

    # CSV LOGGING

    def _init_csv(self):
        if not os.path.exists(self.CSV_FILE):
            with open(self.CSV_FILE, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Event", "Warning_Type", "EAR", "MAR", "Pitch_Delta", "Duration_Sec"])

    def log_telemetry(self, event, warning_type, current_ear, current_mar, current_pitch_delta, duration=None):
        """event: 'START' or 'END'. One row per episode boundary, not per frame."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp, event, warning_type,
                round(current_ear, 3), round(current_mar, 3), round(current_pitch_delta, 3),
                round(duration, 2) if duration is not None else ""
            ])
        if event == "START":
            print(f"[LOG] {warning_type} STARTED at {timestamp}")
        else:
            print(f"[LOG] {warning_type} ENDED at {timestamp} (duration {duration:.1f}s)")

    # AUDIO ENGINE

    def _generate_tone(self):
        """Pre-generate the alert waveform once. Triggering an alert is then
        just a buffer write into an already-open stream — no synthesis,
        no subprocess, no file I/O on the hot path."""
        t = np.linspace(0, self.AUDIO_DURATION_SEC,
                         int(self.AUDIO_SAMPLE_RATE * self.AUDIO_DURATION_SEC), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * self.AUDIO_FREQ_HZ * t)
        # Short fade in/out to avoid clicks
        fade_len = max(1, int(0.01 * self.AUDIO_SAMPLE_RATE))
        fade = np.linspace(0, 1, fade_len)
        tone[:fade_len] *= fade
        tone[-fade_len:] *= fade[::-1]
        return tone.astype(np.float32)

    def _open_stream(self):
        try:
            stream = sd.OutputStream(
                samplerate=self.AUDIO_SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            stream.start()
            self.audio_ok = True
            self.audio_status_msg = ""
            return stream
        except Exception as e:
            self.audio_ok = False
            self.audio_status_msg = f"stream open failed: {e}"
            print(f"[ERROR] Could not open audio output stream: {e}")
            return None

    def _audio_watchdog(self):
        """Periodically checks the stream is alive; attempts to reopen it if
        it died (e.g. device unplugged). A dead stream never fails silently —
        self.audio_ok drives an on-screen warning."""
        if not SOUNDDEVICE_AVAILABLE:
            return
        while True:
            time.sleep(2.0)
            try:
                if self.stream is None or not self.stream.active:
                    raise RuntimeError("stream inactive")
                self.audio_ok = True
                self.audio_status_msg = ""
            except Exception as e:
                self.audio_ok = False
                self.audio_status_msg = "audio device lost — reconnecting..."
                print(f"[WARNING] Audio stream problem ({e}), attempting reopen.")
                try:
                    if self.stream is not None:
                        self.stream.close()
                except Exception:
                    pass
                self.stream = self._open_stream()

    def _audio_worker(self):
        """Runs on a background thread so writing audio never blocks the
        camera/detection loop. Writes the pre-generated tone into the
        already-open persistent stream (no per-call process spawn)."""
        while True:
            message = self.audio_queue.get()
            if message is None:
                break
            try:
                if self.stream is not None and self.audio_ok:
                    self.stream.write(self.tone_array.reshape(-1, 1))
                else:
                    # Fallback if audio is unavailable: terminal bell, never crash
                    print('\a', end='', flush=True)
            except Exception as e:
                self.audio_ok = False
                self.audio_status_msg = f"write failed: {e}"
                print(f"[ERROR] Audio write failed: {e}")

    def trigger_alarm(self, warning_type):
        if self.is_muted:
            return
        self.audio_queue.put("TRIGGER_ALARM")
        self.last_beep_time = time.time()

    def maybe_repeat_alarm(self, warning_type):
        """Call every frame while an episode is active. Re-beeps every
        AUDIO_REPEAT_INTERVAL_SEC, decoupled from logging."""
        if self.is_muted or self.AUDIO_REPEAT_INTERVAL_SEC <= 0:
            return
        if time.time() - self.last_beep_time >= self.AUDIO_REPEAT_INTERVAL_SEC:
            self.trigger_alarm(warning_type)

    def shutdown_audio(self):
        self.audio_queue.put(None)
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass

    #  MATH & GEOMETRY FUNCTIONS

    @staticmethod
    def calculate_ear(eye_landmarks, landmarks_array):
        coords = np.array([landmarks_array[idx] for idx in eye_landmarks])
        p2_p6 = np.linalg.norm(coords[1] - coords[5])
        p3_p5 = np.linalg.norm(coords[2] - coords[4])
        p1_p4 = np.linalg.norm(coords[0] - coords[3])
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)

    @staticmethod
    def calculate_mar(mouth_landmarks, landmarks_array):
        coords = np.array([landmarks_array[idx] for idx in mouth_landmarks])
        v1 = np.linalg.norm(coords[1] - coords[7])
        v2 = np.linalg.norm(coords[2] - coords[6])
        v3 = np.linalg.norm(coords[3] - coords[5])
        h  = np.linalg.norm(coords[0] - coords[4])
        return (v1 + v2 + v3) / (3.0 * h)

    def estimate_head_pose(self, landmarks_array, img_size):
        image_points = np.array([landmarks_array[idx] for idx in self.POSE_LANDMARKS], dtype=np.float32)
        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        rmat, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rmat, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        return euler_angles[0, 0], euler_angles[1, 0], euler_angles[2, 0]

    #  MAIN PIPELINE

    def run(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return

        print("\n[SYSTEM] Telemetry Engine Started.")
        print("[CONTROLS] Press 'm' to mute/unmute audio.")
        print("[CONTROLS] Press 'q' to quit.")

        prev_time = 0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print("[WARNING] Failed to read frame.")
                    break

                frame = cv2.resize(frame, (640, 480))
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
                prev_time = curr_time

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                status_color = (0, 255, 0)
                warning_text = "SAFE"
                avg_ear, mar, pitch_raw, pitch_delta, pitch_smooth = 0, 0, 0, 0, 0

                # CALIBRATION OVERLAY
                if self.pitch_baseline is None:
                    progress = len(self.calibration_pitches)
                    bar_w = int((progress / self.CALIBRATION_FRAMES) * (w - 40))
                    cv2.rectangle(frame, (20, h//2 - 40), (w - 20, h//2 + 40), (0, 0, 0), -1)
                    cv2.putText(frame, "Sit normally & look at screen...", (30, h//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Calibrating: {progress}/{self.CALIBRATION_FRAMES}", (30, h//2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.rectangle(frame, (20, h//2 + 50), (20 + bar_w, h//2 + 65), (0, 255, 0), -1)
                    cv2.imshow("Driver Telemetry Dashboard", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"): break

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            landmarks_array = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
                            pitch_raw, _, _ = self.estimate_head_pose(landmarks_array, (h, w))
                            self.calibration_pitches.append(pitch_raw)

                            if len(self.calibration_pitches) >= self.CALIBRATION_FRAMES:
                                self.pitch_baseline = float(np.median(self.calibration_pitches))
                                print(f"[INFO] Pitch baseline set to {self.pitch_baseline:.1f} degrees")
                    continue

                #  CORE DETECTION LOGIC
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks_array = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                        avg_ear = (self.calculate_ear(self.LEFT_EYE, landmarks_array) + self.calculate_ear(self.RIGHT_EYE, landmarks_array)) / 2.0
                        mar = self.calculate_mar(self.MOUTH, landmarks_array)
                        pitch_raw, _, _ = self.estimate_head_pose(landmarks_array, (h, w))

                        self.pitch_buffer.append(pitch_raw)
                        pitch_smooth = float(np.mean(self.pitch_buffer))
                        pitch_delta = pitch_smooth - self.pitch_baseline

                        eye_warn, yawn_warn, pitch_warn = False, False, False

                        # Metrics Counters
                        self.ear_counter = self.ear_counter + 1 if avg_ear < self.EAR_THRESH else max(0, self.ear_counter - 2)
                        self.mar_counter = self.mar_counter + 1 if mar > self.MAR_THRESH else max(0, self.mar_counter - 2)
                        self.pitch_counter = self.pitch_counter + 1 if pitch_delta < -self.PITCH_DELTA_THRESH else max(0, self.pitch_counter - 2)

                        if self.ear_counter >= self.EAR_CONSEC_FRAMES: eye_warn = True
                        if self.mar_counter >= self.MAR_CONSEC_FRAMES: yawn_warn = True
                        if self.pitch_counter >= self.PITCH_CONSEC_FRAMES: pitch_warn = True

                        # Priority Hierarchy
                        active_threat = None
                        if pitch_warn:
                            warning_text, status_color, active_threat = "WARNING: HEAD NOD", (0, 0, 255), "nod"
                        elif eye_warn:
                            warning_text, status_color, active_threat = "WARNING: DROWSINESS", (0, 0, 255), "eyes"
                        elif yawn_warn:
                            warning_text, status_color, active_threat = "WARNING: YAWNING", (0, 165, 255), "yawn"
                        else:
                            warning_text, status_color = "SAFE", (0, 255, 0)

                        # Episode-based alarm & logging: fires ONCE on the
                        # transition into a threat, not once per frame/cooldown-tick.
                        if active_threat:
                            if active_threat != self.active_episode:
                              
                                if self.active_episode is not None:
                                    duration = time.time() - self.episode_start_time
                                    self.log_telemetry("END", self.active_episode, avg_ear, mar, pitch_delta, duration)

                                self.active_episode = active_threat
                                self.episode_start_time = time.time()
                                self.trigger_alarm(active_threat)
                                self.log_telemetry("START", active_threat, avg_ear, mar, pitch_delta)
                            else:
                                # Same episode still ongoing: no re-log;
                                self.maybe_repeat_alarm(active_threat)
                        else:
                            if self.active_episode is not None:
                                duration = time.time() - self.episode_start_time
                                self.log_telemetry("END", self.active_episode, avg_ear, mar, pitch_delta, duration)
                                self.active_episode = None
                                self.episode_start_time = None

                        # Draw Landmarks
                        for idx in self.LEFT_EYE + self.RIGHT_EYE + self.MOUTH:
                            cv2.circle(frame, landmarks_array[idx], 1, status_color, -1)

                # DASHBOARD DISPLAY
                if self.pitch_baseline is not None:
                    cv2.rectangle(frame, (10, 10), (370, 180), (0, 0, 0), -1)
                    cv2.putText(frame, warning_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}  (thresh {self.EAR_THRESH})", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                    cv2.putText(frame, f"MAR: {mar:.2f}  (thresh {self.MAR_THRESH})", (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                    cv2.putText(frame, f"Pitch: {pitch_smooth:.1f}  base: {self.pitch_baseline:.1f}  delta: {pitch_delta:.1f}", (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                    cv2.putText(frame, f"Nod thresh: delta < -{self.PITCH_DELTA_THRESH}", (20, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
                    cv2.putText(frame, f"Ctrs  EAR:{self.ear_counter}  MAR:{self.mar_counter}  P:{self.pitch_counter}", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

                
                if not self.audio_ok:
                    cv2.rectangle(frame, (10, h - 45), (400, h - 15), (0, 0, 0), -1)
                    cv2.putText(frame, f"AUDIO ALERT UNAVAILABLE: {self.audio_status_msg}",
                                (18, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

                cv2.putText(frame, f"FPS: {int(fps)}", (w - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if fps > 15 else (0, 0, 255), 2)

                cv2.imshow("Driver Telemetry Dashboard", frame)

                #  KEYBOARD CONTROLS
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("m"):
                    self.is_muted = not self.is_muted
                    state = "MUTED" if self.is_muted else "ON"
                    print(f"[SYSTEM] Audio is now {state}")

        finally:
            # Close out any still-open episode so the log has a matching END row
            if self.active_episode is not None:
                duration = time.time() - self.episode_start_time
                self.log_telemetry("END", self.active_episode, avg_ear, mar, pitch_delta, duration)

            cap.release()
            cv2.destroyAllWindows()
            self.shutdown_audio()
            print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()