import cv2
import math
import time
import logging
import numpy as np
import mediapipe as mp
from enum import Enum
from collections import deque
from typing import Tuple, List

# Konfigurasi Logging Standar Industri
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class Intensi(Enum):
    """Enumerasi Status untuk State Machine yang lebih aman dan rapi"""
    STANDBY = "STANDBY"
    TERTUTUP_PENUH = "TERTUTUP PENUH"
    KEDIP_KIRI = "KEDIP KIRI"
    KEDIP_KANAN = "KEDIP KANAN"
    ATAS = "LIRIK ATAS"
    BAWAH = "LIRIK BAWAH"
    KIRI = "LIRIK KIRI"
    KANAN = "LIRIK KANAN"

class AssistiveGazeTracker:
    """
    Sistem Navigasi 7-Arah Berbasis MediaPipe FaceMesh.
    Dirancang khusus untuk Aksesibilitas Disabilitas Motorik.
    """
    
    # --- KONFIGURASI TITIK WAJAH ---
    LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
    LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER = 33, 133
    LEFT_IRIS_CENTER = 468
    LEFT_IRIS_EDGES = [469, 470, 471, 472]
    
    RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
    RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER = 362, 263
    RIGHT_IRIS_CENTER = 473
    RIGHT_IRIS_EDGES = [474, 475, 476, 477]

    def __init__(self, buffer_size: int = 5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Buffer Data untuk Smoothing
        self.ear_kiri_buffer = deque(maxlen=buffer_size)
        self.ear_kanan_buffer = deque(maxlen=buffer_size)
        self.gaze_x_buffer = deque(maxlen=buffer_size)
        self.gaze_y_buffer = deque(maxlen=buffer_size)
        
        # Manajemen State (Status)
        self.current_state = Intensi.STANDBY
        self.target_dwell_time = 0.0
        self.state_start_time = 0.0
        self.last_action_executed = ""
        self.action_display_timer = 0.0

    @staticmethod
    def _hitung_jarak(p1, p2) -> float:
        return math.dist([p1.x, p1.y], [p2.x, p2.y])

    def _ekstrak_metrik(self, landmarks) -> Tuple[float, float, float, float]:
        """Menghitung metrik EAR dan Gaze dari koordinat landmark."""
        l_width = self._hitung_jarak(landmarks[self.LEFT_EYE_LEFT_CORNER], landmarks[self.LEFT_EYE_RIGHT_CORNER])
        l_height = self._hitung_jarak(landmarks[self.LEFT_EYE_TOP], landmarks[self.LEFT_EYE_BOTTOM])
        r_width = self._hitung_jarak(landmarks[self.RIGHT_EYE_LEFT_CORNER], landmarks[self.RIGHT_EYE_RIGHT_CORNER])
        r_height = self._hitung_jarak(landmarks[self.RIGHT_EYE_TOP], landmarks[self.RIGHT_EYE_BOTTOM])

        ear_kiri = l_height / l_width if l_width > 0 else 0.0
        ear_kanan = r_height / r_width if r_width > 0 else 0.0
        
        gaze_x, gaze_y = 0.5, 0.5
        if l_width > 0 and r_width > 0:
            l_gaze_x = self._hitung_jarak(landmarks[self.LEFT_EYE_LEFT_CORNER], landmarks[self.LEFT_IRIS_CENTER]) / l_width
            r_gaze_x = self._hitung_jarak(landmarks[self.RIGHT_EYE_LEFT_CORNER], landmarks[self.RIGHT_IRIS_CENTER]) / r_width
            gaze_x = (l_gaze_x + r_gaze_x) / 2.0

        if l_height > 0 and r_height > 0:
            l_gaze_y = self._hitung_jarak(landmarks[self.LEFT_EYE_TOP], landmarks[self.LEFT_IRIS_CENTER]) / l_height
            r_gaze_y = self._hitung_jarak(landmarks[self.RIGHT_EYE_TOP], landmarks[self.RIGHT_IRIS_CENTER]) / r_height
            gaze_y = (l_gaze_y + r_gaze_y) / 2.0
            
        return ear_kiri, ear_kanan, gaze_x, gaze_y

    def _tentukan_intensi(self, ear_kiri: float, ear_kanan: float, gaze_x: float, gaze_y: float) -> Tuple[Intensi, float]:
        """Mengembalikan objek Enum Intensi beserta waktu tahan (dwell time) yang dibutuhkan."""
        EAR_TUTUP = 0.16
        EAR_BUKA = 0.20

        if ear_kiri < EAR_TUTUP and ear_kanan < EAR_TUTUP:
            return Intensi.TERTUTUP_PENUH, 2.0
        elif ear_kiri < EAR_TUTUP and ear_kanan > EAR_BUKA:
            return Intensi.KEDIP_KIRI, 0.8
        elif ear_kanan < EAR_TUTUP and ear_kiri > EAR_BUKA:
            return Intensi.KEDIP_KANAN, 0.8
            
        if gaze_y < 0.38: return Intensi.ATAS, 1.2
        elif gaze_y > 0.65: return Intensi.BAWAH, 1.2
        elif gaze_x < 0.40: return Intensi.KANAN, 1.2
        elif gaze_x > 0.60: return Intensi.KIRI, 1.2
            
        return Intensi.STANDBY, 0.0

    def _gambar_detail_mata(self, frame, landmarks, w: int, h: int):
        """Merender pelacakan iris mata dengan tingkat akurasi tinggi (Crosshair & Edges)."""
        for center_idx, edge_indices in [(self.LEFT_IRIS_CENTER, self.LEFT_IRIS_EDGES), 
                                         (self.RIGHT_IRIS_CENTER, self.RIGHT_IRIS_EDGES)]:
            
            # 1. Menggambar tepi Iris (Polygon Lingkaran)
            iris_points = []
            for idx in edge_indices:
                pt = landmarks[idx]
                iris_points.append([int(pt.x * w), int(pt.y * h)])
            
            iris_pts = np.array(iris_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [iris_pts], isClosed=True, color=(0, 255, 0), thickness=1)
            
            # 2. Menggambar Titik Tengah (Pupil Crosshair)
            pt_center = landmarks[center_idx]
            cx, cy = int(pt_center.x * w), int(pt_center.y * h)
            
            # Marker Crosshair Merah yang profesional
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, 
                           markerSize=8, thickness=1, line_type=cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 1, (0, 255, 255), -1)

    def _eksekusi_aksi(self, intensi: Intensi):
        """Memetakan Intensi pengguna menjadi aksi nyata."""
        map_aksi = {
            Intensi.TERTUTUP_PENUH: "[SISTEM] CLOSE APP",
            Intensi.ATAS: "[DOKUMEN] SCROLL UP",
            Intensi.BAWAH: "[DOKUMEN] SCROLL DOWN",
            Intensi.KIRI: "[HALAMAN] PREV PAGE",
            Intensi.KANAN: "[HALAMAN] NEXT PAGE",
            Intensi.KEDIP_KIRI: "[KLIK] MOUSE ENTER / KIRI",
            Intensi.KEDIP_KANAN: "[KLIK] MOUSE BACK / KANAN"
        }
        self.last_action_executed = map_aksi.get(intensi, "")
        logging.info(f"AKSI TERPICU: {self.last_action_executed}")

    def proses_frame(self, frame):
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Render visual pelacakan
            self._gambar_detail_mata(frame, landmarks, w, h)

            # Ekstrak metrik
            ear_kiri, ear_kanan, gaze_x, gaze_y = self._ekstrak_metrik(landmarks)
            self.ear_kiri_buffer.append(ear_kiri)
            self.ear_kanan_buffer.append(ear_kanan)
            self.gaze_x_buffer.append(gaze_x)
            self.gaze_y_buffer.append(gaze_y)

            if len(self.ear_kiri_buffer) == self.ear_kiri_buffer.maxlen:
                avg_ear_kiri = sum(self.ear_kiri_buffer) / len(self.ear_kiri_buffer)
                avg_ear_kanan = sum(self.ear_kanan_buffer) / len(self.ear_kanan_buffer)
                avg_gaze_x = sum(self.gaze_x_buffer) / len(self.gaze_x_buffer)
                avg_gaze_y = sum(self.gaze_y_buffer) / len(self.gaze_y_buffer)

                intensi, req_dwell_time = self._tentukan_intensi(avg_ear_kiri, avg_ear_kanan, avg_gaze_x, avg_gaze_y)

                if intensi != self.current_state:
                    self.current_state = intensi
                    self.target_dwell_time = req_dwell_time
                    self.state_start_time = time.time()
                else:
                    if self.current_state != Intensi.STANDBY:
                        elapsed = time.time() - self.state_start_time
                        progress = min(elapsed / self.target_dwell_time, 1.0)
                        
                        # Progress Bar Dinamis
                        bar_w = int(300 * progress)
                        cv2.rectangle(frame, (50, 400), (350, 430), (40, 40, 40), -1) 
                        cv2.rectangle(frame, (50, 400), (50 + bar_w, 430), (0, 255, 150), -1) 
                        cv2.putText(frame, f"Memproses [{self.current_state.value}]: {int(progress * 100)}%", 
                                    (60, 422), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        if elapsed >= self.target_dwell_time:
                            self._eksekusi_aksi(self.current_state)
                            self.action_display_timer = time.time()
                            self.state_start_time = time.time() 

            # Tampilan UI Atas
            if time.time() - self.action_display_timer < 2.5 and self.last_action_executed != "":
                cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
                cv2.putText(frame, self.last_action_executed, (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (50, 255, 255), 3)
            else:
                cv2.putText(frame, f"Niat: {self.current_state.value}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def rilis(self):
        self.face_mesh.close()

def main():
    logging.info("Sistem Smart Reading Diinisialisasi. Tekan 'Q' pada jendela kamera untuk keluar.")
    cap = cv2.VideoCapture(0)
    tracker = AssistiveGazeTracker()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            logging.error("Gagal membaca frame kamera.")
            break
        
        processed_frame = tracker.proses_frame(frame)
        cv2.imshow('Sistem Assistive Gaze - Pro Version', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Sistem dihentikan oleh pengguna.")
            break

    cap.release()
    tracker.rilis()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()