# pattern/pattern_auth.py
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time

class PatternAuth:
    def __init__(self, data_path="pattern/pattern_data.pkl",
                 fps_sample=30, duration_default=4,
                 ema_alpha=0.3, min_move_thresh=0.003,
                 smooth_window=5, resample_n=100, dtw_threshold=0.05,
                 min_attempts_for_enroll=2, max_line_jump=0.2):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.data_path = data_path
        self.patterns = self._load_patterns()

        # parámetros
        self.fps_sample = fps_sample
        self.duration_default = duration_default
        self.ema_alpha = ema_alpha
        self.min_move_thresh = min_move_thresh
        self.smooth_window = smooth_window
        self.resample_n = resample_n
        self.dtw_threshold = dtw_threshold  # valor base (fallback)
        self.min_attempts_for_enroll = min_attempts_for_enroll
        self.max_line_jump = max_line_jump  # salto máximo permitido entre puntos (0–1 normalizado)

    # -------------------------
    # I/O patrones
    # -------------------------
    def _load_patterns(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_patterns(self):
        os.makedirs(os.path.dirname(self.data_path) or ".", exist_ok=True)
        with open(self.data_path, "wb") as f:
            pickle.dump(self.patterns, f)

    # -------------------------
    # utilidades señales
    # -------------------------
    def _stable_point_from_landmarks(self, lm):
        nose_idx = 1
        left_eye_idx = 33
        right_eye_idx = 263

        nose = lm[nose_idx]
        left = lm[left_eye_idx]
        right = lm[right_eye_idx]

        eye_mid = ((left.x + right.x) / 2.0, (left.y + right.y) / 2.0)
        px = 0.7 * nose.x + 0.3 * eye_mid[0]
        py = 0.7 * nose.y + 0.3 * eye_mid[1]
        return (px, py)

    def _ema(self, prev, cur, alpha):
        if prev is None:
            return cur
        return (alpha * np.array(cur) + (1 - alpha) * np.array(prev)).tolist()

    def _moving_average(self, traj, w):
        if w <= 1:
            return np.array(traj)
        traj = np.array(traj)
        kernel = np.ones(w) / w
        x = np.convolve(traj[:,0], kernel, mode='same')
        y = np.convolve(traj[:,1], kernel, mode='same')
        return np.stack([x, y], axis=1)

    def _resample(self, traj, n):
        traj = np.array(traj)
        if len(traj) == 0:
            return np.zeros((n,2))
        if len(traj) == 1:
            return np.repeat(traj, n, axis=0)
        d = np.sqrt(((traj[1:] - traj[:-1])**2).sum(axis=1))
        s = np.concatenate([[0.0], np.cumsum(d)])
        total = s[-1]
        if total == 0:
            return np.repeat(traj[:1], n, axis=0)
        s_norm = s / total
        query = np.linspace(0, 1, n)
        resampled = []
        for q in query:
            idx = np.searchsorted(s_norm, q) - 1
            idx = np.clip(idx, 0, len(traj)-2)
            t0, t1 = s_norm[idx], s_norm[idx+1]
            ratio = (q - t0) / (t1 - t0 + 1e-8)
            p = traj[idx] * (1 - ratio) + traj[idx+1] * ratio
            resampled.append(p)
        return np.array(resampled)

    def _normalize_traj(self, traj):
        arr = np.array(traj)
        if arr.size == 0:
            return np.zeros((self.resample_n,2))
        mn = arr.min(axis=0)
        mx = arr.max(axis=0)
        span = mx - mn
        span[span == 0] = 1.0
        norm = (arr - mn) / span
        return self._resample(norm, self.resample_n)

    # -------------------------
    # DTW
    # -------------------------
    def _dtw_distance(self, a, b):
        a = np.array(a)
        b = np.array(b)
        na, nb = len(a), len(b)
        D = np.full((na+1, nb+1), np.inf)
        D[0,0] = 0.0
        for i in range(1, na+1):
            for j in range(1, nb+1):
                cost = np.linalg.norm(a[i-1] - b[j-1])
                D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
        return D[na, nb] / (na + nb)

    # -------------------------
    # Captura trayectoria
    # -------------------------
    def _capture_trajectory(self, duration=None, show_window=True):
        if duration is None:
            duration = self.duration_default
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir la cámara")

        traj = []
        ema_prev = None
        start = time.time()
        wait_ms = int(1000 / max(1, self.fps_sample))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            h, w = frame.shape[:2]
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                pt = self._stable_point_from_landmarks(lm)
                ema_prev = self._ema(ema_prev, pt, self.ema_alpha)

                if len(traj) == 0:
                    traj.append(ema_prev)
                else:
                    last = traj[-1]
                    dx = abs(ema_prev[0] - last[0])
                    dy = abs(ema_prev[1] - last[1])
                    if max(dx, dy) >= self.min_move_thresh:
                        traj.append(ema_prev)

                px, py = int(ema_prev[0] * w), int(ema_prev[1] * h)
                cv2.circle(frame, (px, py), 3, (0,0,255), -1)
            else:
                cv2.putText(frame, "No face detected", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            if show_window:
                cv2.imshow("Pattern Capture (press q to abort)", frame)

            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
            if (time.time() - start) >= duration:
                break

        cap.release()
        if show_window:
            cv2.destroyAllWindows()

        if len(traj) > 0:
            traj_sm = self._moving_average(traj, self.smooth_window)
            return traj_sm.tolist()
        else:
            return []

    # -------------------------
    # Enroll / Verify / Reset
    # -------------------------
    def enroll_pattern(self, username, attempts=2, duration=None):
        if duration is None:
            duration = self.duration_default
        samples = []
        for a in range(attempts):
            print(f"[ENROLL] Captura {a+1}/{attempts} - dibuja el patrón frente a la cámara")
            traj = self._capture_trajectory(duration=duration, show_window=True)
            traj_norm = self._normalize_traj(traj)

            # Mostrar trazo (con filtro de saltos grandes)
            canvas = 255 * np.ones((400, 400, 3), dtype=np.uint8)
            h, w = canvas.shape[:2]
            pts = [(int(x*w), int(y*h)) for (x,y) in traj_norm]
            for i in range(1, len(pts)):
                dist = np.linalg.norm(np.array(pts[i]) - np.array(pts[i-1])) / max(h,w)
                if dist < self.max_line_jump:  # filtro de saltos largos
                    cv2.line(canvas, pts[i-1], pts[i], (0,0,255), 2)

            cv2.imshow(f"Patron Captura {a+1}", canvas)
            print(f"[ENROLL] Captura {a+1} -> {len(traj)} puntos (resample -> {traj_norm.shape})")
            print("Pulsa 'y' para guardar, 'n' para repetir, 'q' para cancelar")

            # esperar decisión en ventana
            resp = None
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == ord('y'):
                    resp = "y"
                    break
                elif k == ord('n'):
                    resp = "n"
                    break
                elif k == ord('q'):
                    resp = "q"
                    break

            cv2.destroyWindow(f"Patron Captura {a+1}")

            if resp == "y":
                samples.append(traj_norm)
            elif resp == "n":
                print("[ENROLL] Rehaciendo captura...")
                return self.enroll_pattern(username, attempts=attempts, duration=duration)
            elif resp == "q":
                print("[ENROLL] Cancelado por el usuario.")
                return False

        if samples:
            avg = np.mean(np.stack(samples, axis=0), axis=0)

            # 🔑 calcular umbral dinámico según variación entre muestras
            intra_dists = []
            for i in range(len(samples)):
                for j in range(i+1, len(samples)):
                    intra_dists.append(self._dtw_distance(samples[i], samples[j]))
            user_threshold = float(np.mean(intra_dists) * 1.3) if intra_dists else self.dtw_threshold

            self.patterns[username] = {
                "pattern": avg,
                "threshold": user_threshold
            }
            self._save_patterns()
            print(f"[ENROLL] Patrón guardado para {username} (samples={len(samples)}, threshold={user_threshold:.4f})")
            return True
        else:
            print("[ENROLL] No se guardaron capturas válidas.")
            return False

    def verify_pattern(self, username, duration=None):
        if duration is None:
            duration = self.duration_default
        if username not in self.patterns:
            print(f"[VERIFY] No existe patrón para {username}")
            return False

        print(f"[VERIFY] Capturando intento para {username} (dur={duration}s).")
        traj = self._capture_trajectory(duration=duration, show_window=True)
        if len(traj) == 0:
            print("[VERIFY] No se capturó trayectoria válida.")
            return False

        traj_norm = self._normalize_traj(traj)

        # Dibujar comparación
        canvas = 255 * np.ones((400, 400, 3), dtype=np.uint8)
        h, w = canvas.shape[:2]
        pts = [(int(x*w), int(y*h)) for (x,y) in traj_norm]
        for i in range(1, len(pts)):
            dist = np.linalg.norm(np.array(pts[i]) - np.array(pts[i-1])) / max(h,w)
            if dist < self.max_line_jump:
                cv2.line(canvas, pts[i-1], pts[i], (0,255,0), 2)
        cv2.imshow("Verify Pattern", canvas)
        cv2.waitKey(1000)
        cv2.destroyWindow("Verify Pattern")

        stored_info = self.patterns[username]
        if isinstance(stored_info, dict):
            stored = stored_info["pattern"]
            threshold = stored_info.get("threshold", self.dtw_threshold)
        else:
            # compatibilidad con patrones antiguos
            stored = stored_info
            threshold = self.dtw_threshold

        dist = self._dtw_distance(stored, traj_norm)
        print(f"[VERIFY] DTW distance = {dist:.5f} (threshold={threshold})")
        return dist <= threshold

    def reset_pattern(self, username):
        if username in self.patterns:
            del self.patterns[username]
            self._save_patterns()
            print(f"[INFO] Patrón eliminado para usuario: {username}")
            return True
        print("[WARN] No existe patrón.")
        return False


if __name__ == "__main__":
    pa = PatternAuth()
    name = input("Nombre usuario para enrolar: ").strip()
    pa.enroll_pattern(name, attempts=2, duration=4)
    ok = pa.verify_pattern(name, duration=3)
    print("Resultado verify:", ok)
