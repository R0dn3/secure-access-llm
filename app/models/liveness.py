import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Simple blink-based liveness + texture-based cue (variance of Laplacian)
# Returns spoof_score in [0,1] where LOWER is more real
def estimate_spoof_score(image_rgb):
    h, w, _ = image_rgb.shape
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Texture sharpness (replay/photo often lower high-frequency content)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalize rough heuristic to [0,1] (project-specific tuning)
    tex_score = 1.0 - np.tanh(lap_var / 200.0)  # lower variance => higher spoof prob

    # Eye-aspect-ratio like heuristic via face mesh (single frame proxy)
    blink_score = 0.5  # neutral
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
        res = fm.process(image_rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            # sample a few points around eyes to approximate openness (very rough)
            def pt(i):
                return np.array([lm[i].x * w, lm[i].y * h])
            # Right eye indices (MediaPipe canonical) - coarse selection
            upper = pt(159); lower = pt(145)
            dist = np.linalg.norm(upper - lower)
            blink_score = 1.0 - np.tanh(dist / 5.0)  # smaller vertical dist => more closed => possibly blink

    # Combine (you can train a small classifier to fuse better)
    spoof_score = float(np.clip(0.6*tex_score + 0.4*blink_score, 0.0, 1.0))
    return spoof_score
