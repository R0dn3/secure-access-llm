import cv2
import numpy as np
import mediapipe as mp

mp_face = mp.solutions.face_detection

def detect_and_crop_face(image_rgb, target_size=(112, 112)):

    """
    Detecta y recorta un rostro de la imagen, usado en el entrenamiento.
    """
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.2) as fd:
        results = fd.process(image_rgb)
        if not results.detections:
            return None

        # Tomamos la detección con mayor confianza
        det = max(results.detections, key=lambda d: d.score[0])
        bbox = det.location_data.relative_bounding_box
        h, w, _ = image_rgb.shape
        x1 = max(int(bbox.xmin * w), 0)
        y1 = max(int(bbox.ymin * h), 0)
        x2 = min(int((bbox.xmin + bbox.width) * w), w - 1)
        y2 = min(int((bbox.ymin + bbox.height) * h), h - 1)

        face = image_rgb[y1:y2, x1:x2]
        if face.size == 0:
            return None

        face = cv2.resize(face, target_size)
        return face


def detect_faces(frame):
    """
    Detecta TODOS los rostros en la imagen y devuelve una lista de coordenadas.
    Usado en camera_verify.py para dibujar rectángulos en tiempo real.
    """
    h, w, _ = frame.shape
    face_list = []

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as fd:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = fd.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                face_list.append((x, y, width, height))

    return face_list
