import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from setting import *

class FaceEmbedder:

    def __init__(self):
        self.app = FaceAnalysis(name=MODEL_NAME)

        self.app.prepare(
            ctx_id=0,
            det_size=DETECTION_SIZE
        )

    def get_embedding(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            return None

        faces = self.app.get(image)

        if len(faces) == 0:
            return None

        valid_faces = []

        for face in faces:

            x1, y1, x2, y2 = face.bbox.astype(int)

            area = (x2 - x1) * (y2 - y1)

            if area < MIN_FACE_AREA:
                continue

            if face.det_score < MIN_FACE_CONFIDENCE:
                continue

            valid_faces.append(face)

        if len(valid_faces) == 0:
            return None

        face = max(
            valid_faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )
        )

        return face.embedding