from insightface.app import FaceAnalysis
import numpy as np


class Preprocessor:
    def __init__(self):
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=0)
        print('Preprocessor loaded')

    @staticmethod
    def __crop_face(img, bbox):
        H, W = len(img), len(img[0])
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1 - x0
        h = y1 - y0

        w0_margin = w / 4
        w1_margin = w / 4
        h0_margin = h / 4
        h1_margin = h / 4

        w0_margin *= 2
        w1_margin *= 2
        h0_margin *= 1
        h1_margin *= 1

        y0_new = max(0, int(y0 - h0_margin))
        y1_new = min(H, int(y1 + h1_margin) + 1)
        x0_new = max(0, int(x0 - w0_margin))
        x1_new = min(W, int(x1 + w1_margin) + 1)

        img_cropped = img[y0_new:y1_new, x0_new:x1_new]
        bbox_cropped = np.zeros_like(bbox)
        for i, (p, q) in enumerate(bbox):
            bbox_cropped[i] = [p - x0_new, q - y0_new]
        return img_cropped, y0_new, y1_new, x0_new, x1_new

    def get_cropped_face(self, face_img):
        faces = self.app.get(face_img)
        if len(faces) == 0:
            return None
        bbox = faces[0]['bbox']
        bbox = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
        return Preprocessor.__crop_face(face_img, bbox)

    def get(self, face_image):
        return self.app.get(face_image)
