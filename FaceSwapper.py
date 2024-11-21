import insightface
from constants import PROVIDER, FACE_SWAP_MODEL


class FaceSwapper:
    def __init__(self):
        print(f'Face swapper loaded with model {FACE_SWAP_MODEL}')
        self.swapper = insightface.model_zoo.get_model(FACE_SWAP_MODEL, providers=[PROVIDER])

    def swap_faces(self, target_image, target_face, source_face):
        return self.swapper.get(target_image, target_face, source_face, paste_back=True)


