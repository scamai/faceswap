import insightface

MODEL = 'inswapper_128.onnx'


class FaceSwapper:
    def __init__(self):
        print(f'Face swapper loaded with model {MODEL}')
        self.swapper = insightface.model_zoo.get_model(MODEL)

    def swap_faces(self, target_image, target_face, source_face):
        return self.swapper.get(target_image, target_face, source_face, paste_back=True)


