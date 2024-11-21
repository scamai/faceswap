from flask import Flask, request, jsonify, send_file
from FaceSwapper import FaceSwapper
from Preprocessor import Preprocessor
import cv2
from io import BytesIO
import numpy as np
from PIL import Image

app = Flask(__name__)
preprocessor = Preprocessor()
face_swapper = FaceSwapper()


def _convert_to_image(file_storage):
    image_bytes = file_storage.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image


@app.route('/swap', methods=['POST'])
def swap_faces():
    if 'source' not in request.files:
        return jsonify({'error': 'No source image provided'}), 400
    if 'target' not in request.files:
        return jsonify({'error': 'No target image provided'}), 400

    source_image_original = _convert_to_image(request.files['source'])
    target_image_original = _convert_to_image(request.files['target'])
    source_image = source_image_original.copy()
    target_image = target_image_original.copy()
    source_face, source_y0_new, source_y1_new, source_x0_new, source_x1_new = preprocessor.get_cropped_face(source_image)
    target_face, target_y0_new, target_y1_new, target_x0_new, target_x1_new = preprocessor.get_cropped_face(target_image)

    if source_face is None:
        return jsonify({'error': 'No face detected in source target image'}), 400
    if target_face is None:
        return jsonify({'error': 'No face detected in target image'}), 400

    source_face_app_format = preprocessor.get(source_face)
    target_face_app_format = preprocessor.get(target_face)

    if len(source_face_app_format) == 0:
        return jsonify({'error': 'No face detected in source image'}), 400
    if len(target_face_app_format) == 0:
        return jsonify({'error': 'No face detected in target image'}), 400

    swapped_face = face_swapper.swap_faces(target_face, target_face_app_format[0], source_face_app_format[0])
    target_image_original[target_y0_new:target_y1_new, target_x0_new:target_x1_new] = swapped_face

    target_image_original = cv2.cvtColor(target_image_original, cv2.COLOR_BGR2RGB)
    target_image_pil = Image.fromarray(np.uint8(target_image_original))
    img_io = BytesIO()
    target_image_pil.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run()
