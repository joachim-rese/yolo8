from flask import Flask, request, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import json
from paddleocr import PaddleOCR, draw_ocr
from pylibdmtx.pylibdmtx import decode
import os


TAG_PREFIX = 'HU'
TAG_LEN0 = 6
TAG_LEN1 = 4

PORT = os.getenv('PORT') 

app = Flask(__name__)

Model = YOLO('resources/model.pt')
OCR = PaddleOCR(use_angle_cls=True, rec_model='CRNN', lang='en')

@app.route('/res', methods=['GET'])
def res():
    filepath = os.path.join(app.root_path, 'img.jpg')
    return send_file(filepath, as_attachment=False)

@app.route('/', methods=['GET','POST'])
def infer():
    if request.method == 'GET':
        return '''Server Works!<hr>
            <form action="/" method="POST" enctype="multipart/form-data">
            <input type="file" name="image">
            <button>OK</button>
            </form>    
            '''
   
    byte_str = request.files["image"].read()

    image = cv2.imdecode(np.fromstring(byte_str, np.uint8), 1)


    result = Model(image)[0].cpu().boxes
    conf = result.conf
    boxes = result.xyxy.numpy().round().astype(int)

    message = {'tags': [], 'labels': []}
    for index, box in enumerate(boxes):
        if conf[index] >= 0.33:
            (x0, y0, x1, y1) = box
            image_cropped = image[box[1]:box[3], box[0]:box[2]]
            decoder, tag = analyze(image_cropped)
            message['tags'].append({'decoder': decoder, 'tag': tag})
            _, buffer = cv2.imencode('.png', image_cropped)
            image_cropped_base64 = base64.b64encode(buffer).decode('ascii')
            message['labels'].append(image_cropped_base64)

            cv2.rectangle(image, (x0,y0), (x1,y1), (255,255,0), 5)
            cv2.putText(image, decoder+'-'+tag, (x0,y0), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,0), 2, cv2.LINE_8, False)

    cv2.imwrite('img.jpg', image)
    print("Image written")

    ret = json.dumps(message)
    return ret


def analyze(image_label):
    tag = ''
    decoder = 'none'

    # (1) Read datamatrix
    result = decode(image_label)
    # expected result:
    # [Decoded(data=b'[)> 06;HU1178323874;PA9736100325;Q500;2S86267730', rect=Rect(left=67, top=71, width=107, height=120))]
    print(str(result))

    if len(result) >= 1:

        tags = [x for x in result[0].data.decode('ascii').split(';') if x.startswith(TAG_PREFIX)]
        if len(tags) == 1:
            tag = tags[0]
            decoder = 'dmtx'

    if tag == '':
        # (2) Perform OCR

        result = OCR.ocr(image_label)[0]
        # expexted result:
        # [ [[[175.0, 16.0], [322.0, 9.0], [325.0, 66.0], [178.0, 73.0]], ('3874', 0.9949043393135071)],
        #   [[[98.0, 32.0], [173.0, 27.0], [175.0, 52.0], [100.0, 58.0]], ('117832', 0.9982736110687256)],
        #   [[[206.0, 89.0], [305.0, 82.0], [307.0, 109.0], [208.0, 116.0]], ('01.08.23', 0.9938185811042786)],
        #   [[[238.0, 118.0], [284.0, 118.0], [284.0, 156.0], [238.0, 156.0]], ('38', 0.997283935546875)],
        #   [[[50.0, 182.0], [348.0, 161.0], [351.0, 205.0], [53.0, 226.0]], ('001-01-019', 0.9946553111076355)] ]

        if result != None and len(result) >= 2:
            
            #sort text boxes by lower boundary
            result.sort(key=lambda x: x[0][3][1])
            tag0 = str(result[0][1][0])
            tag1 = str(result[1][1][0])
            
            # correct tag, if necessary (label might be rotated)
            if len(tag0) != TAG_LEN0:
                tag0 = '?'
                for idx in range(len(result)):
                    _tag0 = str(result[idx][1][0])
                    if len(_tag0) == TAG_LEN0:
                        tag0 = _tag0
                        break
            if len(tag1) != TAG_LEN1:
                tag1 = '?'
                for idx in range(len(result)):
                    _tag1 = str(result[idx][1][0])
                    if len(_tag1) == TAG_LEN1:
                        tag1 = _tag1
                        break 

            tag = tag0 + tag1
            decoder = 'ocr'


    if not tag.startswith(TAG_PREFIX):
        tag = TAG_PREFIX + tag

    return decoder, tag


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)

                         