import cv2
from yolov6.layers.common import DetectBackend
from Yolov6Util import FrameProcessor
from yolov6.utils.events import load_yaml
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest, InternalServerError
import ssl
import urllib.request
import numpy as np

app = Flask(__name__)
context = ssl.SSLContext()
device:str = "cpu" #@param ["gpu", "cpu"]
half:bool = False #@param {type:"boolean"}
img_size:int = 640 #@param {type:"integer"}
model = DetectBackend(f"license_plate_detector_ckpt.pt", device=device)
frameProcessor = FrameProcessor(img_size, half,  device, model)

@app.route('/detections', methods=['POST'])
def detectVehicleRegistrationPlate():
  response = list()
  for imageId in request.get_json():
    response_data = dict()
    response_data['imageId'] = imageId
    response_data['registrationPlate'] = list()
    try: 
      with urllib.request.urlopen('https://storage.googleapis.com/'+imageId, context=context) as resp:
        image = np.asarray(bytearray(resp.read()), dtype="uint8") 
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        registrationPlateIndex = 1
        registrationPlate = list()
        for x1, y1, x2, y2, conf, _ in frameProcessor.process_frame(image):
          if float(conf) > 0.5:
            registrationPlateItem =  dict()
            registrationPlateItem['registrationPlateIndex'] = registrationPlateIndex
            registrationPlateIndex += 1
            registrationPlateItem['registrationPlateCoordinates'] = dict()
            registrationPlateItem['registrationPlateCoordinates']['x1'] = int(x1.item())
            registrationPlateItem['registrationPlateCoordinates']['y1'] = int(y1.item())
            registrationPlateItem['registrationPlateCoordinates']['x2'] = int(x2.item())
            registrationPlateItem['registrationPlateCoordinates']['y2'] = int(y2.item())
            registrationPlate.append(registrationPlateItem)
        response_data['registrationPlate'].append(registrationPlate)
    except Exception as ex:
       raise BadRequest(description='Exception occured: {}. {} while processing {}.'.format(type(ex).__name__, ex, request_data))
    response.append(response_data)
  return app.make_response((response, 200))

@app.errorhandler(BadRequest)
def handle_invalid_usage(error):
    return app.make_response((jsonify(message=error.description), 400))

@app.errorhandler(InternalServerError)
def handle_internal_server_error(error):
    return app.make_response((jsonify(message=error.description), 500))

@app.errorhandler(Exception)
def handle_internal_server_error(error):
    return app.make_response((jsonify(message=str(error)), 500))

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8081)
