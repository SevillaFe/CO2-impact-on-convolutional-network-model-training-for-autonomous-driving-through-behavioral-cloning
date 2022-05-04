import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
 
sio = socketio.Server()
app = Flask(__name__) #'__main__'

#speed_limit = 20

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral

controller = SimplePIController(0.1, 0.002)
set_speed = 20
controller.set_desired(set_speed)

def img_preprocess(img):
    img = img[40:140, :, :]
    #img = img[60:-25, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66), cv2.INTER_AREA)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img =  (img - 128.) / 128 #(img - 128.) / 128
    return img 
 
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data['image']))) 
        image = np.asarray(image) # from PIL image to numpy array
        image = img_preprocess(image) # apply the preprocessing
        image = np.array([image]) # the model expects 4D array
        # predict the steering angle for the image
        steering_angle = float(model.predict(image))

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)
 
 
 
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)

 
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)
 
 
if __name__ == '__main__':
    model = load_model('model_nor.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)