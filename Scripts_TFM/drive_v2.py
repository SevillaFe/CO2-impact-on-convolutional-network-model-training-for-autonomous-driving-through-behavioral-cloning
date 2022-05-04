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

MAX_SPEED = 20
MIN_SPEED = 15
speed_limit = MAX_SPEED

# Preprocess
def img_preprocess(img):
    img = img[40:140, :, :] 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66), cv2.INTER_AREA)
    img = (img - 128.) / 128 
    return img 
 
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data['image']))) 
        image = np.asarray(image) # from PIL image to numpy array
        image = img_preprocess(image) # apply the preprocessing
        image = np.array([image]) # the model expects 4D array
        # predict the steering angle for the image
        steering_angle = float(model.predict(image)) #batch_size=1

        global speed_limit
        if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
        else:
                speed_limit = MAX_SPEED
        throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

        print('{} {} {}'.format(steering_angle, throttle, speed))
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
    model = load_model('model_VGG.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)