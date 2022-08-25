from flask import Flask, render_template, Response,jsonify,request
import cv2
import numpy as np
import os
import sys

app=Flask(__name__)


@app.route('/predict_image',methods=['POST'])
def Predict_Image():
  
    image = request.files['image'].read()
    npimg = np.fromstring(image, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

    cv2.imshow('image',img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    dic = {"status":200,"msg":"ok"}

    return jsonify(dic)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
