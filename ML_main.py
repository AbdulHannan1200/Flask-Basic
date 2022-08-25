from flask import Flask, render_template, Response,jsonify,request
import cv2
import numpy as np
import os
import sys
from tensorflow import keras
import numpy as np

app=Flask(__name__)
IMG_SIZE=180

@app.route('/predict_image',methods=['POST'])
def Predict_Image():
  
    image = request.files['image'].read()
    npimg = np.fromstring(image, np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR) #This img variable will have your image

    resized_img = cv2.resize(img,(IMG_SIZE,IMG_SIZE)) #resizing the image
    resized_img = resized_img.reshape(-1,IMG_SIZE,IMG_SIZE,3) #reshaping the image

    print(type(resized_img));print(resized_img);print(resized_img.shape)
    
    model = keras.models.load_model('model2.h5')

    prediction = model.predict(resized_img)[0][0]

    print("Prediction: ",prediction)

    if prediction<=0.5:
        prediction_class = 'Cat'
    elif prediction>0.5:
        prediction_class = 'Dog'

    print("Prediction Label: ",prediction_class)

    # cv2.imshow('image',img)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 

    dic = {"status":200,"msg":"ok","Predicted_Label":prediction_class}

    return jsonify(dic)


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True,debug=False) 
