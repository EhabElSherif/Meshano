from flask import Flask, jsonify, request
import cv2 as cv
import json
from os import startfile
from sys import exc_info

app = Flask(__name__)

@app.route("/sendvideo", methods=["POST"])
def get_video():
    try:
        data = request.files['video']
        data.save(dst="out.mp4")
        print('DONE')
    except:
        return jsonify({"message":"fail"})
    return jsonify({"message":"done"})

@app.route("/sendimages",methods=["POST"])
def get_images():
    try:
        uploaded_files = request.files.getlist("file[]")
        for idx,file in enumerate(uploaded_files):
            file.save(dst=file.filename+".jpg")
        print('DONE')
    except:
        print(exc_info())
        return jsonify({"message":"fail"})
    return jsonify({"message":"done"})


@app.route("/create", methods=["POST"])
def home():
    return jsonify({'hi':"data"})

# @app.route("/", methods=["GET"])
# def cxc():
#     print("arrive")
#     return "Sent"

@app.route("/multi/<int:num>" , methods=["GET"])
def multi10(num):
    return jsonify({'result':num*10})

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)