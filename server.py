from flask import Flask, jsonify, request
import cv2 as cv
import json
from os import startfile
from sys import exc_info

app = Flask(__name__)

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
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)