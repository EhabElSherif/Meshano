from flask import Flask, jsonify, request
from sys import exc_info
import os, shutil
import pmvs
import mesh
import camera

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

app = Flask(__name__)
datasetPath = "Data/"

@app.route("/sendimages",methods=["POST"])
def get_images():
    clear_folder(datasetPath+"images")
    clear_folder(datasetPath+"projections")
    try:
        uploaded_files = request.files.getlist("file[]")
        for idx,file in enumerate(uploaded_files):
            file.save(dst=datasetPath+"images/"+file.filename+".jpg")
            
        camera.main(datasetPath)
        pmvs.main(datasetPath)
        mesh.main("expansion_pointcloud.ply")
        print('DONE')
    except:
        print(exc_info())
        return jsonify({"message":"fail"})
    return jsonify({"message":"done"})
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)