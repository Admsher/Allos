from flask import Flask, request, render_template,jsonify

import json
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',text="fire")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        import ML
        file.save('../Allos/let.jpeg')
        # cv2.imshow('picture',file)
        box,label,score=ML.processing(image=file)
        results = {
        "bounding_boxes": box,
        "class_labels": label,
        "confidence_scores": score
    }
        
        json_list=json.dumps(results)
        return jsonify(json_list)


    return "Error uploading file"

if __name__ == '__main__':
    app.run(debug=True ,port=8000, host='0,0,0,0')
