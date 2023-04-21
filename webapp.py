import argparse
import io
import os
from PIL import Image
import datetime

import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M"

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST": 
        if "file1" not in request.files:
            return redirect(request.url)
        file1 = request.files["file1"]
        if not file1:
            return
        
        if "file2" not in request.files:  
            return redirect(request.url)
        file2 = request.files["file2"]
        if not file2:
            return

        front_bytes = file1.read()
        back_bytes = file2.read()
        #print(file)         <FileStorage: 'test.jpg' ('image/jpeg')>

        f_img = Image.open(io.BytesIO(front_bytes))
        f_results = model([f_img])

        b_img = Image.open(io.BytesIO(back_bytes))
        b_results = model([b_img])


        '''
        print(f"front: {f_results}") #image 1/1: 613x960 60 apples
                                     #Speed: 23.9ms pre-process, 667.5ms inference, 1.3ms NMS per image at shape (1, 3, 416, 640)
        '''

        f_count =  str(f_results).split()[-17] #Count number of apples
        b_count =  str(b_results).split()[-17]

        all_count =  int((int(f_count) + int(b_count)) * 1.8)
        print(all_count)

        f_results.render()
        real_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        f_img_savename = f"result/front/{real_time}.png"
        Image.fromarray(f_results.ims[0]).save(f_img_savename)


        b_results.render() 
        b_img_savename = f"result/back/{real_time}.png"
        Image.fromarray(b_results.ims[0]).save(b_img_savename)

        return render_template('index.html', all_count = all_count)
    
    return render_template("index.html")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load("/home/hj/yolov5", 'custom', path='/home/hj/yolov5/runs/train/exp2/weights/best.pt', source='local')
    model.eval()
    app.run(host="0.0.0.0", port=args.port)
