from flask import Flask, render_template, request
import sys
import os
import re
import base64
from train import *

sys.path.append(os.path.abspath("./model"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")
    
@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())
    z=pre()
    return z
    
    
def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))
    
if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)
