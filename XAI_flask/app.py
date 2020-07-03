import tensorflow as tf

from flask import Flask, render_template, send_file, request, __main__, make_response

from nocache import nocache

from models.shap_vision import SHAP_MNIST
from models.shap_nlp import SHAP_NLP
from models.FilterViz import FilterViz
from models.PDP import PDP_BOSTON


import numpy as np
import json
import base64
from subprocess import check_output

'''
    각종 모델을 초기화하는 부분
'''

shap_mnist = SHAP_MNIST()
filter_viz = FilterViz()

shap_nlp = SHAP_NLP()

pdp_ml = PDP_BOSTON()

app = Flask(__name__)


# index 페이지 endpoint
@app.route('/')
def index():
    return render_template('index.html')


# Vision에서 SHAP에 대한 endpoint
@app.route('/vision')
def vision():
    return render_template('vision.html')


@app.route('/nlp')
def nlp():
    return render_template('nlp.html')


@app.route('/machinelearning')
def machinelearning():
    return render_template('machinelearning.html')


@app.route('/introduction')
def introduction():
    return render_template('introduction.html')


'''
    이 아래로는 Vision의 결과를 요청하는 endpoint
'''


@app.route('/visionshap', methods=['POST'])
@nocache
def visionshap():
    img = np.array(json.loads(request.form['image']))
    return base64.b64encode(shap_mnist.plot(img).getvalue())


# vision lrp
@app.route('/visionlrp', methods=['POST'])
@nocache
def visionlrp():
    img = request.form['image']
    try:
        result = check_output(['python', 'models/lrp.py'], input=img.encode())
    except:
        try:
            result = check_output(['python3', 'models/lrp.py'], input=img.encode())
        except:
            raise FileNotFoundError("no python interpreter found")
    return result



# vision lime
@app.route('/visionlime', methods=['POST'])
@nocache
def visionlime():
    img = request.form['image']
    try:
        result = check_output(['python', 'models/lime_vision.py'], input=img.encode())
    except:
        try:
            result = check_output(['python3', 'models/lime_vision.py'], input=img.encode())
        except:
            raise FileNotFoundError("no python interpreter found")
    print(result)
    result = result.decode().split('=')[0] + '='
    return result


# vision filtervisualization
@app.route('/visionfv/<int:layer>', methods=['POST'])
@nocache
def visionfv(layer):
    img = np.array(json.loads(request.form['image']))
    return base64.b64encode(filter_viz.get_FilterViz(img, layer).getvalue())


'''
    이 아래로는 NLP의 결과를 요청하는 endpoint
'''


@app.route('/nlpshap')
@nocache
def nlpshap():
    requested_text = request.args['text']
    return send_file(shap_nlp.plot(requested_text), mimetype='image/png')


@app.route('/nlplime')
@nocache
def nlplime():
    requested_text = request.args['text']
    try:
        result = check_output(['python', 'models/lime_nlp.py'], input=requested_text.encode())
    except:
        try:
            result = check_output(['python3', 'models/lime_nlp.py'], input=requested_text.encode())
        except:
            raise FileNotFoundError("no python interpreter found")
    return result


@app.route('/pdp/<int:i>')
@nocache
def pdp(i):
    print(i)
    return send_file(pdp_ml.plot(i), mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
