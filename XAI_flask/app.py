from flask import Flask, render_template, send_file, make_response
from nocache import nocache
from models.shap import SHAP_MNIST

shap_mnist = SHAP_MNIST()

app = Flask(__name__)


# index 페이지 endpoint
@app.route('/')
def index():
    return render_template('index.html')


# XAI 소개 페이지 endpoint
@app.route('/XAI')
def XAI():
    return render_template('XAI.html')


# Vision에서 SHAP에 대한 endpoint
@app.route('/visionshap')
def visionshap():
    return render_template('visionshap.html')


# Vision에서 mnist 결과를 get으로 보내는 endpoint
@app.route('/visionshapmnist')
@nocache
def visionshapmnist():
    print('debug')
    return send_file(shap_mnist.plot(), mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
