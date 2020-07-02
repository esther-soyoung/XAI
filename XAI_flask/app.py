from tensorflow.keras.models import load_model

from flask import Flask, render_template, send_file, request, __main__

from nocache import nocache


from models.lime import LIME_NLP
from models.shap_vision import SHAP_MNIST
from models.shap_nlp import SHAP_NLP
from models.lrp import LRP

'''
    각종 모델을 초기화하는 부분
'''
model_mnist = load_model('models/pretrained/mnist_model.h5')

shap_mnist = SHAP_MNIST(model_mnist)
# lrp_mnist = LRP(model_mnist, (-1, 28, 28, 1))

shap_nlp = SHAP_NLP()
lime_nlp = LIME_NLP()

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


'''
    이 아래로는 Vision의 결과를 요청하는 endpoint
'''
@app.route('/visionshap/<string:dataset>')
@nocache
def visionshap(dataset):
    if dataset == 'mnist':
        return send_file(shap_mnist.plot(), mimetype='image/png')
    elif dataset == 'cifar':
        None


# vision lrp
@app.route('/visionlrp/<string:dataset>')
@nocache
def visionlrp(dataset):
    return ''


# vision lime
@app.route('/visionlime/<string:dataset>')
@nocache
def visionlime(dataset):
    return ''


# vision filtervisualization
@app.route('/visionfv/<string:dataset>')
@nocache
def visionfv(dataset):
    return ''


'''
    이 아래로는 NLP의 결과를 요청하는 endpoint
'''
@app.route('/nlpshap')
@nocache
def nlpshap():
    requested_text = request.args['text']
    return send_file(shap_nlp.plot(requested_text), mimetype='image/png')


@app.route('/nlplrp')
@nocache
def nlplrp():
    requested_text = request.args['text']
    return 'hello i am nlp lrp'


@app.route('/nlplime')
@nocache
def nlplime():
    requested_text = request.args['text']
    return lime_nlp.plot(requested_text)


if __name__ == '__main__':
    from models.lime import TextsToSequences
    app.run(host='0.0.0.0', port=80)
