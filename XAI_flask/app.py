from flask import Flask, render_template, send_file
from models.shap import SHAP_MNIST

shap_mnist = SHAP_MNIST()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/XAI')
def XAI():
    return render_template('index.html')


@app.route('/visionshap')
def visionshap():
    return send_file(shap_mnist.plot(), mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
