from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/XAI')
def introduceXAI():
    return render_template('introduction.html')


# @app.route('SHAP')
# def SHAP():
#     return render_template('SHAP.html')


if __name__ == '__main__':
    app.run()
