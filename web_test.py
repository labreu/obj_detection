from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import os
import glob
import logging
logFormatStr = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
logging.basicConfig(format = logFormatStr, filename = "global.log", level=logging.DEBUG)
formatter = logging.Formatter(logFormatStr,'%m-%d %H:%M:%S')
fileHandler = logging.FileHandler("logs_web.log")
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(formatter)

os.makedirs('static', exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_PATH'] = 1024*5
app.logger.addHandler(fileHandler)
app.logger.info("Logging is set up.")

@app.route('/')
def index():
    files = [f for f in glob.glob('static/*') if '_test.' not in f]
    files_pred = [f for f in glob.glob('static/*') if '_test.' in f]
    return render_template('index.html', files=files, files_pred=files_pred)

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join('static', secure_filename(f.filename)))
      os.system('python predict_boxes_web.py --pbfile model800600export/frozen_inference_graph.pb &')

      return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict():
    fname = request.form['fname']

    return request.form['fname']

@app.route('/clear')
def clear():
    fs = glob.glob('static/*')
    for f in fs:
        os.remove(f)
    return redirect(url_for('index'))

if __name__=='__main__':
    app.run(port=5000, debug=False, host='0.0.0.0')

