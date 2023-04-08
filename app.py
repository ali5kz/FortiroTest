
from flask import Flask, render_template, request,jsonify
from pdf2image import convert_from_path,convert_from_bytes
from src.pipeline.predict_pipeline import predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Document-Classifier', methods=['POST'])
def pdf_to_image():
    pdf_file  = request.files['file']
 
    images = convert_from_bytes(pdf_file.read())
    image = images[0]
    shape = image.height
    lab = predict(image)
    return lab
    # if shape%2==0:
    
    #     return "this is even : " + str(shape)
    # else:
    #     return "this is odd : " + str(shape)

if __name__ == '__main__':
    app.run(debug=True)
