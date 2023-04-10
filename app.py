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
 
    # Check if the uploaded file is a PDF
    if not pdf_file.filename.endswith('.pdf'):
        return render_template('index.html',results='Error: File is not a PDF')
    images = convert_from_bytes(pdf_file.read())
    image = images[0]
    results=predict(image)
    print("after Prediction")
    return render_template('index.html',results=results)
if __name__ == '__main__':
    #app.run(host="0.0.0.0")
    app.run(debug=True)


# import requests

# api_url = 'http://localhost:5000//Document-Classifier'

# with open('sample.pdf', 'rb') as f:
#     response = requests.post(api_url, files={'file': f})

# if response.status_code == 200:
#     results = response.json()['results']
#     print('Prediction results:', results)
# else:
#     print('Error:', response.json()['error'])