from flask import Flask, request, jsonify
from pdf2image import convert_from_bytes
from src.pipeline.predict_pipeline import predict

app = Flask(__name__)

@app.route('/pdf-to-image', methods=['POST'])
def pdf_to_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    pdf_file = request.files['file']
    if not pdf_file.filename.endswith('.pdf'):
        return jsonify({'error': 'File is not a PDF'}), 400
    
    images = convert_from_bytes(pdf_file.read())
    image = images[0]
    results = predict(image)
    
    return jsonify({'results': results}), 200

if __name__ == '__main__':
    app.run(debug=True)