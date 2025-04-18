import os
import logging
from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for
from werkzeug.utils import secure_filename
import google.generativeai as genai
from dotenv import load_dotenv
from utils import process_image, generate_scene_description

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Configure Google Generative AI (moved to generate_scene_description function)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-dev-secret")

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detector')
def detector():
    return render_template('detector.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['image']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image for obstacle detection
        try:
            processed_path, direction, contours_info = process_image(filepath)
            
            # Generate scene description using Gemini API
            prompt = request.form.get('prompt', '')
            description = generate_scene_description(filepath, prompt)
            
            return jsonify({
                'success': True,
                'original_image': filepath,
                'processed_image': processed_path,
                'direction': direction,
                'description': description,
                'contours_info': contours_info
            })
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return jsonify({
        'success': False,
        'error': 'Invalid file type. Please upload a JPG, JPEG or PNG file.'
    }), 400

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('detector')), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error. Please try again later.'
    }), 500

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
