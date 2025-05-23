import os
import logging
import tempfile
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
# Import our image processing module
from image_processor import process_zip_file, process_individual_images, extract_zip, generate_image_description
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure CORS to allow requests from any origin
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Ensure CORS headers are properly set on all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
    return response

# Configure upload settings
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'image_descriptions')
OUTPUT_FOLDER = os.path.join(tempfile.gettempdir(), 'image_descriptions_output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB limit

# Allowed file extensions
ALLOWED_EXTENSIONS = {'zip', 'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/generate-descriptions', methods=['POST', 'OPTIONS'])
def generate_descriptions():
    """
    API endpoint to generate descriptions for images.
    Accepts a zip file or individual image files along with subject and audience parameters.
    """
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
        return response
        
    # Check if OPENAI_API_KEY is set
    if not os.getenv('OPENAI_API_KEY'):
        return jsonify({
            'error': 'OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable.'
        }), 500

    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    files = request.files.getlist('file')
    
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    # Get subject and audience parameters
    subject = request.form.get('subject', 'General Subject')
    audience = request.form.get('audience', 'Students')
    
    # Generate unique job ID and create directories
    job_id = str(uuid.uuid4())
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    output_dir = os.path.join(OUTPUT_FOLDER, job_id)
    
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files and process based on file type
    try:
        if len(files) == 1 and files[0].filename.lower().endswith('.zip'):
            # Handle ZIP file
            file = files[0]
            if not allowed_file(file.filename):
                return jsonify({'error': 'File type not allowed'}), 400
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_dir, filename)
            file.save(file_path)
            
            # Process the ZIP file
            result = process_zip_file(file_path, output_dir, subject, audience)
            
        else:
            # Handle individual image files
            image_paths = []
            for file in files:
                if not allowed_file(file.filename):
                    continue  # Skip files with invalid extensions
                
                filename = secure_filename(file.filename)
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                image_paths.append(file_path)
            
            if not image_paths:
                return jsonify({'error': 'No valid image files found'}), 400
            
            # Process the individual image files
            result = process_individual_images(image_paths, output_dir, subject, audience)
        
        # Return successful response with excel file path for download
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'Successfully processed {result["total_images"]} images',
            'total_images': result['total_images'],
            'excel_file': os.path.basename(result['excel_file'])
        })
        
    except Exception as e:
        logging.error(f"Error processing files: {e}")
        return jsonify({
            'error': f'Error processing files: {str(e)}'
        }), 500

@app.route('/download/<job_id>', methods=['GET'])
def download_results(job_id):
    """Download the Excel file with generated descriptions."""
    try:
        # Validate job_id format (basic UUID validation)
        try:
            uuid_obj = uuid.UUID(job_id)
        except ValueError:
            return jsonify({'error': 'Invalid job ID format'}), 400
        
        # Construct the path to the Excel file
        excel_file_path = os.path.join(OUTPUT_FOLDER, job_id, "descriptions.xlsx")
        
        if not os.path.exists(excel_file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Send the file as an attachment
        return send_file(
            excel_file_path,
            as_attachment=True,
            download_name="image_descriptions.xlsx",
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        logging.error(f"Error downloading file: {e}")
        return jsonify({
            'error': f'Error downloading file: {str(e)}'
        }), 500

@app.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Check status of a processing job."""
    try:
        # Validate job_id
        try:
            uuid_obj = uuid.UUID(job_id)
        except ValueError:
            return jsonify({'error': 'Invalid job ID format'}), 400
        
        # Check if output directory exists
        output_dir = os.path.join(OUTPUT_FOLDER, job_id)
        if not os.path.exists(output_dir):
            return jsonify({'status': 'not_found'}), 404
        
        # Check for Excel file
        excel_file_path = os.path.join(output_dir, "descriptions.xlsx")
        if os.path.exists(excel_file_path):
            return jsonify({
                'status': 'completed',
                'excel_file': 'descriptions.xlsx'
            })
        
        # If directory exists but no Excel file, it's still processing
        return jsonify({'status': 'processing'})
        
    except Exception as e:
        logging.error(f"Error checking job status: {e}")
        return jsonify({
            'error': f'Error checking job status: {str(e)}'
        }), 500

@app.route('/upload', methods=['POST'])
def upload():
    zip_file = request.files.get('zipfile')
    subject = request.form.get('subject', 'Mathematics')
    audience = request.form.get('audience', 'school students')

    if not zip_file:
        return "No file uploaded", 400

    zip_path = os.path.join(UPLOAD_FOLDER, zip_file.filename)
    zip_file.save(zip_path)

    image_paths = extract_zip(zip_path, UPLOAD_FOLDER)
    results = {}

    for img_path in image_paths:
        try:
            description = generate_image_description(img_path, subject, audience)
            results[os.path.basename(img_path)] = description
        except Exception as e:
            results[os.path.basename(img_path)] = f"Error: {str(e)}"

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
