# Image Description Generator API

This Flask-based API generates accessible descriptions for images to assist visually impaired students. It uses OpenAI's Vision API to analyze images and create contextual descriptions based on the educational subject and audience.

## Features

- Process individual images or ZIP files containing multiple images
- Generate descriptions tailored to specific subjects and educational levels
- Return results in Excel format for easy integration with educational materials
- RESTful API design for integration with web or mobile applications

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd image-description-api
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   
5. Edit the `.env` file and add your OpenAI API key.

## Running the API Locally

Start the Flask development server:

```bash
python app.py
```

The API will be available at `http://localhost:5000`.

## API Endpoints

### Generate Descriptions
- **URL:** `/api/generate-descriptions`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file`: One or more image files, or a ZIP file containing images (required)
  - `subject`: The educational subject context (required)
  - `audience`: The target audience (required)
- **Response:** JSON with job ID and status information

### Check Job Status
- **URL:** `/api/jobs/<job_id>`
- **Method:** `GET`
- **Response:** JSON with job status information

### Download Results
- **URL:** `/api/download/<job_id>`
- **Method:** `GET`
- **Response:** Excel file with image descriptions

## Deployment

### Deploy to Render

1. Create a new Web Service on Render
2. Link your repository
3. Set the environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
4. Use the following build settings:
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`

### Deploy to Heroku

1. Install the Heroku CLI and login:
   ```bash
   heroku login
   ```

2. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```

3. Set the OpenAI API key:
   ```bash
   heroku config:set OPENAI_API_KEY=your_api_key_here
   ```

4. Push to Heroku:
   ```bash
   git push heroku main
   ```

## Integration with React Frontend

The API is designed to work with the ImageDescriptionGenerator React component. The frontend component handles:

1. File selection and upload
2. Subject and audience inputs
3. Job status monitoring
4. Excel file download

The API's CORS support allows seamless integration with frontend applications served from different domains.

## License

[MIT License](LICENSE)