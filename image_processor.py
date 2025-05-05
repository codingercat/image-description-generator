import os
import zipfile
import pandas as pd
import logging
from datetime import datetime
from PIL import Image
import requests
import json
import base64
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_zip(zip_path, extract_dir):
    """Extract contents of a zip file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Get list of image files
    image_files = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    return image_files

def generate_image_description(image_path, subject, audience):
    """
    Generate a description for an image using OpenAI's API.
    
    Args:
        image_path: Path to the image file
        subject: The subject area (e.g., Mathematics, Biology)
        audience: The target audience (e.g., Elementary school students)
        
    Returns:
        A description of the image contextual to the subject and audience
    """
    try:
        # Get basic image info
        with Image.open(image_path) as img:
            width, height = img.size
            format_name = img.format
            
        # Prepare a message for OpenAI API
        # First, encode the image to base64
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
                
        base64_image = encode_image(image_path)
        
        # Direct API call without using the client library
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "system", 
                    "content": f"You are a VI educator, expert at describing images for {audience} (blind students) studying {subject}. "
                               f"Provide clear, detailed, and educational descriptions that focus on aspects "
                               f"relevant to {subject}. Keep descriptions between 100-150 words. "
                               f"Be factual, educational, and appropriate for the audience level."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Please describe this image for {audience} (blind students) studying {subject}."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{format_name.lower()};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response_data = response.json()
        
        if 'error' in response_data:
            raise ValueError(f"API Error: {response_data['error']['message']}")
            
        # Extract and return the description
        description = response_data['choices'][0]['message']['content']
        logging.info(f"Generated description for {os.path.basename(image_path)}")
        return description
        
    except Exception as e:
        logging.error(f"Error generating description for {image_path}: {str(e)}")
        return f"Error generating description: {str(e)}"

def process_individual_images(image_paths, output_dir, subject, audience):
    """
    Process a list of individual image files.
    
    Args:
        image_paths: List of paths to image files
        output_dir: Directory to save results
        subject: Subject area for context
        audience: Target audience for descriptions
        
    Returns:
        Dictionary with processing results
    """
    results = []
    total_images = 0
    
    logging.info(f"Processing {len(image_paths)} individual images")
    
    for image_path in image_paths:
        try:
            filename = os.path.basename(image_path)
            
            # Get image dimensions
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format
            
            # Generate the description
            description = generate_image_description(image_path, subject, audience)
            
            # Add to results
            results.append({
                "Filename": filename,
                "Format": format_name,
                "Width": width,
                "Height": height,
                "Subject": subject,
                "Audience": audience,
                "Description": description,
                "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            total_images += 1
            
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            results.append({
                "Filename": os.path.basename(image_path),
                "Format": "Error",
                "Width": 0,
                "Height": 0,
                "Subject": subject,
                "Audience": audience,
                "Description": f"Error: {str(e)}",
                "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Save results to Excel
    excel_file = os.path.join(output_dir, "descriptions.xlsx")
    df = pd.DataFrame(results)
    df.to_excel(excel_file, index=False)
    
    return {
        "total_images": total_images,
        "excel_file": excel_file
    }

def process_zip_file(zip_path, output_dir, subject, audience):
    """
    Process a zip file containing images.
    
    Args:
        zip_path: Path to the zip file
        output_dir: Directory to save results
        subject: Subject area for context
        audience: Target audience for descriptions
        
    Returns:
        Dictionary with processing results
    """
    # Create a directory for extracted files
    extract_dir = os.path.join(os.path.dirname(zip_path), "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract the zip file
    logging.info(f"Extracting zip file: {zip_path}")
    image_paths = extract_zip(zip_path, extract_dir)
    
    # Process the extracted images
    logging.info(f"Found {len(image_paths)} images in zip file")
    return process_individual_images(image_paths, output_dir, subject, audience)