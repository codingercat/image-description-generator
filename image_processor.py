import os
import zipfile
import pandas as pd
import logging
import gc
import traceback
from datetime import datetime
from PIL import Image
import requests
import json
import base64
import time
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

def generate_image_description(image_path, subject, audience, max_retries=3):
    """
    Generate a description for an image using OpenAI's API.
    
    Args:
        image_path: Path to the image file
        subject: The subject area (e.g., Mathematics, Biology)
        audience: The target audience (e.g., Elementary school students)
        max_retries: Maximum number of retries on API failure
        
    Returns:
        A description of the image contextual to the subject and audience
    """
    retry_count = 0
    backoff_time = 2  # Initial backoff time in seconds
    temp_path = None
    format_name = "unknown"
    width = 0
    height = 0
    
    while retry_count < max_retries:
        try:
            # Get basic image info and prepare image - use context manager for auto-closing
            image_path_to_use = image_path
            
            try:
                # Use context manager to ensure image is properly closed
                with Image.open(image_path) as img:
                    width, height = img.size
                    format_name = img.format or "unknown"
                    
                    # If image is too large, resize it to reduce API payload and memory usage
                    if width * height > 4000000:  # 4 million pixels
                        factor = (width * height / 4000000) ** 0.5
                        new_width = int(width / factor)
                        new_height = int(height / factor)
                        
                        # Create a new image inside this context - will be auto-closed
                        resized_img = img.resize((new_width, new_height))
                        
                        # Save to a temporary file
                        temp_path = f"{os.path.splitext(image_path)[0]}_resized.jpg"
                        resized_img.save(temp_path, format="JPEG", quality=80)
                        
                        # Update path to use resized image
                        image_path_to_use = temp_path
                        format_name = "jpeg"
            except Exception as img_error:
                logging.error(f"Error processing image {image_path}: {str(img_error)}")
                raise ValueError(f"Image processing error: {str(img_error)}")
            
            # Encode the image to base64 with memory-efficient approach
            base64_image = None
            try:
                with open(image_path_to_use, "rb") as image_file:
                    image_data = image_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    # Explicitly delete large variables to free memory
                    del image_data
            except Exception as encode_error:
                logging.error(f"Error encoding image {image_path_to_use}: {str(encode_error)}")
                raise ValueError(f"Image encoding error: {str(encode_error)}")
            
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment variables")
                
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            # Prepare payload for GPT-4 Vision API
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "system", 
                        # "content": f"You are a VI educator, expert at describing images for {audience} (blind students) studying {subject}. "
                        #           f"Provide clear, detailed, and educational descriptions that focus on aspects "
                        #           f"relevant to {subject}. Avoid adding irrelevant information like colours, "
                        #           f"Be factual, educational, and appropriate for the audience level."
                        "content": f"Generate image description for a blind person, keep it contextual, considering the age and grade of the book."
                        f"Go from general to specific, be concise and objective and keep tone and language of the description relevant to the grade of the book."
                        f"Don't include colours and irrelevant information and make it straightforward."
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
            
            # Explicit cleanup of large variables
            del base64_image
            gc.collect()
            
            # Make the API request with proper error handling and timeout
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=90  # 90 second timeout - increased for safety
                )
                
                # Check if the response status code indicates success
                response.raise_for_status()
                
                # Parse the response JSON
                response_data = response.json()
                
                if 'error' in response_data:
                    raise ValueError(f"API Error: {response_data['error']['message']}")
                    
                # Extract the description
                description = response_data['choices'][0]['message']['content']
                logging.info(f"Generated description for {os.path.basename(image_path)}")
                
                # Clean up temporary file if we created one
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as rm_error:
                        logging.warning(f"Failed to remove temp file {temp_path}: {str(rm_error)}")
                
                # Clean up response data
                del response_data
                del response
                gc.collect()
                    
                return description
                
            except requests.exceptions.HTTPError as http_err:
                raise ValueError(f"HTTP Error: {http_err}")
            except requests.exceptions.ConnectionError:
                raise ValueError("Connection Error: Could not connect to OpenAI API")
            except requests.exceptions.Timeout:
                logging.error(f"API request timed out for {os.path.basename(image_path)}")
                raise ValueError("Timeout Error: The request to OpenAI API timed out")
            except requests.exceptions.RequestException as req_err:
                raise ValueError(f"Request Error: {req_err}")
            
        except Exception as e:
            retry_count += 1
            logging.warning(f"Attempt {retry_count}/{max_retries} failed for {image_path}: {str(e)}")
            
            if retry_count < max_retries:
                logging.info(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff
            else:
                logging.error(f"Failed to generate description after {max_retries} attempts: {str(e)}")
                
                # Clean up temporary file if we created one
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                    
                return f"Error generating description after {max_retries} attempts: {str(e)}"
        
        finally:
            # Force garbage collection
            gc.collect()

def process_individual_images(image_paths, output_dir, subject, audience, batch_size=1):
    """
    Process a list of individual image files with improved memory management.
    
    Args:
        image_paths: List of paths to image files
        output_dir: Directory to save results
        subject: Subject area for context
        audience: Target audience for descriptions
        batch_size: Number of images to process in a batch before saving progress
        
    Returns:
        Dictionary with processing results
    """
    results = []
    already_processed = set()
    total_images = 0
    
    logging.info(f"Processing {len(image_paths)} individual images")
    
    # Save progress periodically to avoid losing work on timeout
    excel_file = os.path.join(output_dir, "descriptions.xlsx")
    
    # Check if progress file exists and load it
    if os.path.exists(excel_file):
        try:
            existing_df = pd.read_excel(excel_file)
            results = existing_df.to_dict('records')
            
            # Track already processed files
            for row in results:
                already_processed.add(row.get("Filename", ""))
                
            logging.info(f"Loaded {len(results)} existing results from {excel_file}")
        except Exception as e:
            logging.error(f"Error loading existing progress file: {str(e)}")
    
    for i, image_path in enumerate(image_paths):
        # Handle potential encoding issues with filenames
        try:
            filename = os.path.basename(image_path)
            
            # Skip already processed images
            if filename in already_processed:
                logging.info(f"Skipping already processed image: {filename}")
                continue
                
            # Get image dimensions with proper cleanup
            width, height, format_name = 0, 0, "Unknown"
            
            try:
                # Use context manager for auto-closing
                with Image.open(image_path) as img:
                    width, height = img.size
                    format_name = img.format or "Unknown"
            except Exception as img_error:
                logging.error(f"Error reading image {image_path}: {str(img_error)}")
                # Continue processing with default values
                
            try:
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
                already_processed.add(filename)
                
            except Exception as desc_error:
                logging.error(f"Error generating description for {filename}: {str(desc_error)}")
                
                # Add error entry to results
                results.append({
                    "Filename": filename,
                    "Format": format_name,
                    "Width": width, 
                    "Height": height,
                    "Subject": subject,
                    "Audience": audience,
                    "Description": f"Error generating description: {str(desc_error)}",
                    "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                already_processed.add(filename)
                
            # Save progress after each batch or after processing an image that had errors
            if (i + 1) % batch_size == 0 or i == len(image_paths) - 1:
                save_progress_to_excel(results, excel_file)
                logging.info(f"Progress saved: {len(already_processed)}/{len(image_paths)} images processed")
            
            # Force garbage collection after each image
            gc.collect()
            
        except Exception as e:
            logging.error(f"Critical error processing image path {i}: {str(e)}")
            logging.error(traceback.format_exc())
            
            # Try to get filename even in error cases
            try:
                filename = os.path.basename(image_path)
            except:
                filename = f"unknown_file_{i}"
                
            # Add error to results
            results.append({
                "Filename": filename,
                "Format": "Error",
                "Width": 0,
                "Height": 0,
                "Subject": subject,
                "Audience": audience,
                "Description": f"Critical error: {str(e)}",
                "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Save progress after critical error
            save_progress_to_excel(results, excel_file)
            
            # Force garbage collection after error
            gc.collect()
    
    # Final save
    save_progress_to_excel(results, excel_file)
    
    return {
        "total_images": total_images,
        "excel_file": excel_file
    }

def save_progress_to_excel(results, excel_file):
    """Helper function to safely save progress to Excel file"""
    for attempt in range(3):  # Try up to 3 times
        try:
            pd.DataFrame(results).to_excel(excel_file, index=False)
            return  # Success, exit function
        except Exception as save_error:
            logging.error(f"Save attempt {attempt+1} failed: {str(save_error)}")
            time.sleep(1)  # Wait a bit before retrying
            
            if attempt == 1:  # On second attempt, try alternative method
                try:
                    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                        pd.DataFrame(results).to_excel(writer, index=False)
                    return  # Success with alternative method
                except Exception as alt_error:
                    logging.error(f"Alternative save method failed: {str(alt_error)}")
    
    logging.error("All attempts to save progress failed")

def process_zip_file(zip_path, output_dir, subject, audience):
    """
    Process a zip file containing images with improved memory management.
    
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
    
    # Process the extracted images with smaller batch size for memory efficiency
    logging.info(f"Found {len(image_paths)} images in zip file")
    
    # Use a smaller batch size (1) to avoid memory issues
    return process_individual_images(image_paths, output_dir, subject, audience, batch_size=1)