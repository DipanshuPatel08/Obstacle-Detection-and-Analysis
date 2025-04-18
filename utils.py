import cv2
import numpy as np
import os
import google.generativeai as genai
import uuid
import logging
from PIL import Image
import base64
import io

def make_chunks(EdgeArray, size_of_chunk):
    """
    Creates chunks from edge array
    
    Args:
        EdgeArray: Array of edge points
        size_of_chunk: Size of each chunk
        
    Returns:
        List of chunks
    """
    chunks = []
    for i in range(0, len(EdgeArray), size_of_chunk):
        chunks.append(EdgeArray[i:i + size_of_chunk])
    return chunks

def process_image(image_path):
    """
    Process an image for obstacle detection
    
    Args:
        image_path: Path to the uploaded image
        
    Returns:
        processed_path: Path to the processed image
        direction: Navigation direction (preserved for backward compatibility)
        contours_info: Information about detected contours
    """
    # Read the image
    frame = cv2.imread(image_path)
    
    if frame is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Create copies for different visualizations
    original_frame = frame.copy()
    img_processed = frame.copy()
    
    # Process image for edge detection
    blur = cv2.bilateralFilter(img_processed, 9, 40, 40)
    edges = cv2.Canny(blur, 50, 100)
    
    # Draw contours
    blurred_frame = cv2.bilateralFilter(img_processed, 9, 75, 75)
    gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 106, 255, 1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create contours info and draw white highlights for obstacles
    contours_info = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            # Draw white rectangle around obstacle
            cv2.rectangle(img_processed, (x, y), (x+w, y+h), (255, 255, 255), 2)
            # Add object number
            cv2.putText(img_processed, f"Object {i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            contours_info.append({
                'id': i,
                'area': float(area),
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            })
    
    # Add title to the processed image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_processed, "Detected Obstacles", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Save processed image
    filename = f"processed_{uuid.uuid4().hex}.jpg"
    processed_path = os.path.join('static/uploads', filename)
    cv2.imwrite(processed_path, img_processed)
    
    # Return a placeholder for direction to maintain backward compatibility
    direction = "Obstacles Detected"
    
    return processed_path, direction, contours_info

def input_image_setup(image_path):
    """
    Setup image for Google Generative AI
    
    Args:
        image_path: Path to the image
        
    Returns:
        List containing image part dictionary
    """
    with open(image_path, "rb") as f:
        bytes_data = f.read()
    
    mime_type = "image/jpeg"
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"
    
    image_parts = [{"mime_type": mime_type, "data": bytes_data}]
    return image_parts

def generate_scene_description(image_path, input_text=""):
    """
    Generate scene description using Google Generative AI
    
    Args:
        image_path: Path to the image
        input_text: Input prompt
        
    Returns:
        Description text
    """
    try:
        if not input_text:
            input_text = "Describe what's in this image"
        
        image_parts = input_image_setup(image_path)
        
        prompt = """
        You are an expert in analyzing images for robotics applications and obstacle detection. 
        Provide a detailed description of the image, focusing on:
        
        1. Scene Overview: Briefly describe what's in the image (1-2 sentences)
        
        2. Obstacles Identified: List and describe each potential obstacle, including their:
           - Approximate position in the scene (top-left, center, etc.)
           - Estimated size/dimensions
           - Type (static object, potential moving object, terrain feature)
        
        3. Path Planning Challenges: Explain what difficulties a robot might face navigating this scene
        
        4. Navigation Recommendations: Suggest the safest route through the scene
        
        Format your response in clean paragraphs without using markdown symbols like * or **. 
        Use simple HTML for organization if needed, but keep formatting minimal and clean.
        Keep your analysis concise but informative for robotics applications.
        """
        
        # Check if API key is configured
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Google API key is not configured. Please set the GOOGLE_API_KEY environment variable."
        
        # Configure Gemini with API key
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input_text, image_parts[0], prompt])
        return response.text
    except Exception as e:
        logging.error(f"Error generating scene description: {e}")
        return f"Could not generate description: {str(e)}"
