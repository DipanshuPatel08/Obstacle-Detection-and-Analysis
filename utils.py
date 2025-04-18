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
        direction: Navigation direction
        contours_info: Information about detected contours
    """
    # Read the image
    frame = cv2.imread(image_path)
    
    if frame is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Create copies for different visualizations
    original_frame = frame.copy()
    img_edgerep = frame.copy()
    img_contour = frame.copy()
    img_navigation = frame.copy()
    
    # Process image for edge detection
    blur = cv2.bilateralFilter(img_edgerep, 9, 40, 40)
    edges = cv2.Canny(blur, 50, 100)
    
    img_edgerep_h = img_edgerep.shape[0] - 1
    img_edgerep_w = img_edgerep.shape[1] - 1
    
    EdgeArray = []
    StepSize = 5
    
    # Edge detection logic
    for j in range(0, img_edgerep_w, StepSize):
        pixel = (j, 0)
        for i in range(img_edgerep_h - 5, 0, -1):
            if edges.item(i, j) == 255:
                pixel = (j, i)
                break
        EdgeArray.append(pixel)
    
    # Draw edges
    for x in range(len(EdgeArray) - 1):
        cv2.line(img_edgerep, EdgeArray[x], EdgeArray[x + 1], (0, 255, 0), 1)
    
    for x in range(len(EdgeArray)):
        cv2.line(img_edgerep, (x * StepSize, img_edgerep_h), EdgeArray[x], (0, 255, 0), 1)
    
    # Draw contours
    blurred_frame = cv2.bilateralFilter(img_contour, 9, 75, 75)
    gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 106, 255, 1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_edgerep, contours, -1, (0, 0, 255), 3)
    
    # Navigation direction
    number_of_chunks = 3
    size_of_chunk = int(len(EdgeArray) / number_of_chunks)
    chunks = make_chunks(EdgeArray, size_of_chunk)
    avg_of_chunk = []
    
    for i in range(len(chunks) - 1):
        x_vals = []
        y_vals = []
        for (x, y) in chunks[i]:
            x_vals.append(x)
            y_vals.append(y)
        avg_x = int(np.average(x_vals))
        avg_y = int(np.average(y_vals))
        avg_of_chunk.append([avg_y, avg_x])
        cv2.line(frame, (int(img_edgerep_w / 2), img_edgerep_h), (avg_x, avg_y), (255, 0, 0), 2)
    
    if len(avg_of_chunk) > 0:
        forwardEdge = avg_of_chunk[min(1, len(avg_of_chunk)-1)]
        cv2.line(frame, (int(img_edgerep_w / 2), img_edgerep_h), (forwardEdge[1], forwardEdge[0]), (0, 255, 0), 3)
        farthest_point = min(avg_of_chunk)
        
        # Determine direction
        if forwardEdge[0] > 250:
            if farthest_point[1] < 310:
                direction = "Move left"
            else:
                direction = "Move right"
        else:
            direction = "Move forward"
    else:
        direction = "Cannot determine direction"
    
    # Add direction text to image
    font = cv2.FONT_HERSHEY_SIMPLEX
    navigation = cv2.putText(frame, direction, (int(img_edgerep_w/2)-100, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Save processed image
    filename = f"processed_{uuid.uuid4().hex}.jpg"
    processed_path = os.path.join('static/uploads', filename)
    cv2.imwrite(processed_path, navigation)
    
    # Create contours info
    contours_info = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            contours_info.append({
                'id': i,
                'area': float(area),
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            })
    
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
        1. Identifying potential obstacles and their positions
        2. Describing the scene environment
        3. Explaining what path planning challenges might exist
        4. Suggesting possible navigation strategies
        Keep your analysis concise but informative for robotics applications.
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input_text, image_parts[0], prompt])
        return response.text
    except Exception as e:
        logging.error(f"Error generating scene description: {e}")
        return f"Could not generate description: {str(e)}"
