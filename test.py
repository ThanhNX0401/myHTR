import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import random

# Paths for your images
sentence_images_folder = '/kaggle/input/d/phucthaiv02/vnondb/InkData_word/'  # Folder containing sentence images
background_images_path = '/kaggle/working/resized_images/*.jpg'  # Assuming resized images are in this folder
output_folder = '/kaggle/working/processed_images'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Load all background images
background_images = glob.glob(background_images_path)

def boldWords(sentences_image):
    if sentence_image.shape[2] == 4:  # If RGBA
        # Use the RGB channels to create a mask
        gray = cv2.cvtColor(sentence_image[:, :, :3], cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(sentence_image, cv2.COLOR_BGR2GRAY)

    # Threshold to create a binary mask for black text
    _, binary_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # 50 is a threshold to distinguish black

    # Dilation to bold the text
    kernel = np.ones((7, 7), np.uint8)  # Adjust kernel size as needed
    bold_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # Create a new image to hold the bold text with transparency
    bolded_sentence_image = np.zeros_like(sentence_image)  # Fully transparent background
    bolded_sentence_image[..., 3] = 0  # Set alpha channel to fully transparent

    # Set the black parts to black in the bolded image
    bolded_sentence_image[bold_mask == 255] = [0, 0, 0, 255] 
    return bolded_sentence_image

def get_random_position(box_width, box_height, placed_boxes, background_size=1024, center_area=768):
    margin = (background_size - center_area) // 2
    max_attempts = 100
    
    for _ in range(max_attempts):
        # Generate random position within the center 768x768 area
        x = random.randint(margin, margin + center_area - box_width)
        y = random.randint(margin, margin + center_area - box_height)
        
        # Check for overlap with existing boxes
        new_box = (x, y, x + box_width, y + box_height)
        if not any(boxes_overlap(new_box, existing_box) for existing_box in placed_boxes):
            return x, y
    
    return None, None  # Return None if no valid position found

def boxes_overlap(box1, box2):
    # Check if two boxes overlap
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return not (x2_1 < x1_2 or x1_1 > x2_2 or y2_1 < y1_2 or y1_1 > y2_2)

# Process each background image
for background_image_path in background_images:
    background_image = cv2.imread(background_image_path)
    
    # Get all sentence images and randomly select 10
    all_sentence_paths = glob.glob(os.path.join(sentence_images_folder, '*.png'))
    selected_sentences = random.sample(all_sentence_paths, min(10, len(all_sentence_paths)))
    
    placed_boxes = []  # Keep track of placed sentences
    sentence_coordinates = []  # Store coordinates of placed sentences
    
    # Process each selected sentence
    for img_path in selected_sentences:
        sentence_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        bold_sentence_image = boldWords(sentence_image)
        
        # Calculate appropriate scaling for the sentence
        scale_factor = min(200 / sentence_image.shape[1], 100 / sentence_image.shape[0])  # Max size 200x100
        new_width = int(sentence_image.shape[1] * scale_factor)
        new_height = int(sentence_image.shape[0] * scale_factor)
        resized_sentence = cv2.resize(bold_sentence_image, (new_width, new_height))
        
        # Find random position that doesn't overlap
        x_offset, y_offset = get_random_position(new_width, new_height, placed_boxes)
        
        if x_offset is not None:
            # Add the box to placed_boxes
            placed_boxes.append((x_offset, y_offset, x_offset + new_width, y_offset + new_height))
            sentence_coordinates.append({
                'sentence': os.path.basename(img_path),
                'position': (x_offset, y_offset),
                'size': (new_width, new_height)
            })
            
            # Add the sentence to the background using alpha blending
            for c in range(0, 3):
                background_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width, c] = (
                    background_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width, c] *
                    (1 - resized_sentence[..., 3] / 255.0) +
                    resized_sentence[..., c] * (resized_sentence[..., 3] / 255.0)
                )
    
    # Save the resulting image
    output_path = os.path.join(output_folder, f"combined_{os.path.basename(background_image_path)}")
    cv2.imwrite(output_path, background_image)
    
    # Save coordinates to a text file
    coord_file = output_path.rsplit('.', 1)[0] + '_coordinates.txt'
    with open(coord_file, 'w') as f:
        for coord in sentence_coordinates:
            f.write(f"Sentence: {coord['sentence']}\n")
            f.write(f"Position: {coord['position']}\n")
            f.write(f"Size: {coord['size']}\n\n")
    
    # Display result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB))
    plt.title("Combined Result")
    plt.axis("off")
    plt.show()
    
    print(f"Processed and saved: {output_path}")
    print(f"Coordinates saved: {coord_file}")
