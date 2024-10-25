from PIL import Image
from lang_sam import LangSAM
from lang_sam.utils import draw_image
from lang_sam.utils import load_image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import json
import cv2

def complete_json(json_path, saved_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        frame_iter = data["frames"]
    num_quote  = 0
    for index, file_name in enumerate(frame_iter):
        file_num = file_name["file_path"].split('_')[1].split('.')[0]
        destination_path = f"{saved_path}/{file_num}_masks.jpg"
        file_name["mask_path"] = destination_path
    with open(json_path, 'w') as file:
        json.dump(data, file)


def filling_contour(files_paths):
    if not os.path.isdir(files_paths):
        raise ValueError("Provided path is not a directory.")
    file_list = sorted([f for f in os.listdir(files_paths) if os.path.isfile(os.path.join(files_paths, f))])
    print(f"file length {len(file_list)}")
    num_quote  = 0
    for index, file_name in enumerate(file_list):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(files_paths, file_name)
            num_quote += 1
            # Read the mask image (ensure it is a binary mask, 0 for background, 255 for foreground)
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blur, 10, 200)  # Edge detection
            edges = cv2.dilate(edges, None)  # 默认(3x3)
            edges = cv2.erode(edges, None)      
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            mask = 255 - np.zeros_like(img, np.uint8)
            max_contour = contours[0]  
            cv2.drawContours(mask, [max_contour], -1,(0,0,0), -1)

            # Display the original and filled mask
            cv2.imwrite(f"{files_paths}{file_name}", mask)
    print(num_quote)

def obtains_masks(text_prompt, files_paths, saved_path, transform_path):

    print(f"####################Cuda status: {torch.cuda.is_available()}###################################")
    processed_result={'masks':[], 'boxes':[]}
    if not os.path.isdir(files_paths):
        raise ValueError("Provided path is not a directory.")
    file_list = reversed(sorted([f for f in os.listdir(files_paths) if os.path.isfile(os.path.join(files_paths, f))]))
    model = LangSAM()
    num_quote  = 0
    for index, file_name in enumerate(file_list):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = os.path.join(files_paths, file_name)
            frame_num = file_name.split('_')[1].split('.')[0]
            destination_path = f"{saved_path}/{frame_num}_masks.jpg"
            image_pil = load_image(image_path)
            image_shape = image_pil.size
            masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
                #Append the processed results to the dictionary
            processed_result['masks'].append(masks)
            processed_result['boxes'].append(boxes)
            binary_array = (masks * 255).byte().numpy().reshape((image_shape[1],image_shape[0]))
            binary_array = 255 - binary_array
    # # Create an image using the binary array
            image = Image.fromarray(binary_array, mode='L')
    # Save the image to the specified file path
            image.save(destination_path)

    filling_contour(saved_path)
    complete_json(transform_path, saved_path)
            
    return processed_result

 

