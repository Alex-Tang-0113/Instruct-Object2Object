import os
import subprocess
import cv2
import numpy as np

IMAGE_PATH = "/workspace/nerfstudio/nerfdata/pbarry2/images_2/"
MASK_PATH = "/workspace/nerfstudio/nerfdata/pbarry2/masks_2/"

counter = 0
for img, mask in zip(sorted(os.listdir(IMAGE_PATH)), sorted(os.listdir(MASK_PATH))):
    img_path = IMAGE_PATH+img
    mask_path = MASK_PATH+mask
    mask = cv2.imread(mask_path)
    size = (mask.shape[1], mask.shape[0])
    mask = np.where(mask < 10, 0, 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=8)
    cv2.imwrite('erode_mask.jpg', mask)
    command = 'cargo run --release -- --out-size 480 --inpaint erode_mask.jpg -o sqr_inpainted.jpg generate {}'.format(img_path)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    inpainted = cv2.imwrite('out/inpainted_{}.jpg'.format(counter), cv2.resize(cv2.imread('sqr_inpainted.jpg'), size))
    print("inpainted", counter, "finished")
    counter+=1
    