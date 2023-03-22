import os
import cv2
import sys
import torch
import numpy as np
import torch.nn.functional as F

from os import path
from src import utils
from model import UNet


INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
MODEL_PATH = sys.argv[3]    
THRESHOLD = 500

def run_single_frame(model, file, threshold=THRESHOLD):
    """ Runs semantic segmentation on a single image frame """
    
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    orig_img = image.copy()
    image = utils.preprocess_image(image)
    if torch.cuda.is_available():
        image = image.cuda()
    
    with torch.no_grad():
        out = model(image)
        out = F.logsigmoid(out).exp()
        out = out.detach().cpu().squeeze(0).numpy()
        out = np.where(out > 0.6, 1, 0).astype(np.uint8)
        
        box = utils.locate_plate(out, threshold=threshold)
        assert len(box) > 0
    
    plate = utils.get_warped_plate(orig_img, box)
    
    name = path.splitext(path.basename(file))[0]
    
    plate_path = path.join(OUTPUT_DIR, f'{name}_plate.jpg')
    box_path = path.join(OUTPUT_DIR, f'{name}_box.txt')
    
    cv2.imwrite(str(plate_path), plate)
    with open(str(box_path), 'w') as box_file:
        box_file.write(','.join([f'{value}' for value in box]))
    

def run_dir(model, img_dir):
    """ Runs segmentation on a folder of images """
    print(f'Loading weights from {MODEL_PATH}...Done!')
    print('\n\t***** PERFORMING SEGMENTATION *****\n')
    
    for file in os.scandir(img_dir):
        
        try:
            if file.is_file():
                print(f'\tScanning {file.path}')
                run_single_frame(model, file.path)
        except AssertionError:
            print(f'\t`ERROR: {file.path}`')
            continue

if __name__ == "__main__":
    
    model_state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model = UNet()
    model.load_state_dict(model_state["model"])
    model.eval()
    
    run_dir(model, INPUT_DIR)

