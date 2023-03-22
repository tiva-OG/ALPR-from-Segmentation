import os
import cv2
import sys
import matplotlib.pyplot as plt

from os import path
from data.loader import load_files
from src.label import read_box, read_label
from src.drawing import draw_box, write_label

RED = (0, 0, 255)
YELLOW = (0, 255, 255)

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
FINAL_DIR = path.join(OUTPUT_DIR, 'final_output')
os.makedirs(FINAL_DIR, exist_ok=True)

if __name__ == '__main__':
    print('\n\t***** GENERATING OUTPUTS *****\n')
    
    img_files = load_files(INPUT_DIR)
    
    for file in img_files:
        name = path.splitext(file)[0]
        
        img_path = path.join(INPUT_DIR, file)
        print(f'\tScanning {img_path}...')
        img = cv2.imread(img_path)
        box_path = path.join(OUTPUT_DIR, f'{name}_box.txt')
        box = read_box(box_path) # read box points and return an array of points
        
        if len(box) > 0:
            draw_box(img, box, color=YELLOW) # draw box on image indicating license-plate position
            
            label_path = path.join(OUTPUT_DIR, f'{name}_label.txt')
            label = read_label(label_path) # read (and process) predicted label by OCR and return a string
            # print(f'\tLabel: {label}; Type: {type(label)}')
            write_label(img, label, box, color=RED) # write label on image above license-plate position
            
            final_path = path.join(FINAL_DIR, f'{name}.jpg')
            cv2.imwrite(final_path, img)
        
            # plt.imshow(img)
            # plt.show()