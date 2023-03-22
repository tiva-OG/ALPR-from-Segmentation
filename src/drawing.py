import cv2

FONT_TYPE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
THICKNESS = 2

def draw_box(img, box, color=(255, 0, 0), thickness=2):
    cv2.drawContours(img, [box], contourIdx=0, color=color, thickness=thickness)

def write_label(img, label, box, color=(0, 0, 0), font_type=FONT_TYPE, font_scale=FONT_SCALE, thickness=THICKNESS):
    offset = 5
    x = min([pt[0] for pt in box])
    y = min([pt[1] for pt in box]) - offset
    
    cv2.putText(img, label, (x, y), fontFace=font_type, fontScale=font_scale, color=color, thickness=thickness)