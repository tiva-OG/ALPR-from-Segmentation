import cv2
import numpy as np

from scipy.spatial import distance
from torchvision import transforms


def preprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    
    transformation = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
        ])
    
    image = transformation(image)
    
    return image.unsqueeze(0)

def postprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    
    inv_transformation = transforms.Compose([
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/std[0], 1/std[1], 1/std[2]]), 
        transforms.Normalize(mean=[-1*mean[0], -1*mean[1], -1*mean[2]], std=[1.0, 1.0, 1.0])
        ])
    
    inv_image = inv_transformation(image)
    
    return ((inv_image*255).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy())

def locate_plate(mask, size_factor=1, threshold=500):
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > threshold]
    
    box = []
    if len(contours) > 0:
        contour = contours[0]
        rect = cv2.minAreaRect(contour)
        rect = (
            (rect[0][0]*size_factor, rect[0][1]*size_factor), 
            (rect[1][0]*size_factor, rect[1][1]*size_factor), 
            rect[2]
            )
        
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
    return box

def get_warped_plate(image, box):
    
    H = int(distance.euclidean(box[0], box[1]))
    W = int(distance.euclidean(box[1], box[2]))
    
    src_pts = np.array(box).astype("float32")
    dst_pts = np.array([
        [0, H-1], 
        [0, 0], 
        [W-1, 0], 
        [W-1, H-1]
        ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(image, M, (W, H))
    
    if W < H:
        warped_img = cv2.rotate(warped_img, cv2.ROTATE_90_CLOCKWISE)
    
    return warped_img
