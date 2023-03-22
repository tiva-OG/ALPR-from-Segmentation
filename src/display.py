import numpy as np
import torchvision.transforms.functional as F

from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms import Compose, ToTensor

transform = Compose([ToTensor()])

def show(**images):
    n = len(images.items())
    plt.figure(figsize=(8, 8))
    
    for i, (name, image) in enumerate(images.items()):
        image = image.detach()
        image = F.to_pil_image(image)
        
        plt.subplot(1, n, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
        
    plt.show()

def show_batch(images, labels):
    """ Show batch of images with masks """
    
    images = make_grid(images)
    
    labels = labels.unsqueeze(1) if labels.ndim < 4 else labels
    labels = make_grid(labels)
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))
    
    plt.show()

def show_plate(image, prediction):
    """ Show image alongside segmented plate """
    
    image = transform(image)
    prediction = transform(prediction)
    
    image = make_grid(image)
    prediction = make_grid(prediction)
    
    show(image=image, prediction=prediction)
