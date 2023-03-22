import re
import numpy as np
from os import path

class Label:
    
    def __init__(self, cl=-1, tl=np.array([0, 0]), br=np.array([0, 0])):
        
        self.__tl 	= tl
        self.__br 	= br
        self.__cl 	= cl
    
    def __str__(self):
        return f'Class: {self.__cl}, top_left(x:{self.__tl[0]}, y:{self.__tl[1]}), bottom_right(x:{self.__br[0]}, y:{self.__br[1]})'
    
    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)
    
    def wh(self):
        return (self.__br - self.__tl)
    
    def cc(self):
        return (self.__tl + self.wh()/2)
    
    def tl(self):
        return self.__tl
    
    def br(self):
        return self.__br
    
    def tr(self):
        return np.array([self.__br[0], self.__tl[1]])
    
    def bl(self):
        return np.array([self.__tl[0], self.__br[1]])
    
    def cl(self):
        return self.__cl
    
    def area(self):
        return np.prod(self.wh())
    
    def set_class(self, cl):
        self.__cl = cl
    
    def set_tl(self, tl):
        self.__tl = tl
    
    def set_br(self, br):
        self.__br = br
    
    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - (0.5*wh)
        self.__br = cc + (0.5*wh)

def read_box(file_path):
    if not path.isfile(file_path):
        return []
    
    pts = []
    with open(file_path, 'r') as file:
        for line in file:
            line = re.sub('[\[\]]', "", line)
            values = line.strip().split(',')
            
            for value in values:
                value = value.split()
                pts.append([int(x) for x in value])
                
    return np.array(pts)

def read_label(file_path):
    if not path.isfile(file_path):
        return None
    
    with open(file_path, 'r') as file:
        for line in file:
            text = line.strip()
            
            if len(text) > 7:
                text = text[:3] + text[-4:]
    
    return text

def darknet_label_conversion(R, img_width, img_height):
    WH = np.array((img_width, img_height), dtype=float)
    L  = []
    
    for r in R:
        center = np.array(r[2][:2]) / WH
        WH2 = (np.array(r[2][2:]) / WH) * 0.5
        
        L.append(Label(ord(r[0]), tl=center-WH2, br=center+WH2))
    
    return L
