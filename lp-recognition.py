import sys
import traceback
import numpy as np
import darknet.python.darknet as dn

from os import path
from glob import glob
from metrics import nms
from src import darknet_label_conversion
from darknet.python.darknet import detect

OUTPUT_DIR = sys.argv[1]

OCR_THRESHOLD = 0.6
OCR_WEIGHTS = b'data/ocr/ocr-net.weights'
OCR_CONFIG  = b'data/ocr/ocr-net.cfg'
OCR_DATA = b'data/ocr/ocr-net.data'

if __name__ == '__main__':
    
    try:
        
        ocr_net  = dn.load_net(OCR_CONFIG, OCR_WEIGHTS, 0)
        ocr_meta = dn.load_meta(OCR_DATA)
        
        imgs_paths = sorted(glob(f'{OUTPUT_DIR}/*_plate.jpg'))
        print('\n\t***** PERFORMING OCR *****\n')
        
        for i, img_path in enumerate(imgs_paths):
            print(f'\tScanning {img_path}')
            
            name = path.basename(path.splitext(img_path)[0])
            R, (W, H) = detect(ocr_net, ocr_meta, bytes(img_path, encoding='utf-8'), thresh=OCR_THRESHOLD, nms=None)
            
            if len(R):
                L = darknet_label_conversion(R, W, H)
                L = nms(L, 0.45)
                L.sort(key=lambda x: x.tl()[0])
                
                lp_str = ''.join([chr(l.cl()) for l in L])
                
                label_name = name.split('_')[0] + '_label.txt'
                with open(f'{OUTPUT_DIR}/{label_name}', 'w') as f:
                    f.write(lp_str + '\n')
                
                print(f'\t\t --- LP: {lp_str} ---')
            
            else:
                print('No characters found')
        
    except:
        traceback.print_exc()
        sys.exit(1)
    
    sys.exit(0)
