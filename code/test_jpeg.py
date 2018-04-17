import argparse
import os
import numpy as np
from scipy.misc import imread, imresize, imsave
from metric import *

from tqdm import tqdm
from IPython import embed
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, 
        default = '../data/test/', help='input dir')
parser.add_argument('--OutputImage', type=str, 
        default = 'result/jpeg_image/', help='output image dir')

def is_image_file(filename):
    return filename.endswith('png')

def main():
    global args
    args = parser.parse_args()
    if not os.path.exists(args.OutputImage):
        os.makedirs(args.OutputImage)
    
    for filename in tqdm(os.listdir(args.input)):
        if is_image_file(filename):
            filename_new = filename.replace(' ','_')
            output_path_dir = args.OutputImage+filename_new
            output_path_dir = output_path_dir[:-4]+'/'
            if not os.path.exists(output_path_dir):
                os.makedirs(output_path_dir)
            
            jpeg_img_path = output_path_dir + filename_new.split('.')[0] + '.jpeg'
            img = cv2.imread(args.input+filename)
            cv2.imwrite(jpeg_img_path, img)
            ssim_score = msssim(args.input+filename, jpeg_img_path)
            psnr_score = psnr(args.input+filename, jpeg_img_path)
            print("ssim: "+str(ssim_score)+" , psnr: "+str(psnr_score))
            
if __name__ == '__main__':
    main()
