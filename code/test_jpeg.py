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
        default = '/home/manchen/wmc/compression_result/img/jpeg_01/', help='output image dir')

def is_image_file(filename):
    return filename.endswith('png')

def main():
    global args
    args = parser.parse_args()
    if not os.path.exists(args.OutputImage):
        os.makedirs(args.OutputImage)
    
    cnt = 1
    total_psnr = 0.0
    total_ssim = 0.0
    total_bpp = 0.0
    input_list = os.listdir(args.input)
    np.random.seed(23)
    index = np.random.permutation(len(input_list))
    input_list = [input_list[i] for i in index[:100]]
    for filename in tqdm(input_list):
        """
        if cnt <=10:
            cnt = cnt + 1
        else:
            break
            """
        if is_image_file(filename):
            filename_new = filename.replace(' ','_')
            output_path_dir = args.OutputImage
            output_path_dir = output_path_dir[:-4]+'/'
            if not os.path.exists(output_path_dir):
                os.makedirs(output_path_dir)
            
            jpeg_img_path = output_path_dir + filename_new.split('.')[0] + '.jpg'
            img = cv2.imread(args.input+filename)
            cv2.imwrite(jpeg_img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 1])
            ssim_score = msssim(args.input+filename, jpeg_img_path)
            psnr_score = psnr(args.input+filename, jpeg_img_path)
            jpeg_img_size = os.path.getsize(jpeg_img_path)
            jpeg_img = cv2.imread(jpeg_img_path)
            h, w, c = jpeg_img.shape
            bpp = jpeg_img_size * 8.0 / (h*w)
            print("ssim: "+str(ssim_score)+" , psnr: "+str(psnr_score)+" ,bpp: "+str(bpp))
        total_psnr = total_psnr + psnr_score
        total_ssim = total_ssim + ssim_score
        total_bpp = total_bpp + bpp

    #l = len(os.listdir(args.input))
    l = 100
    avg_psnr = total_psnr / l
    avg_ssim = total_ssim / l
    avg_bpp = total_bpp / l
    print("avg ssim: "+str(avg_ssim)+" , avg psnr:" + str(avg_psnr)+" , avg bpp:" + str(avg_bpp))
            
if __name__ == '__main__':
    main()
