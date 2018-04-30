import argparse
import os
import numpy as np
from scipy.misc import imread, imresize, imsave
from encoder_test import *
from decoder_test import *
from encoder_test_batch import *
from decoder_test_batch import *
from metric import *
import cv2

import torch
from torch.autograd import Variable
from tqdm import tqdm
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--encodermodel', type=int, default = 200,  help='path to model')
parser.add_argument('--decodermodel', type=int, default = 200, help='path to model')
parser.add_argument('--input', '-i', type=str, 
        default = '../data/test/', help='input dir')
parser.add_argument('--OutputCode', type=str, 
        default = '/home/manchen/wmc/compression_result/code/', help='output codes dir')
parser.add_argument('--OutputImage', type=str, 
        default = '/home/manchen/wmc/compression_result/img/', help='output image dir')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument('--encoderiterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--decoderiterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--rnn-type', type=str, default='GRU', help='LSTM or GRU')
parser.add_argument('--loss-type', type=str, default='SSIM', help='L1, L2 or SSIM')
parser.add_argument('--code-size', type=int, default=4)
parser.add_argument('--bybatch', default=False, action='store_true')
parser.add_argument('--network', type=str, default='Big', help='big or small')

def is_image_file(filename):
    return filename.endswith('png')

def main():
    global args
    args = parser.parse_args()
    code_path = args.OutputCode + '{}_{}_{}/'.format(args.rnn_type, args.loss_type, args.code_size)
    img_path =args.OutputImage + '{}_{}_{}/'.format(args.rnn_type, args.loss_type, args.code_size)
    if not os.path.exists(code_path):
        os.makedirs(code_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    
    encoder_model = 'checkpoint/{}_{}_{}/encoder_{:08d}.pth'.format(args.rnn_type, args.loss_type,args.code_size, args.encodermodel)
    decoder_model = 'checkpoint/{}_{}_{}/decoder_{:08d}.pth'.format(args.rnn_type, args.loss_type, args.code_size, args.decodermodel)
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
        if cnt <= 100:
            cnt = cnt + 1
        else:
            break
        """
        if is_image_file(filename):
            
            start_time = time.time()
            filename_new = filename.replace(' ','_')
            print('encoding ' + filename)
            if args.bybatch:
                encoder_test_batch(args.input+filename, code_path+filename_new.replace('png','npz'), encoder_model, args.encoderiterations, args.rnn_type, args.cuda, args.network, args.code_size)
            else:
                encoder_test(args.input+filename, img_path+filename_new.replace('png','npz'), encoder_model, args.encoderiterations, args.rnn_type, args.cuda, args.network, args.code_size)
            encoded_time = time.time()
            #print('encode time is {}'.format(encoded_time - start_time))
            output_path_dir = img_path+filename_new
            output_path_dir = output_path_dir[:-4]+'/'
            if not os.path.exists(output_path_dir):
                os.makedirs(output_path_dir)
            
            if args.bybatch:
                iter_num = decoder_test_batch(code_path+filename_new.replace('png','npz'), 
                    output_path_dir,
                    decoder_model, args.decoderiterations, args.rnn_type, args.cuda, args.network, args.code_size)
            else:
                iter_num = decoder_test(code_path+filename_new.replace('png','npz'), 
                    output_path_dir,
                    decoder_model, args.decoderiterations, args.rnn_type, args.cuda, args.network, args.code_size)
            code_file_size = os.path.getsize(code_path+filename_new.replace('png','npz')) 
            for i in range(min(iter_num, args.decoderiterations)):
                decoded_img_path = output_path_dir + '{:02d}.png'.format(i)
                ssim_score = msssim(args.input+filename, decoded_img_path)
                psnr_score = psnr(args.input+filename, decoded_img_path)
                decode_img = cv2.imread(decoded_img_path)
                h, w, c = decode_img.shape
                bpp = code_file_size * 8.0 / (h*w)
                print("ssim: "+str(ssim_score)+" , psnr: "+str(psnr_score)+" ,bpp: "+str(bpp))
            total_psnr = total_psnr + psnr_score
            total_ssim = total_ssim + ssim_score
            total_bpp = total_bpp + bpp

            #print('decode time is {}'.format(time.time() - encoded_time))
    #l = len(os.listdir(args.input))
    l = 100
    avg_psnr = total_psnr / l
    avg_ssim = total_ssim / l
    avg_bpp = total_bpp / l
    print("avg ssim: "+str(avg_ssim)+" , avg psnr:" +str(avg_psnr)+" ,avg bpp: "+str(avg_bpp))   
if __name__ == '__main__':
    main()
