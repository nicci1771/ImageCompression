import argparse
import os
import numpy as np
from scipy.misc import imread, imresize, imsave
from encoder_test import *
from decoder_test import *
from metric import *

import torch
from torch.autograd import Variable
from tqdm import tqdm
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--encodermodel', type=int, default = 199,  help='path to model')
parser.add_argument('--decodermodel', type=int, default = 199, help='path to model')
parser.add_argument('--input', '-i', type=str, 
        default = '../data/valid/', help='input dir')
parser.add_argument('--OutputCode', type=str, 
        default = 'result/codes/', help='output codes dir')
parser.add_argument('--OutputImage', type=str, 
        default = 'result/decode_image/', help='output image dir')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument('--encoderiterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--decoderiterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--rnn-type', type=str, default='LSTM', help='LSTM or GRU')

def is_image_file(filename):
    return filename.endswith('png')

def main():
    global args
    args = parser.parse_args()
    if not os.path.exists(args.OutputCode):
        os.makedirs(args.OutputCode)
    if not os.path.exists(args.OutputImage):
        os.makedirs(args.OutputImage)
    
    encoder_model = 'checkpoint/{}/encoder_{:08d}.pth'.format(args.rnn_type, args.encodermodel)
    decoder_model = 'checkpoint/{}/decoder_{:08d}.pth'.format(args.rnn_type, args.decodermodel)

    for filename in tqdm(os.listdir(args.input)):
        if is_image_file(filename):
            filename_new = filename.replace(' ','_')
            print('encoding ' + filename)
            encoder_test(args.input+filename, args.OutputCode+filename_new.replace('png','npz'), 
                    encoder_model, args.encoderiterations, args.rnn_type, args.cuda)
            output_path_dir = args.OutputImage+filename_new
            output_path_dir = output_path_dir[:-4]+'/'
            if not os.path.exists(output_path_dir):
                os.makedirs(output_path_dir)
            
            decoder_test(args.OutputCode+filename_new.replace('png','npz'), 
                    output_path_dir,
                    decoder_model, args.decoderiterations, args.rnn_type, args.cuda)
            for i in range(args.decoderiterations):
                decoded_img_path = output_path_dir + '{:02d}.png'.format(i)
                ssim_score = msssim(args.input+filename, decoded_img_path)
                psnr_score = psnr(args.input+filename, decoded_img_path)
                print("ssim: "+str(ssim_score)+" , psnr: "+str(psnr_score))
            #embed()

if __name__ == '__main__':
    main()
