import argparse
import os
import numpy as np
from scipy.misc import imread, imresize, imsave
from encoder_test import *
from decoder_test import *
from encoder_test_batch import *
from decoder_test_batch import *
from metric import *

import torch
from torch.autograd import Variable
from tqdm import tqdm
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument('--encodermodel', type=int, default = 199,  help='path to model')
parser.add_argument('--decodermodel', type=int, default = 199, help='path to model')
parser.add_argument('--input', '-i', type=str, 
        default = '../data/test/', help='input dir')
parser.add_argument('--OutputCode', type=str, 
        default = 'result/codes/', help='output codes dir')
parser.add_argument('--OutputImage', type=str, 
        default = 'result/decode_image/', help='output image dir')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument('--encoderiterations', type=int, default=8, help='unroll iterations')
parser.add_argument('--decoderiterations', type=int, default=8, help='unroll iterations')
parser.add_argument('--rnn-type', type=str, default='LSTM', help='LSTM or GRU')
parser.add_argument('--loss-type', type=str, default='L1', help='L1, L2 or SSIM')
parser.add_argument('--code-size', type=int, default=32)
parser.add_argument('--bybatch', default=False, action='store_true')
parser.add_argument('--network', type=str, default='Big', help='big or small')

def is_image_file(filename):
    return filename.endswith('png')

def main():
    global args
    args = parser.parse_args()
    if not os.path.exists(args.OutputCode):
        os.makedirs(args.OutputCode)
    if not os.path.exists(args.OutputImage):
        os.makedirs(args.OutputImage)
    
    encoder_model = 'checkpoint/{}_{}_{}/encoder_{:08d}.pth'.format(args.rnn_type, args.loss_type,args.code_size, args.encodermodel)
    decoder_model = 'checkpoint/{}_{}_{}/decoder_{:08d}.pth'.format(args.rnn_type, args.loss_type, args.code_size, args.decodermodel)
    #cnt = 0
    for filename in tqdm(os.listdir(args.input)):
        """
        if cnt < 11:
            cnt = cnt + 1
            continue
        """
        if is_image_file(filename):
            
            start_time = time.time()
            filename_new = filename.replace(' ','_')
            print('encoding ' + filename)
            if args.bybatch:
                encoder_test_batch(args.input+filename, args.OutputCode+filename_new.replace('png','npz'), encoder_model, args.encoderiterations, args.rnn_type, args.cuda, args.network)
            else:
                encoder_test(args.input+filename, args.OutputCode+filename_new.replace('png','npz'), encoder_model, args.encoderiterations, args.rnn_type, args.cuda, args.network, args.code_size)
            encoded_time = time.time()
            #print('encode time is {}'.format(encoded_time - start_time))
            output_path_dir = args.OutputImage+filename_new
            output_path_dir = output_path_dir[:-4]+'/'
            if not os.path.exists(output_path_dir):
                os.makedirs(output_path_dir)
            
            if args.bybatch:
                iter_num = decoder_test_batch(args.OutputCode+filename_new.replace('png','npz'), 
                    output_path_dir,
                    decoder_model, args.decoderiterations, args.rnn_type, args.cuda, args.network)
            else:
                iter_num = decoder_test(args.OutputCode+filename_new.replace('png','npz'), 
                    output_path_dir,
                    decoder_model, args.decoderiterations, args.rnn_type, args.cuda, args.network, args.code_size)
            for i in range(min(iter_num, args.decoderiterations)):
                decoded_img_path = output_path_dir + '{:02d}.png'.format(i)
                ssim_score = msssim(args.input+filename, decoded_img_path)
                psnr_score = psnr(args.input+filename, decoded_img_path)
                print("ssim: "+str(ssim_score)+" , psnr: "+str(psnr_score))
            
            #print('decode time is {}'.format(time.time() - encoded_time))

if __name__ == '__main__':
    main()
