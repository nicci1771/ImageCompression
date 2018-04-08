import argparse
import os
import numpy as np
from scipy.misc import imread, imresize, imsave
from encoder import *
from decoder import *

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', required=True, type=str, help='path to model')
parser.add_argument('--input', '-i', required=True, type=str, help='input dir')
parser.add_argument('--OutputCode', '-o', required=True, type=str, help='output codes dir')
parser.add_argument('--OutputImage', '-o', required=True, type=str, help='output image dir')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')

def is_image_file(filename):
    return filename.endswith('png')

def main():
    global args
    args = parser.parse_args()
    if not os.path.exists(args.OutputCode):
        os.makedirs(args.OutputCode)
    if not os.path.exists(args.OutputImage):
        os.makedirs(args.OutputImage)

    for filename in os.listdir(args.input):
        if is_image_file(filename):
            print 'encoding ' + filename
            encoder(args.input+filename, args.OutputCode+filename.replace('png','npz'), 
                    args.model, args.iterations, args.cuda)
            decoder(args.OutputCode+filename.replace('png','npz'), 
                    args.OutputImage+filename,
                    args.model, args.iterations, args.cuda)
if __name__ == '__main__':
    main()
