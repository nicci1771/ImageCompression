import os
import argparse

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch
from torch.autograd import Variable

import compression_net

def decoder(input,output_dir,model,iterations,use_cuda=True):
    content = np.load(input)
    codes = np.unpackbits(content['codes'])
    codes = np.reshape(codes, content['shape']).astype(np.float32) * 2 - 1

    codes = torch.from_numpy(codes)
    iters, batch_size, channels, height, width = codes.size()
    height = height * 16
    width = width * 16

    codes = Variable(codes, volatile=True)

    decoder = compression_net.CompressionDecoder()
    decoder.eval()

    decoder.load_state_dict(torch.load(model))

    decoder_h_1 = (Variable(
        torch.zeros(batch_size, 512, height // 16, width // 16), volatile=True),
                   Variable(
                       torch.zeros(batch_size, 512, height // 16, width // 16),
                       volatile=True))
    decoder_h_2 = (Variable(
        torch.zeros(batch_size, 512, height // 8, width // 8), volatile=True),
                   Variable(
                       torch.zeros(batch_size, 512, height // 8, width // 8),
                       volatile=True))
    decoder_h_3 = (Variable(
        torch.zeros(batch_size, 256, height // 4, width // 4), volatile=True),
                   Variable(
                       torch.zeros(batch_size, 256, height // 4, width // 4),
                       volatile=True))
    decoder_h_4 = (Variable(
        torch.zeros(batch_size, 128, height // 2, width // 2), volatile=True),
                   Variable(
                       torch.zeros(batch_size, 128, height // 2, width // 2),
                       volatile=True))

    if use_cuda:
        decoder = decoder.cuda()

        codes = codes.cuda()

        decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
        decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
        decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
        decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

    image = torch.zeros(1, 3, height, width) + 0.5
    for iters in range(min(iterations, codes.size(0))):
        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            codes[iters], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
        image = image + output.data.cpu()

        imsave(
            os.path.join(output_dir, '{:02d}.png'.format(iters)),
            np.squeeze(image.numpy().clip(0, 1) * 255.0).astype(np.uint8)
                .transpose(1, 2, 0))
