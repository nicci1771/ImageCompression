import os
import argparse

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch
from torch.autograd import Variable

import compression_net
import compression_net_smaller

def decoder_test(input,output_dir,model,iterations,rnn_type, use_cuda=True, network='Big', code_size=32):
    content = np.load(input)
    codes = np.unpackbits(content['codes'])
    codes = np.reshape(codes, content['shape']).astype(np.float32) * 2 - 1
    delta = content['delta']
    delta_h = int(delta[0])
    delta_w = int(delta[1])

    codes = torch.from_numpy(codes)
    iters, batch_size, channels, height, width = codes.size()
    height = height * 16
    width = width * 16

    codes = Variable(codes, volatile=True)

    if network == 'Big':
        decoder = compression_net.CompressionDecoder(rnn_type, code_size)
    else:
        decoder = compression_net_smaller.CompressionDecoder(rnn_type)
    decoder.eval()

    if use_cuda:
        decoder.load_state_dict(torch.load(model))
    else:
        decoder.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))


    decoder_h_0 = (Variable(
        torch.zeros(batch_size, 512, height // 32, width // 32), volatile=True),
                   Variable(
                       torch.zeros(batch_size, 512, height // 32, width // 32),
                       volatile=True))
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
        
        decoder_h_0 = (decoder_h_0[0].cuda(), decoder_h_0[1].cuda())
        decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
        decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
        decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
        decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

    image = torch.zeros(1, 3, height, width) + 0.5
    if network == 'Big':
        for iters in range(min(iterations, codes.size(0))):
            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes[iters], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
            image = image + output.data.cpu()
            image_save = image[:, :, :-delta_h, :-delta_w]
            #from IPython import embed
            #embed()
            imsave(
                os.path.join(output_dir, '{:02d}.png'.format(iters)),
                np.squeeze(image_save.numpy().clip(0, 1) * 255.0).astype(np.uint8)
                    .transpose(1, 2, 0))
    else:
        for iters in range(min(iterations, codes.size(0))):
            output, decoder_h_0, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes[iters], decoder_h_0, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
            image = image + output.data.cpu()
            image_save = image[:, :, :-delta_h, :-delta_w]

            imsave(
                os.path.join(output_dir, '{:02d}.png'.format(iters)),
                np.squeeze(image_save.numpy().clip(0, 1) * 255.0).astype(np.uint8)
                    .transpose(1, 2, 0))

    return min(iterations, codes.size(0))
