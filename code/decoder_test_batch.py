import os
import argparse

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch
from torch.autograd import Variable

import compression_net

def decoder_test_batch(input,output_dir,model,iterations,rnn_type, use_cuda=True):
    content = np.load(input)
    codes = np.unpackbits(content['codes'])
    codes = np.reshape(codes, content['shape']).astype(np.float32) * 2 - 1

    iters, batch_size, channels, height, width = codes.shape
    #batch_size = 1
    h_num, w_num  = height // 2, width // 2
    codes = codes[:,0,:,:,:].reshape(iters, channels, h_num, 2, w_num, 2)

    codes = np.transpose(codes, (2, 0, 4, 1, 3, 5))
    codes = torch.from_numpy(codes)
    final_image = []
    for i in range(h_num):
        print('Deocding line {}'.format(i))
        images = decoder_test_each(codes[i], output_dir, model, rnn_type, use_cuda, iterations)
        final_image.append(images)
    final_image = np.array(final_image)
    final_image = np.transpose(final_image, (1, 0, 4, 2, 5, 3))
    d1, d2, d3, d4, d5, d6 = final_image.shape
    final_image = final_image.reshape(d1, d2*d3, d4*d5, d6)
    for iters in range(min(iterations, codes.size(0))):
        imsave(
        os.path.join(output_dir, '{:02d}.png'.format(iters)),
        (final_image[iters].clip(0, 1) * 255.0).astype(np.uint8))
    return min(iterations, codes.size(0))
    
def decoder_test_each(codes, output_dir, model, rnn_type, use_cuda, iterations):
    iters, batch_size, channels, height, width = codes.size()
    
    height = height * 16
    width = width * 16

    codes = Variable(codes, volatile=True)

    decoder = compression_net.CompressionDecoder(rnn_type)
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
    images = []
    for iters in range(min(iterations, codes.size(0))):
        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            codes[iters], decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
        image = image + output.data.cpu()
        images.append(image.numpy())
    return images
