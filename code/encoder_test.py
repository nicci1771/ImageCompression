import numpy as np
from scipy.misc import imread, imresize, imsave
import torch
from torch.autograd import Variable

import compression_net
import compression_net_smaller
import time

def encoder_test(input,output_path,model,iterations,rnn_type, use_cuda=True, network='Big'):
    raw_image = imread(input, mode='RGB')
    h, w, c = raw_image.shape
    new_h = (h // 32 +1)* 32
    new_w = (w // 32 + 1)*32
    image = np.zeros((new_h, new_w, c), dtype = np.float32)
    image[:h, :w, :] = raw_image
    image = torch.from_numpy(
        np.expand_dims(
            np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)), 0))
    batch_size, input_channels, height, width = image.size()
    assert height % 32 == 0 and width % 32 == 0

    image = Variable(image, volatile=True)
    if network == 'Big':
        encoder = compression_net.CompressionEncoder(rnn_type = rnn_type)
        binarizer = compression_net.CompressionBinarizer()
        decoder = compression_net.CompressionDecoder(rnn_type = rnn_type)
    else:
        encoder = compression_net_smaller.CompressionEncoder(rnn_type = rnn_type)
        binarizer = compression_net_smaller.CompressionBinarizer()
        decoder = compression_net_smaller.CompressionDecoder(rnn_type = rnn_type)

    encoder.eval()
    binarizer.eval()
    decoder.eval()

    encoder.load_state_dict(torch.load(model))
    binarizer.load_state_dict(
        torch.load(model.replace('encoder', 'binarizer')))
    decoder.load_state_dict(torch.load(model.replace('encoder', 'decoder')))

    encoder_h_1 = (Variable(
        torch.zeros(batch_size, 256, height // 4, width // 4), volatile=True),
                   Variable(
                       torch.zeros(batch_size, 256, height // 4, width // 4),
                       volatile=True))
    encoder_h_2 = (Variable(
        torch.zeros(batch_size, 512, height // 8, width // 8), volatile=True),
                   Variable(
                       torch.zeros(batch_size, 512, height // 8, width // 8),
                       volatile=True))
    encoder_h_3 = (Variable(
        torch.zeros(batch_size, 512, height // 16, width // 16), volatile=True),
                   Variable(
                       torch.zeros(batch_size, 512, height // 16, width // 16),
                       volatile=True))
    encoder_h_4 = (Variable(
        torch.zeros(batch_size, 1024, height // 32, width // 32), volatile=True),
                   Variable(
                       torch.zeros(batch_size, 1024, height // 32, width // 32),
                       volatile=True))

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
        encoder = encoder.cuda()
        binarizer = binarizer.cuda()
        decoder = decoder.cuda()

        image = image.cuda()

        encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
        encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
        encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

        decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
        decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
        decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
        decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())

    codes = []
    res = image - 0.5
    ori_res = image - 0.5
    
    if network == 'Big':
        for iters in range(iterations):
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                res, encoder_h_1, encoder_h_2, encoder_h_3)

            code = binarizer(encoded)

            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                 code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
            
            res = res - output
            #res = ori_res - output
            codes.append(code.data.cpu().numpy())
    else:
        for iters in range(iterations):
            encoded, encoder_h_1, encoder_h_2, encoder_h_3, encoder_h_4 = encoder(
                res, encoder_h_1, encoder_h_2, encoder_h_3, encoder_h_4)

            code = binarizer(encoded)

            output, decoder_h_0, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                 code, decoder_h_0, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)
            
            res = res - output
            #res = ori_res - output
            codes.append(code.data.cpu().numpy())

        #print('Iter: {:02d}; Loss: {:.06f}'.format(iters, res.data.abs().mean()))

    codes = (np.stack(codes).astype(np.int8) + 1) // 2

    export = np.packbits(codes.reshape(-1))
    np.savez_compressed(output_path, shape=codes.shape, codes=export)
