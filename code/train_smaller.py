import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms

import dataset
import compression_net_smaller
import argparse
import os
from logger import Logger
import time
from IPython import embed
import pytorch_ssim

parser = argparse.ArgumentParser(description = 'Image Compression')
parser.add_argument('--batch-size', '-N', type=int, default=32, help='batch size')
#parser.add_argument('--train-paths', '-f', required=True, type=str, help='folder of training images')

parser.add_argument('--train-paths', '-f', nargs='+', required=True)

parser.add_argument('--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
# parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--num-workers', type=int, default=16, help='number of data loaders')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
parser.add_argument('--resume', type=int, help='the checkpoint epoch that you want to resume')
parser.add_argument('--start-epoch', type=int, default=0, help='start epoch')
parser.add_argument('--epochs', type=int, default=200, help = 'number of epochs')
parser.add_argument('--print_freq', type=int, default=1, help = 'frequency of printing')
parser.add_argument('--save_freq', type=int, default=50, help = 'frequency of saaving model')
parser.add_argument('--rnn-type', type=str, default='LSTM', help = 'LSTM or GRU')
parser.add_argument('--loss-type', type=str, default='L1', help = 'L1, L2 or SSIM')

def main():
    global args
    args = parser.parse_args()
    train_transform = transforms.Compose([
        transforms.RandomCrop((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    
    train_dataset = dataset.ImageFolder(roots = args.train_paths, transform = train_transform)
    train_loader = data.DataLoader(dataset = train_dataset,\
            batch_size = args.batch_size,\
            shuffle = True,\
            num_workers = args.num_workers 
            )

    encoder = compression_net.CompressionEncoder(rnn_type=args.rnn_type).cuda()
    binarizer = compression_net.CompressionBinarizer().cuda()
    decoder = compression_net.CompressionDecoder(rnn_type=args.rnn_type).cuda()

    optimizer = optim.Adam([{'params': encoder.parameters()},
                            {'params': binarizer.parameters()},
                            {'params': decoder.parameters()},],
                            lr = args.lr)
    
    start_epoch = args.start_epoch
    if args.resume:
        if os.path.isdir('checkpoint'):
            print("=> loading checkpoint '{}'".format(args.resume))
            encoder.load_state_dict(
                torch.load('checkpoint/{}_{}/encoder_{:08d}.pth'.format(args.rnn_type, args.loss_type, args.resume)))
            binarizer.load_state_dict(
                torch.load('checkpoint/{}_{}/binarizer_{:08d}.pth'.format(args.rnn_type, args.loss_type, args.resume)))
            decoder.load_state_dict(
                torch.load('checkpoint/{}_{}/decoder_{:08d}.pth'.format(args.rnn_type, args.loss_type, args.resume)))
            #args.start_epoch = checkpoint['epoch']
            #model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}'".format(args.resume))
            start_epoch = args.resume + 1 
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    torch.manual_seed(23)
    data_logger = Logger('./logs/', name = 'compression_net')
    scheduler = LS.MultiStepLR(optimizer, milestones=[3, 10, 20, 50, 100], gamma=0.5)

    for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
        scheduler.step()
        train(train_loader, encoder, binarizer, decoder, epoch, optimizer, data_logger)
        if epoch % args.save_freq == 0 or epoch == args.epochs-1:
            if not os.path.exists('checkpoint'):
                os.mkdir('checkpoint')
            if not os.path.exists('checkpoint/{}_{}'.format(args.rnn_type, args.loss_type)):
                os.mkdir('checkpoint/{}_{}'.format(args.rnn_type, args.loss_type))
            torch.save(encoder.state_dict(), 
                    'checkpoint/{}_{}/encoder_{:08d}.pth'.format(args.rnn_type, args.loss_type, epoch))
            torch.save(binarizer.state_dict(),
               'checkpoint/{}_{}/binarizer_{:08d}.pth'.format(args.rnn_type, args.loss_type, epoch))
            torch.save(decoder.state_dict(), 
                    'checkpoint/{}_{}/decoder_{:08d}.pth'.format(args.rnn_type, args.loss_type, epoch))

def train(train_loader, encoder, binarizer, decoder, epoch, optimizer, data_logger):
    for batch, data in enumerate(train_loader):
        begin_time = time.time()
        ## init lstm state
        if args.rnn_type == 'LSTM':
            encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                    Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
            encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
            encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))
            encoder_h_4 = (Variable(torch.zeros(data.size(0), 1024, 1, 1).cuda()),
                   Variable(torch.zeros(data.size(0), 1024, 1, 1).cuda()))

            decoder_h_0 = (Variable(torch.zeros(data.size(0), 512, 1, 1).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 1, 1).cuda()))
            decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()))
            decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                   Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()))
            decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                   Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()))
            decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()),
                       Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()))
        else:
            encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                        None)
            encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                        None)
            encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                        None)
            encoder_h_4 = (Variable(torch.zeros(data.size(0), 1024, 1, 1).cuda()),
                        None)

            decoder_h_0 = (Variable(torch.zeros(data.size(0), 512, 1, 1).cuda()),
                       None)
            decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, 2, 2).cuda()),
                       None)
            decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, 4, 4).cuda()),
                       None)
            decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, 8, 8).cuda()),
                       None)
            decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, 16, 16).cuda()),
                       None)

        input_img = Variable(data.cuda())
        optimizer.zero_grad()
        losses = []
        
        res = input_img - 0.5
        image = 0.5
        for i in range(args.iterations):   # default is 16
            encoded, encoder_h_1, encoder_h_2, encoder_h_3, encoder_h_4 = encoder(
                res, encoder_h_1, encoder_h_2, encoder_h_3, encoder_h_4)
            codes = binarizer(encoded)
            output, decoder_h_0, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes, decoder_h_0, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

            res = res - output
            image = image + output
            
            if args.loss_type == 'L1':
                losses.append(res.abs().mean())
            elif args.loss_type == 'L2':
                losses.append((res*res).mean())
            elif args.loss_type == 'SSIM':
                ssim_value = pytorch_ssim.ssim(input_img, image, size_average = False)
                difference = 1 - ssim_value
                losses.append((res.abs() * difference[:, None, None, None]).mean())
            #elif args.loss_type == 'localSSIM':
            else:
                input_img_local = input_img.view(input_img.shape[0], input_img.shape[1], 4, 8, 4, 8).permute(0, 2, 4, 1, 3, 5).contiguous()
                input_img_local = input_img_local.view(-1, input_img_local.shape[3], input_img_local.shape[4], input_img_local.shape[5])
                image_local = image.view(image.shape[0], image.shape[1], 4, 8, 4, 8).permute(0, 2, 4, 1, 3, 5).contiguous()
                image_local = image_local.view(-1, image_local.shape[3], image_local.shape[4], image_local.shape[5])
                ssim_value = pytorch_ssim.ssim(input_img_local, image_local, size_average = False)
                difference = 1 - ssim_value
                
                res_local = res.view(res.shape[0], res.shape[1], 4, 8, 4, 8).permute(0, 2, 4, 1, 3, 5).contiguous()
                res_local = res_local.view(-1, res_local.shape[3], res_local.shape[4], res_local.shape[5])
                losses.append((res_local.abs() * difference[:, None, None, None]).mean())
        loss = sum(losses) / args.iterations
        loss.backward()
        optimizer.step()

        end_time = time.time()
        
        if batch % args.print_freq == 0:
            print(
                '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Time: {:.4f} sec'.
                format(epoch, batch, len(train_loader), loss.data[0], end_time - begin_time))
        
        index = epoch * len(train_loader) + batch
        data_logger.scalar_summary(tag='train/loss', value=loss, step = index)

if __name__ == '__main__':
    main()
