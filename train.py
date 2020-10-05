import os
import math
import torch
import numpy as np
import argparse
import random
import yaml
from easydict import EasyDict
import gensim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter

from models.standard import *
from util.utils import *
from util.loss import *
from util.data import *


def set_parser():
    parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
    parser.add_argument('--train_set', default='/data2/wn/Video_dataset/train/frame', help='facades')
    parser.add_argument('--test_set', default='/data2/wn/Video_dataset/test/frame', help='facades')
    parser.add_argument('--logfile', default='./log', help='trainlogs.dat')
    parser.add_argument('--log_freq', type=int, default=25, help='log frequency (/iteration)')
    parser.add_argument('--sample_freq', type=int, default=250, help='sample frequency (/iteration)')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency (/epoch)')
    parser.add_argument('--sample_path', default="./samples", help='sample path')
    parser.add_argument('--board_path', default="./board", help='tensorboard path')
    parser.add_argument('--checkpoint_path', default="", help='load pre-trained model?')
    parser.add_argument('--save_path', default="./param", help='the path to save the model to')
    parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
    parser.add_argument('--epoch', type=int, default=500, help='number of epochs to train for')
    parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='不知道是啥channel')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.002')
    parser.add_argument('--beta1', type=float, default=0, help='beta1 for adam. default=0.5')
    parser.add_argument('--workers', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=1234, help='random seed to use. Default=123')
    parser.add_argument('--L1lamb', type=int, default=10, help='weight on L1 term in objective')
    parser.add_argument('--Stylelamb', type=int, default=1000, help='weight on Style term in objective')
    parser.add_argument('--Contentlamb', type=int, default=1, help='weight on Content term in objective')
    parser.add_argument('--Adversariallamb', type=int, default=0.1, help='weight on Adv term in objective')
    return parser.parse_args()


if __name__ == '__main__':
    opt = set_parser()

    # assert torch.cuda.is_available()
    # device = torch.device('cuda')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    netC = ContentGenerator()
    netS = StyleGenerator()
    netF = FusionGenerator()
    netD = Discriminator(in_channels=7, use_sigmoid=True)

    optimizerC = optim.Adam(netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerS = optim.Adam(netS.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerF = optim.Adam(netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr * 0.1, betas=(opt.beta1, 0.999))

    epoch = 1
    count = 0

    criterionGAN = AdversarialLoss()
    criterionSTYLE = StyleLoss()
    criterionCONTENT = PerceptualLoss()
    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()

    netC = netC.to(device)
    netS = netS.to(device)
    netF = netF.to(device)
    netD = netD.to(device)
    criterionGAN = criterionGAN.to(device)
    criterionL1 = criterionL1.to(device)
    critertionSTYLE = criterionSTYLE.to(device)
    criterionCONTENT = criterionCONTENT.to(device)
    criterionMSE = criterionMSE.to(device)

    if opt.checkpoint_path:
        epoch, count = load_state(opt.checkpoint_path, netC, netS, netF, netD, optimizerC,
                                  optimizerS, optimizerF, optimizerD)

    train_set = MyDataset(opt.train_set)
    test_set = MyDataset(opt.test_set)

    training_data_loader = DataLoader(train_set, batch_size=opt.train_batch_size,
                                      pin_memory=True, num_workers=opt.workers)
    testing_data_loader = DataLoader(train_set, batch_size=opt.test_batch_size,
                                     pin_memory=True, num_workers=opt.workers)
    sample_iterator = create_iterator(6, test_set)

    writer = SummaryWriter(opt.board_path)

    for epoch in range(epoch, opt.epoch + 1):
        for iteration, [input, real, prev] in enumerate(training_data_loader):

            input, real, prev = input.to(device), real.to(device), prev.to(device)

            ############################
            # (1) train discriminator
            ############################

            for p in netD.parameters():
                p.requires_grad = True
            for p in netC.parameters():
                p.requires_grad = False
            for p in netS.parameters():
                p.requires_grad = False
            for p in netF.parameters():
                p.requires_grad = False

            netD.zero_grad()

            # train with fake
            fake = netF(netC(input), netS(prev))
            pred_fake = netD(torch.cat((input, prev, fake), 1))
            loss_D_fake = criterionGAN(pred_fake, False, True)

            # train with real
            pred_real = netD(torch.cat((input, prev, real), 1))
            loss_D_real = criterionGAN(pred_real, True, True)

            # combine the loss
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()

            # Discriminator parameters update every 12 iterations
            if iteration == 1 or iteration % 12 == 0:
                optimizerD.step()

            ############################
            # (2) train content generator
            ############################

            for p in netD.parameters():
                p.requires_grad = False
            for p in netC.parameters():
                p.requires_grad = True
            for p in netS.parameters():
                p.requires_grad = False
            for p in netF.parameters():
                p.requires_grad = False

            netC.zero_grad()

            fake = netF(netC(input), netS(prev))
            loss_C_content = criterionCONTENT(fake, real)
            loss_C_style = - math.log(criterionSTYLE(fake, real).item())
            loss_C = loss_C_content + loss_C_style
            # pred_fake = netD(torch.cat((input, prev, fake), 1))
            # Adversarial loss
            # loss_G_gan = criterionGAN(pred_fake, True, False)

            loss_C.backward()
            optimizerC.step()

            ############################
            # (3) train style generator
            ############################

            for p in netD.parameters():
                p.requires_grad = False
            for p in netC.parameters():
                p.requires_grad = False
            for p in netS.parameters():
                p.requires_grad = True
            for p in netF.parameters():
                p.requires_grad = False

            netS.zero_grad()

            fake = netF(netC(input), netS(prev))
            loss_S_style = criterionSTYLE(fake, real)
            loss_S_content = - math.log(criterionCONTENT(fake, real).item())
            loss_S = loss_S_style + loss_S_content

            # pred_fake = netD(torch.cat((input, prev, fake), 1))
            # Adversarial loss
            # loss_G_gan = criterionGAN(pred_fake, True, False)

            loss_S.backward()
            optimizerS.step()

            ############################
            # (4) train fusion generator
            ############################

            for p in netD.parameters():
                p.requires_grad = False
            for p in netC.parameters():
                p.requires_grad = False
            for p in netS.parameters():
                p.requires_grad = False
            for p in netF.parameters():
                p.requires_grad = True

            netF.zero_grad()

            fake = netF(netC(input), netS(prev))
            pred_fake = netD(torch.cat((input, prev, fake), 1))

            # Adversarial loss
            loss_F_gan = criterionGAN(pred_fake, True, False)
            # L1 loss
            loss_F_l1 = criterionL1(fake, real) * opt.L1lamb
            # style loss
            loss_F_style = criterionSTYLE(fake, real) * opt.Stylelamb
            # content loss
            loss_F_content = criterionCONTENT(fake, real) * opt.Contentlamb

            # sum the loss up
            loss_F = loss_F_gan + loss_F_l1 + loss_F_style + loss_F_content

            loss_F.backward()
            optimizerF.step()

            count = count + 1

            ############################
            # (5) log & sample & save
            ############################

            if iteration % opt.log_freq == 0:
                logs = [("epoc", epoch), ("iter", iteration),
                        ("Loss_C_content", loss_C_content.item()), ("Loss_S_style", loss_S_style.item()),
                        ("loss_C_style", loss_C_style), ("loss_S_content", loss_S_content),
                        ("Loss_F", loss_F.item()), ("Loss_F_gan", loss_F_gan.item()),
                        ("Loss_F_l1", loss_F_l1.item()), ("Loss_F_style", loss_F_style.item()),
                        ("Loss_F_content", loss_F_content.item()), ("Loss_D", loss_D.item())]
                log_train_data(logs, opt)

                writer.add_scalar("loss_C_content", loss_C_content.item(), count)
                writer.add_scalar("loss_S_style", loss_S_style.item(), count)
                writer.add_scalar("loss_C_style", loss_C_style, count)
                writer.add_scalar("loss_S_content", loss_S_content, count)
                writer.add_scalar("loss_F", loss_F.item(), count)
                writer.add_scalar("loss_F_gan", loss_F_gan.item(), count)
                writer.add_scalar("loss_F_l1", loss_F_l1.item(), count)
                writer.add_scalar("loss_F_style", loss_F_style.item(), count)
                writer.add_scalar("loss_F_content", loss_F_content.item(), count)
                writer.add_scalar("loss_D", loss_D.item(), count)
                writer.add_scalar("loss_D_real", loss_D_real.item(), count)
                writer.add_scalar("loss_D_fake", loss_D_fake.item(), count)

            if iteration % opt.sample_freq == 0:
                sample(epoch, iteration, count, opt, sample_iterator, netC, netS, netF, device, writer)

            print("===> Epoch[{}]({}/{})".format(epoch, iteration, len(training_data_loader)))

        if epoch % opt.save_freq == 0:
            save_state({'state_dictC': netC.state_dict(),
                        'optimizerC': optimizerC.state_dict(),
                        'state_dictS': netS.state_dict(),
                        'optimizerS': optimizerS.state_dict(),
                        'state_dictF': netF.state_dict(),
                        'optimizerF': optimizerF.state_dict(),
                        'state_dictD': netD.state_dict(),
                        'optimizerD': optimizerD.state_dict(),
                        'epoch': epoch,
                        'count': count}, opt.save_path, epoch)
