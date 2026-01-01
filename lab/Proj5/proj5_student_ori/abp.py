import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms as T

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='warm', type=str, help='warm or cold')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate')
    parser.add_argument('--langevin_step_size', type=float, default=0.05, help='stepsize of langevin')
    parser.add_argument('--langevin_num_steps', type=int, default=120, help='number of langevin steps')
    parser.add_argument('--mse_sigma', type=float, default=1, help='scale of mse loss, factor analysis')

    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--nz', type=int, default=2, help='size of the latent z')
    parser.add_argument('--img_size', type=int, default=128, help='image resolution')
    parser.add_argument('--prior_sigma', type=float, default=1, help='prior of z')
    parser.add_argument('--ninterp', default=12, type=int)
    
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--gamma', default=0.996, help='lr decay')
    parser.add_argument('--n_epochs', default=2000, type=int)
    parser.add_argument('--n_log', type=int, default=100, help='log each n iterations')
    parser.add_argument('--n_plot', type=int, default=200, help='plot each n epochs')
    parser.add_argument('--n_stats', type=int, default=100, help='stats each n epochs')
    return parser.parse_args()


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def set_cuda():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


# hydra-style logging
def get_logger(exp_dir):
    logger = logging.getLogger(__name__)
    logger.handlers = []
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(exp_dir, 'output.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.setLevel(logging.INFO)
    return logger


def load_img_tensors(img_dir, transform, device):
    img_tensors = []
    for fname in os.listdir(img_dir):
        img = Image.open(os.path.join(img_dir, fname))
        img = transform(img)
        img_tensors.append(img)
    return torch.stack(img_tensors).to(device)


def sample_gaussian_prior(batch_size, nz, sig=1):
    return sig * torch.randn(batch_size, nz).to(device)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


class GenNet(nn.Module):
    def __init__(self):
        """
        Decode a 2-dim latent variable z to an image at 128x128.
        (2) -> (8192) -> (512, 4, 4) -> (256, 8, 8) -> (128, 16, 16) -> (64, 32, 32) -> (32, 64, 64) -> (3, 128, 128)
        """
        super().__init__()
        self.latent_proj = nn.Linear(2, 8192)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.latent_proj(z.view(-1,2))
        # [8192]
        x = F.relu(x).view(-1, 512, 4, 4)
        # [512, 4, 4]
        x = self.block1(x)
        # [256, 8, 8]
        x = self.block2(x)
        # [128, 16, 16]
        x = self.block3(x)
        # [64, 32, 32]
        x = self.block4(x)
        # [32, 64, 64]
        x = self.block5(x)
        # [3, 128, 128]
        return x


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.Gnet = GenNet()
        self.Gnet.apply(weights_init_xavier)

    def sample_langevin(self, z, x):
        z = z.clone().detach()
        z.requires_grad = True
        for i in range(self.args.langevin_num_steps):
            x_hat = self.Gnet(z)
            log_lkhd = 1.0 / (2.0 * self.args.mse_sigma * self.args.mse_sigma) * F.mse_loss(x_hat, x, reduction='sum')
            z_grad = torch.autograd.grad(log_lkhd, z)[0]

            z.data = z.data - 0.5 * (self.args.langevin_step_size ** 2) * (z_grad  + 1.0 / (self.args.prior_sigma ** 2) * z.data)
            z.data += 4 * self.args.langevin_step_size * torch.randn_like(z).data

        return z.detach()

    def forward(self, z):
        return self.Gnet(z)


def plot_stats(exp_dir, stats, interval):
    content = stats.keys()
    f, axs = plt.subplots(len(content), 1, figsize=(20, len(content) * 5))
    for j, (k, v) in enumerate(stats.items()):
        if len(content) ==1:
            axs.plot(interval, v)
            axs.set_ylabel(k)
        else:
            axs[j].plot(interval, v)
            axs[j].set_ylabel(k)
    f.savefig(os.path.join(exp_dir, 'stat.pdf'), bbox_inches='tight')
    f.savefig(os.path.join(exp_dir, 'stat.png'), bbox_inches='tight')
    plt.close(f)


def save_images(img, path, nrow):
    vutils.save_image(img, path, normalize=True, nrow=nrow)


def save_interp(model, args, z_std, device, path):
    n_interp = args.ninterp
    inter_image = torch.zeros((n_interp ** 2, 3, args.img_size, args.img_size))
    inter_number1 = torch.linspace(-z_std[0], z_std[0], n_interp)
    inter_number2 = torch.linspace(-z_std[1], z_std[1], n_interp)
    height, width = torch.meshgrid(inter_number1, inter_number2)
    z_inter = torch.column_stack((height.reshape(-1, 1), width.reshape(-1, 1)))

    for i in range(n_interp):
        z_g_si = z_inter[(i * n_interp): ((i + 1) * n_interp)]
        inter_image[i * n_interp:((i + 1) * n_interp)] = model.Gnet(
            z_g_si.to(device).float())
    save_images(inter_image,path,n_interp)


def run(args, exp_dir, logger):
    transform = T.Compose([
        T.Resize([args.img_size, args.img_size]),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    data = load_img_tensors(img_dir='images', transform=transform, device=device)
    batch_size = len(data)

    model = Model(args).to(device)

    optG = torch.optim.Adam(model.Gnet.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optG, args.gamma)

    stats = {'loss': []}
    interval = []

    z_0 = Variable(sample_gaussian_prior(
        batch_size=batch_size, nz=args.nz, sig=args.prior_sigma
    ))
    z = z_0.clone()
    for epoch in range(args.n_epochs):
        # update latent z via Langevin sampling
        if args.start == 'warm':
            # TODO
            pass
        else:
            # cold
            # TODO
            pass

        # update synthesized images
        # TODO, variable named `x_hat`

        # compute loss
        # TODO, variable named `loss`
        
        # backpropagate and update generator parameters
        # TODO
        lr_schedule.step()

        # log
        if (epoch + 1) % args.n_log == 0:
            logger.info(
                f"Epoch {epoch+1:>4d}: Loss = {loss:<4.2f}, " + 
                f"z_mean = {z.mean(0).detach().cpu().numpy()}, " + 
                f"z_std = {z.std(0).detach().cpu().numpy()}"
            )

        # stats
        if (epoch + 1) % args.n_stats == 0:
            stats['loss'].append(loss.item())
            interval.append(epoch+1)
            plot_stats(exp_dir, stats, interval)

        # visualize
        if (epoch + 1) % args.n_plot == 0:
            # reconstruction
            save_images(
                img=x_hat,
                path=os.path.join(exp_dir, f'{epoch+1:>04d}_recon.png'),
                nrow=int(np.sqrt(batch_size)),
            )

            # random sample
            if args.start == 'warm':
                # TODO, variable named `z_sampled`
                pass
            else:
                # cold
                # TODO, variable named `z_sampled`
                pass
            x_sampled = model(z_sampled)
            save_images(
                img=x_sampled,
                path=os.path.join(exp_dir, f'{epoch+1:>04d}_sampled.png'),
                nrow=int(np.sqrt(batch_size)),
            )

            # latent interpolation
            with torch.no_grad():
                z_std = z.std(0)
                save_interp(
                    model=model, args=args, z_std=z_std, device=device,
                    path=os.path.join(exp_dir, f'{epoch+1:>04d}_interp.png'),
                )


def main():
    args = parse_args()
    assert args.start in ['warm', 'cold']
    exp_dir = f'{args.start}_{args.lr}_{args.langevin_step_size}_{args.langevin_num_steps}_{args.mse_sigma}'
    os.makedirs(exp_dir, exist_ok=True)

    set_seed(args.seed)
    set_cuda()

    logger = get_logger(exp_dir)
    logger.info(args)
    run(args, exp_dir, logger)


if __name__ == '__main__':
    main()
