from torchstat import stat
# from thop import profile
# from thop import clever_format

import torch
import torch.backends.cudnn as cudnn

import warnings
import argparse
from utils import get_network


def main():
    warnings.filterwarnings('ignore')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='efficientnetb7', help='net type')
    args = parser.parse_args()
    args.gpu = False

    model = get_network(args)

    stat(model, (3,224,224))

    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, inputs=(input, ))
    # flops, params = clever_format([flops, params], "%.3f")
    # print('flops:', flops)
    # print('params:', params)
    # print()

    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: %.2fM' % (total/1e6))

if __name__ == '__main__':
    main()