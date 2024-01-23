from train import Trainer
import argparse
import os
import torch

def config():
    parser = argparse.ArgumentParser('   ==========  Training Script   ==========   ')
    parser.add_argument('-v', '--view',action='store',dest='view',type=str, required=False, default='L', help='View for training model')
    parser.add_argument('-p', '--path',action='store',dest='path',type=str, required=False, default='./Data/', help='Path for extracting data')
    parser.add_argument('-g', '--gpu',action='store',dest='gpu',type=str, required=False, default='0', help='Gpu')
    parser.add_argument('-b', '--batch',action='store',dest='batch',type=int, required=False, default=32,help='Batch number')
    parser.add_argument('-e', '--epochs',action='store',dest='epochs',type=int, required=False, default=500,help='Epochs')
    return parser

print('Loading parser')

parser = config()
args = parser.parse_args()


print()
print('Loading gpu')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

print()
print('Loading trainer')

JUAN = Trainer('./Data/', args.view, device, args.batch, args.gpu)
JUAN.train(args.epochs)