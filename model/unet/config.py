import argparse

def config():
    parser = argparse.ArgumentParser('   ==========   Fetal U_Net segmentation script made by Marisol Lemus (November 16, 2021 ver.1)   ==========   ')
    parser.add_argument('-input', '--input_loc',action='store',dest='inp',type=str, required=True, help='input MR folder name for training')
    parser.add_argument('-output', '--output_loc',action='store',dest='out',type=str, required=True, help='Output path')
    parser.add_argument('-gpu', '--gpu_number',action='store',dest='gpu',type=str, default='-1',help='Select GPU')
    parser.add_argument('-mr', '--merge', dest='merge', action='store_false',help='merge subplate with inner')
    parser.add_argument('-cp_seg', action='store', dest='cp_seg',help='use cp segmentation to define the CP on SP segmentation')
    return parser