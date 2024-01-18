

"""Console script for fetal_brain_segmentation."""
import argparse, tempfile
import sys
import numpy as np
import medpy
from medpy.io import load, save
import glob, os, time, pickle
#sys.path.append(os.path.dirname(__file__))

from model.unet.unet_network import *
import tensorflow as tf
#from model.utils.deep_util_sp import *


def main():
    parser = argparse.ArgumentParser('   ==========   Fetal U_Net segmentation script made by Marisol Lemus (November 16, 2021 ver.1)   ==========   ')
    parser.add_argument('-input', '--input_loc',action='store',dest='inp',type=str, required=True, help='input MR folder name for training')
    parser.add_argument('-output', '--output_loc',action='store',dest='out',type=str, required=True, help='Output path')
    parser.add_argument('-gpu', '--gpu_number',action='store',dest='gpu',type=str, default='-1',help='Select GPU')
    parser.add_argument('-mr', '--merge', dest='merge', action='store_false',help='merge subplate with inner')
    parser.add_argument('-cp_seg', action='store', dest='cp_seg',help='use cp segmentation to define the CP on SP segmentation')

    curent_path = os.getcwd()
    result_loc = os.path.join(os.getcwd()+'result/conv_style/')
    weight_loc = os.path.join(os.getcwd() +'./')
    axi = ('./model/unet/weights/fold0axi.h5')
    cor =  ('./model/unet/weights/fold0cor.h5')
    sag =  ('./model/unet/weights/fold0sag.h5')
    hist_loc = os.path.join(os.getcwd() + 'history/')
    isize = [192, 192]
    #style = 'basic'
    coef='dice_coef'

    ilabel = [1,2,5,6,4,3]
    olabel = [161,160,1,42,4,5]
    #merge = False
    iaxis = [7,7,4]
    
   
    args = parser.parse_args()
    #print("Arguments: " + str(args._))
    
    if len(sys.argv) < 2:
        parser.print_usage()
        exit()

    if os.path.isdir(args.inp):
        print("Is a directory")
        img_list = np.asarray(sorted(glob.glob(args.inp+'/*nuc.nii*')))
    elif os.path.isfile(args.inp):
        print("Is a file")
        img_list = np.asarray(sorted(glob.glob(args.inp)))
    else:
        print("Is a leprechaun")
        img_list = np.asarray(sorted(glob.glob(args.inp)))
    
    if len(img_list)==0:
        print('No such file or directory')
        exit()

    set_gpu(args.gpu)
    max_shape = get_maxshape(img_list)

   
    if args.cp_seg: 
       cp_seg=(os.path.join(os.getcwd()+'/segmentation_to31_final.nii'))
    
    test_dic, _ = make_dic(img_list, img_list, isize, 'axi',max_shape=max_shape)

    #print(f'{test_dic.shape=})
    print(f'{test_dic.shape=})')

    test_dic = test_dic[0]
    print(f'{test_dic.shape=})')
    test_dic = test_dic[None, :]

    print(f'{test_dic.shape=})')

    

    model = Unet_network([*isize,1], iaxis[0], metrics=coef).build()
    
    
    
    callbacks=make_callbacks(axi, hist_loc+'/fold'+str(0)+'axi.tsv')
    model.load_weights(axi)

    model.summary()
    
    #predic_axi = model.predict(test_dic, batch_size=30)
    #predic_axif1 = model.predict(test_dic[:,::-1,:,:], batch_size=30)
    #predic_axif1 = predic_axif1[:,::-1,:,:]
    #predic_axif2 = model.predict(axfliper(test_dic), batch_size=30)
    #predic_axif2 = axfliper(predic_axif2,1)
    #predic_axif3 = model.predict(axfliper(test_dic[:,::-1,:,:]), batch_size=30)
    #predic_axif3 = axfliper(predic_axif3[:,::-1,:,:],1)
    
    # Create a new model that outputs the deepest layer
    deepest_layer_output = model.get_layer('activation_27').output
    model_for_deepest_layer = tf.keras.Model(inputs=model.input, outputs=deepest_layer_output)


    deepest_layer_features = model_for_deepest_layer.predict(test_dic, batch_size=1)



    print("Shape of the output feature map:", deepest_layer_features.shape)

    print("Sample output from the first image in the batch:")
    print(deepest_layer_features[0, :, :, :1])  


    import matplotlib.pyplot as plt

    feature_maps = deepest_layer_features[0]  # first image in the batch

    # Let's visualize and save the first 5 feature maps
    for i in range(1):
        plt.figure(figsize=(5, 5))
        plt.imshow(feature_maps[:, :, i], cmap='gray')  
        plt.title(f'Feature Map {i+1}')
        plt.axis('off')
        plt.savefig(f'{args.out}/feature_map_{i+1}.png')
        plt.close()
    
    del model, test_dic, callbacks
    reset_gpu()
    

    return 0          


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
