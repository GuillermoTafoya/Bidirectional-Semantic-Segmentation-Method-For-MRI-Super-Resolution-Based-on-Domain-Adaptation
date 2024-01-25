

"""Console script for fetal_brain_segmentation."""
import argparse, tempfile
import sys
import numpy as np
import medpy
from medpy.io import load, save
import glob, os, time, pickle
#sys.path.append(os.path.dirname(__file__))

from model.unet.unet_network import *
from model.unet.utils import *
import tensorflow as tf
#from model.utils.deep_util_sp import *


def make_dic(img_list, gold_list, input_size, dim, flip=0,max_shape=[-1,-1,-1]):
    import numpy as np
    import nibabel as nib
    from tqdm import tqdm
    def get_data(img, label, input_size, dim):
        import nibabel as nib
        img = np.squeeze(nib.load(img).get_fdata())
        img = img - img.min()
        img = img / img.max()
        img = img * 255
        label = np.squeeze(nib.load(label).get_fdata())
        if dim == 'axi':
            input_size = [*input_size, img.shape[2]]
        elif dim == 'cor':
            input_size = [input_size[0], img.shape[1], input_size[1]]
        elif dim == 'sag':
            input_size = [img.shape[0], *input_size]
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()

        img = crop_pad_ND(img, input_size)
        label = crop_pad_ND(label, input_size)
        return img, label

    #max_shape = [117,159,126]
    if dim == 'axi':
        dic = np.zeros([max_shape[2]*len(img_list), input_size[0], input_size[1], 1], dtype=int)
        seg = np.zeros([max_shape[2]*len(img_list), input_size[0], input_size[1], 7], dtype=np.float32)
    elif dim == 'cor':
        dic = np.zeros([max_shape[1]*len(img_list), input_size[0], input_size[1], 1], dtype=int)
        seg = np.zeros([max_shape[1]*len(img_list), input_size[0], input_size[1], 7], dtype=np.float32)
    elif dim == 'sag':
        dic = np.zeros([max_shape[0]*len(img_list), input_size[0], input_size[1], 1], dtype=int)
        seg = np.zeros([max_shape[0]*len(img_list), input_size[0], input_size[1], 4], dtype=np.float32)
    else:
        print('available: axi, cor, sag.   Your: '+dim)
        exit()

    for i in tqdm(range(0, len(img_list)),desc=dim+' dic making..'):
        if dim == 'axi':
            img, label = get_data(img_list[i], gold_list[i], input_size, 'axi')
            dic[max_shape[2]*i:max_shape[2]*(i+1),:,:,0]= np.swapaxes(img,2,0)
            img2 = np.swapaxes(label,2,0)
        elif dim == 'cor':
            img, label = get_data(img_list[i], gold_list[i], input_size, 'cor')
            dic[max_shape[1]*i:max_shape[1]*(i+1),:,:,0]= np.swapaxes(img,1,0)
            img2 = np.swapaxes(label,1,0)
        elif dim == 'sag':
            img, label = get_data(img_list[i], gold_list[i], input_size, 'sag')
            dic[max_shape[0]*i:max_shape[0]*(i+1),:,:,0]= img
            img2 = label
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
        if (dim == 'axi') | (dim == 'cor'):
            img3 = np.zeros_like(img2)
            back_loc = np.where(img2<0.5)
            left_plate_loc = np.where((img2>0.5)&(img2<1.5))
            right_plate_loc = np.where((img2>41.5)&(img2<42.5))
            left_subplate_loc = np.where((img2>160.5)&(img2<161.5))
            right_subplate_loc = np.where((img2>159.5)&(img2<160.5)) 
            left_in_loc = np.where((img2>3.5)&(img2<4.5))
            right_in_loc = np.where((img2>45.5)&(img2<46.5))
            img3[back_loc]=1
        elif dim == 'sag':
            img3 = np.zeros_like(img2)
            back_loc = np.where(img<0.5)
            in_loc = np.where(((img2>3.5)&(img2<4.5))|((img2>45.5)&(img2<46.5)))
            plate_loc = np.where(((img2>0.5)&(img2<1.5))|((img2>41.5)&(img2<42.5)))
            subplate_loc = np.where(((img2>160.5)&(img2<161.5))|((img2>159.5)&(img2<160.5)))
            img3[back_loc]=1
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()

        if dim == 'axi':
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[left_in_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[right_in_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[left_subplate_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,3]=img3
            img3[:]=0
            img3[right_subplate_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,4]=img3
            img3[:]=0
            img3[left_plate_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,5]=img3
            img3[:]=0
            img3[right_plate_loc]=1
            seg[max_shape[2]*i:max_shape[2]*(i+1),:,:,6]=img3
            img3[:]=0            
        elif dim == 'cor':
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[left_in_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[right_in_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[left_subplate_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,3]=img3
            img3[:]=0
            img3[right_subplate_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,4]=img3
            img3[:]=0
            img3[left_plate_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,5]=img3
            img3[:]=0
            img3[right_plate_loc]=1
            seg[max_shape[1]*i:max_shape[1]*(i+1),:,:,6]=img3
            img3[:]=0
        elif dim == 'sag':
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[in_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[subplate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[plate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,3]=img3
            img3[:]=0
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
    if flip:
        if dim == 'axi':
            dic=np.concatenate((dic,dic[:,::-1,:,:]),axis=0)
            seg=np.concatenate((seg,seg[:,::-1,:,:]),axis=0)
            dic=np.concatenate((dic, axfliper(dic)),axis=0)
            seg=np.concatenate((seg, axfliper(seg, 1)),axis=0)
        elif dim == 'cor':
            dic=np.concatenate((dic,dic[:,:,::-1,:]),axis=0)
            seg=np.concatenate((seg,seg[:,:,::-1,:]),axis=0)
            dic=np.concatenate((dic, cofliper(dic)),axis=0)
            seg=np.concatenate((seg, cofliper(seg, 1)),axis=0)
        elif dim == 'sag':
            dic = np.concatenate((dic, dic[:,:,::-1,:], dic[:,::-1,:,:]),axis=0)
            seg = np.concatenate((seg, seg[:,:,::-1,:], seg[:,::-1,:,:]),axis=0)
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()

    return dic, seg


def make_result(output, img_list, result_loc, axis, ext=''):
    import nibabel as nib
    import numpy as np, ipdb

    if type(img_list) != np.ndarray: 
        print('\'img_list\' must be list')
        exit()
    for i2 in range(len(img_list)):
        print('filename : '+img_list[i2])
        img = nib.load(img_list[i2])
        img_data = np.squeeze(img.get_fdata())
        pr4=output[i2*(int(output.shape[0]/len(img_list))):(i2+1)*(int(output.shape[0]/len(img_list)))]
        if axis == 'axi':
            pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(int),0,2)
            pr4=crop_pad_ND(pr4, img.shape)
        elif axis == 'cor':
            pr4=np.swapaxes(np.argmax(pr4,axis=3).astype(int),0,1)
            pr4=crop_pad_ND(pr4, img.shape)
        elif axis == 'sag':
            pr4=np.argmax(pr4,axis=3).astype(int)
            pr4=crop_pad_ND(pr4, img.shape)
        else:
            print('available: axi, cor, sag.   Your: '+axis)
            exit()

        img_data[:] = 0
        img_data=pr4
        new_img = nib.Nifti1Image(img_data, img.affine, img.header)
        filename=img_list[i2].split('/')[-1:][0].split('.nii')[0]
        if axis== 'axi':
            filename=filename+'_deep_axi'+ext+'.nii.gz'
        elif axis== 'cor':
            filename=filename+'_deep_cor'+ext+'.nii.gz'
        elif axis== 'sag':
            filename=filename+'_deep_sag'+ext+'.nii.gz'
        else:
            print('available: axi, cor, sag.   Your: '+axis)
            exit()
        print('save result : '+result_loc+filename)
        nib.save(new_img, result_loc+str(filename))

    return 1

def return_shape(test_list, predic_array, dim):
    import nibabel as nib
    import numpy as np

    img = nib.load(test_list[0])
    img_data = np.squeeze(img.get_fdata())
    output = np.zeros([len(test_list),*img_data.shape,7])

    for i2 in range(len(test_list)):
        predic=predic_array[i2*(int(predic_array.shape[0]/len(test_list))):(i2+1)*(int(predic_array.shape[0]/len(test_list)))]
        if dim == 'axi':
            predic=np.swapaxes(predic,0,2)
            predic=crop_pad_ND(predic, (*img.shape,predic.shape[-1]))
        elif dim == 'cor':
            predic=np.swapaxes(predic,0,1)
            predic=crop_pad_ND(predic, (*img.shape,predic.shape[-1]))
        elif dim == 'sag':
            predic=crop_pad_ND(predic, (*img.shape,predic.shape[-1]))
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()
        output[i2]=predic
    return output


def main():
    parser = argparse.ArgumentParser('   ==========   Testing ==========   ')
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

    print(f'{test_dic.shape=}')



    print("Shape of the output feature map:", deepest_layer_features.shape)

    print("Sample output from the first channel in the batch:")
    print(deepest_layer_features[0, :, :, :1])  


    import matplotlib.pyplot as plt

    feature_maps = deepest_layer_features[0]

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
