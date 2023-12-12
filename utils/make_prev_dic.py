from resize import crop_pad_ND
from flipers import *

def make_dic_prev(img_list, gold_list, input_size, dim, flip=0,max_shape=[-1,-1,-1],out_ch = 7):
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
            input_size = [*input_size, max_shape[2]]
        elif dim == 'cor':
            input_size = [input_size[0], max_shape[1], input_size[1]]
        elif dim == 'sag':
            input_size = [max_shape[0], *input_size]
        else:
            print('available: axi, cor, sag.   Your: '+dim)
            exit()

        img = crop_pad_ND(img, input_size)
        label = crop_pad_ND(label, input_size)
        return img, label
    if dim == 'axi':
        dic = np.zeros([max_shape[2]*len(img_list), input_size[0], input_size[1], 1], dtype=np.float16)
        seg = np.zeros([max_shape[2]*len(img_list), input_size[0], input_size[1], out_ch], dtype=np.float16)
    elif dim == 'cor':
        dic = np.zeros([max_shape[1]*len(img_list), input_size[0], input_size[1], 1], dtype=np.float16)
        seg = np.zeros([max_shape[1]*len(img_list), input_size[0], input_size[1], out_ch], dtype=np.float16)
    elif dim == 'sag':
        dic = np.zeros([max_shape[0]*len(img_list), input_size[0], input_size[1], 1], dtype=np.float16)
        seg = np.zeros([max_shape[0]*len(img_list), input_size[0], input_size[1], out_ch], dtype=np.float16)
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
            left_in_loc = np.where((img2>160.5)&(img2<161.5))
            right_in_loc = np.where((img2>159.5)&(img2<160.5)) 
            left_subplate_loc = np.where((img2>4.5)&(img2<5.5))
            right_subplate_loc = np.where((img2>3.5)&(img2<4.5))
            img3[back_loc]=1
        elif dim == 'sag' and out_ch==4:
            img3 = np.zeros_like(img2)
            back_loc = np.where(img<0.5)
            plate_loc = np.where(((img2>0.5)&(img2<1.5))|((img2>41.5)&(img2<42.5)))
            in_loc = np.where(((img2>160.5)&(img2<161.5))|((img2>159.5)&(img2<160.5)))
            subplate_loc = np.where(((img2>3.5)&(img2<4.5))|((img2>4.5)&(img2<5.5)))
            img3[back_loc]=1
        elif dim == 'sag' and out_ch==7:
            img3 = np.zeros_like(img2)
            back_loc = np.where(img2<0.5)
            left_plate_loc = np.where((img2>0.5)&(img2<1.5))
            right_plate_loc = np.where((img2>41.5)&(img2<42.5))
            left_in_loc = np.where((img2>160.5)&(img2<161.5))
            right_in_loc = np.where((img2>159.5)&(img2<160.5)) 
            left_subplate_loc = np.where((img2>4.5)&(img2<5.5))
            right_subplate_loc = np.where((img2>3.5)&(img2<4.5))
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
        elif dim == 'sag' and out_ch==4:
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
        elif dim == 'sag' and out_ch==7:
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,0]=img3
            img3[:]=0
            img3[left_in_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,1]=img3
            img3[:]=0
            img3[right_in_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,2]=img3
            img3[:]=0
            img3[left_subplate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,3]=img3
            img3[:]=0
            img3[right_subplate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,4]=img3
            img3[:]=0
            img3[left_plate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,5]=img3
            img3[:]=0
            img3[right_plate_loc]=1
            seg[max_shape[0]*i:max_shape[0]*(i+1),:,:,6]=img3
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
    print(dic.dtype)
    print(seg.dtype)
    return dic, seg