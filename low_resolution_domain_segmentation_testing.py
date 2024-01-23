

"""Console script for fetal_brain_segmentation."""
import argparse, tempfile
import sys
import numpy as np
import medpy
from medpy.io import load, save
import glob, os, time, pickle
#sys.path.append(os.path.dirname(__file__))

from model.unet.unet_network import *
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
        img_list = np.asarray(sorted(glob.glob(args.inp+'/*nuc.nii*')))
    elif os.path.isfile(args.inp):
        img_list = np.asarray(sorted(glob.glob(args.inp)))
    else:
        img_list = np.asarray(sorted(glob.glob(args.inp)))
    
    if len(img_list)==0:
        print('No such file or directory')
        exit()

    set_gpu(args.gpu)
    max_shape = get_maxshape(img_list)

   
    if args.cp_seg: 
       cp_seg=(os.path.join(os.getcwd()+'/segmentation_to31_final.nii'))
    
    test_dic, _ = make_dic(img_list, img_list, isize, 'axi',max_shape=max_shape)
    model = Unet_network([*isize,1], iaxis[0], metrics=coef).build()
    
    
    
    callbacks=make_callbacks(axi, hist_loc+'/fold'+str(0)+'axi.tsv')
    model.load_weights(axi)
    predic_axi = model.predict(test_dic, batch_size=30)
    predic_axif1 = model.predict(test_dic[:,::-1,:,:], batch_size=30)
    predic_axif1 = predic_axif1[:,::-1,:,:]
    predic_axif2 = model.predict(axfliper(test_dic), batch_size=30)
    predic_axif2 = axfliper(predic_axif2,1)
    predic_axif3 = model.predict(axfliper(test_dic[:,::-1,:,:]), batch_size=30)
    predic_axif3 = axfliper(predic_axif3[:,::-1,:,:],1)
    del model, test_dic, callbacks
    reset_gpu()

    test_dic, _ =make_dic(img_list, img_list, isize, 'cor',max_shape=max_shape)
    model = Unet_network([*isize,1], iaxis[1], metrics=coef).build()
    callbacks=make_callbacks(cor, hist_loc+'/fold'+str(0)+'cor.tsv')
    model.load_weights(cor)

    predic_cor = model.predict(test_dic, batch_size=30)
    predic_corf1 = model.predict(test_dic[:,:,::-1,:], batch_size=30)
    predic_corf1 = predic_corf1[:,:,::-1,:]
    predic_corf2 = model.predict(cofliper(test_dic), batch_size=30)
    predic_corf2 = cofliper(predic_corf2,1)
    predic_corf3 = model.predict(cofliper(test_dic[:,:,::-1,:]), batch_size=30)
    predic_corf3 = cofliper(predic_corf3[:,:,::-1,:],1)

    del model, test_dic, callbacks
    reset_gpu()


    test_dic, _ =make_dic(img_list, img_list, isize, 'sag',max_shape=max_shape)
    model = Unet_network([*isize, 1], iaxis[2], metrics=coef).build()
    callbacks=make_callbacks(sag, hist_loc+'/fold'+str(0)+'sag.tsv')
    model.load_weights(sag)
    predic_sag = model.predict(test_dic, batch_size=30)
    predic_sagf1 = model.predict(test_dic[:,::-1,:,:], batch_size=30)
    predic_sagf1 = predic_sagf1[:,::-1,:,:]
    predic_sagf2 = model.predict(test_dic[:,:,::-1,:], batch_size=30)
    predic_sagf2 = predic_sagf2[:,:,::-1,:]
    del model, test_dic, callbacks
    reset_gpu()

    predic_sag = np.stack((predic_sag[...,0], predic_sag[...,1], predic_sag[...,1],
                                predic_sag[...,2], predic_sag[...,2], predic_sag[...,3],
                                predic_sag[...,3]),axis=-1)
    predic_sagf1 = np.stack((predic_sagf1[...,0], predic_sagf1[...,1], predic_sagf1[...,1],
                                predic_sagf1[...,2], predic_sagf1[...,2], predic_sagf1[...,3],
                                predic_sagf1[...,3]),axis=-1)
    predic_sagf2 = np.stack((predic_sagf2[...,0], predic_sagf2[...,1], predic_sagf2[...,1],
                                predic_sagf2[...,2], predic_sagf2[...,2], predic_sagf2[...,3],
                                predic_sagf2[...,3]),axis=-1)

    import nibabel as nib
    predic_axi = return_shape(img_list, predic_axi, 'axi')
    predic_axif1 = return_shape(img_list, predic_axif1, 'axi')
    predic_axif2 = return_shape(img_list, predic_axif2, 'axi')
    predic_axif3 = return_shape(img_list, predic_axif3, 'axi')
    predic_cor = return_shape(img_list, predic_cor, 'cor')
    predic_corf1 = return_shape(img_list, predic_corf1, 'cor')
    predic_corf2 = return_shape(img_list, predic_corf2, 'cor')
    predic_corf3 = return_shape(img_list, predic_corf3, 'cor')
    predic_sag = return_shape(img_list, predic_sag, 'sag')
    predic_sagf1 = return_shape(img_list, predic_sagf1, 'sag')
    predic_sagf2 = return_shape(img_list, predic_sagf2, 'sag')
    argmax_sum(img_list, result_loc, '', predic_axi, predic_axif1, predic_axif2, predic_axif3, predic_cor, predic_corf1, predic_corf2, predic_corf3, predic_sag, predic_sagf1, predic_sagf2)
    predic_final = predic_axi+predic_axif1+predic_axif2+predic_axif3+predic_cor+predic_corf1+predic_corf2+predic_corf3+predic_sag+predic_sagf1+predic_sagf2

    
 
    if np.shape(img_list):
        
       for i in range(0,len(img_list)):
            img = nib.load(img_list[i])
            new_img = nib.Nifti1Image(np.argmax(predic_final[i],axis=-1), img.affine, img.header)
            filename=img_list[i].split('/')[-1:][0].split('.nii')[0]
            filename_complete=filename+'_deep_agg_sp.nii.gz'
            savedloc = args.out+str(filename_complete)
            nib.save(new_img, savedloc)
            relabel(savedloc,ilabel,olabel)
            if args.merge:
                filename_complete=filename+'_deep_agg_merged.nii.gz'
                savedloc = args.out+str(filename_complete)
                nib.save(new_img, savedloc)
                relabel(savedloc,[1,2,5,6,4,3,4,5],[161,160,1,42,4,5,161,160])
            make_verify(img_list[i],savedloc,args.out)  
            if args.cp_seg:   
                recon_cp, header =medpy.io.load(cp_seg)
                recon_sp, header =medpy.io.load(filename_complete)
                
                for i in range(recon_sp.shape[0]):
                    for j in range(recon_sp.shape[1]):
                        for k in range (recon_sp.shape[2]):
                            if recon_cp[i,j,k]==1.0 and recon_sp[i,j,k]!=1.0:
                                recon_sp[i,j,k]=1.0 
                            elif recon_cp[i,j,k]==42.0 and recon_sp[i,j,k]!=42.0:
                                recon_sp[i,j,k]=42.0   
                            elif recon_cp[i,j,k]==0.0 and recon_sp[i,j,k]!=0.0:
                                recon_sp[i,j,k]=0.0                                
                            elif recon_sp[i,j,k]==5.0 and recon_cp[i,j,k]==0.0:
                                recon_sp[i,j,k]=0.0
                            elif recon_cp[i,j,k]==160.0 and recon_sp[i,j,k]==161.0:
                                recon_sp[i,j,k]=160.0
                            elif recon_cp[i,j,k]==161.0 and recon_sp[i,j,k]==160.0:
                                recon_sp[i,j,k]=161.0   
                            elif recon_cp[i,j,k]==160.0 and recon_sp[i,j,k]!=160.0:
                                recon_sp[i,j,k]=4.0
                            elif recon_cp[i,j,k]==161.0 and recon_sp[i,j,k]!=161.0:
                                recon_sp[i,j,k]=5.0                      
                            elif recon_sp[i,j,k]==42.0 and recon_cp[i,j,k]==0.0:
                                recon_sp[i,j,k]=42.0   
                            elif recon_sp[i,j,k]==4.0 and recon_cp[i,j,k]==0:
                                recon_sp[i,j,k]=0.0
                            elif recon_sp[i,j,k]==4.0 and recon_cp[i,j,k]==160.0:
                                recisizecon_sp[i,j,k]=161.0 
                                
                            else:
                                recon_sp[i,j,k]= recon_sp[i,j,k]
                
                new_sp = nib.Nifti1Image(recon_sp,img.affine, img.header)
                savedloc=args.out+filename+'_deep_agg.nii.gz'
                nib.save(new_sp, savedloc)       
                
    else:
        img = nib.load(img_list)
        new_img = nib.Nifti1Image(np.argmax(predic_final,axis=-1), img.affine, img.header)
        filename=img_list.split('/')[-1:][0].split('.nii')[0]
        savedloc=args.out+filename+'_deep_agg.nii.gz'
        nib.save(new_img, savedloc)
        relabel(savedloc,ilabel,olabel)
       
    return 0          


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
