import numpy as np

def reset_gpu():
    from tensorflow.keras import backend as K
    import tensorflow as tf
    K.clear_session()
    tf.compat.v1.reset_default_graph()

def set_gpu(gpu_num=0):
    import tensorflow as tf
    import os
    #from tensorflow.keras.backend.tensorflow_backend import set_session
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    #os.environ["TF_CPP_MIN_LOG_LEVEL"]=2
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

def relabel(inputfile,inlabel,outlabel):
    import nibabel as nib
    import os
    img=nib.load(inputfile)
    data = np.squeeze(img.get_fdata())
    ori_label = np.array(inlabel)
    relabel = np.array(outlabel)
    for itr in range(len(ori_label)):
        loc = np.where((data>ori_label[itr]-0.5)&(data<ori_label[itr]+0.5))
        data[loc]=relabel[itr]
    #filename=img_list[i].split('/')[-1:][0].split('.nii.gz')[0]
    #filename=filename+'_e1.nii.gz'
    os.remove(inputfile)
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, inputfile)

def axfliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,:,::-1,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis],
                                 array[:,:,:,6,np.newaxis],array[:,:,:,5,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,:,::-1,:]
    return array

def cofliper(array,f=0):
    import numpy as np
    if f:
        array = array[:,::-1,:,:]
        array2 = np.concatenate((array[:,:,:,0,np.newaxis],array[:,:,:,2,np.newaxis],array[:,:,:,1,np.newaxis],
                                 array[:,:,:,4,np.newaxis],array[:,:,:,3,np.newaxis],
                                 array[:,:,:,6,np.newaxis],array[:,:,:,5,np.newaxis]),axis=-1)
        return array2
    else:
        array = array[:,::-1,:,:]
    return array

def crop_pad_ND(img, target_shape):
    import operator, numpy as np
    if (img.shape > np.array(target_shape)).any():
        target_shape2 = np.min([target_shape, img.shape],axis=0)
        start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2))
        end = tuple(map(operator.add, start, target_shape2))
        slices = tuple(map(slice, start, end))
        img = img[tuple(slices)]
    offset = tuple(map(lambda a, da: a//2-da//2, target_shape, img.shape))
    slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
    result = np.zeros(target_shape)
    result[tuple(slices)] = img
    return result

def get_maxshape(img_list):
    import nibabel as nib
    arrmax = [-1,-1,-1]
    for i in range(0,len(img_list)):
        img = nib.load(img_list[i])
        data = img.shape
        print(data)
        if data[0]>arrmax[0]:
            arrmax[0]=data[0]
        if data[1]>arrmax[1]:
            arrmax[1]=data[1]
        if data[2]>arrmax[2]:
            arrmax[2]=data[2]
    return arrmax

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

def argmax_sum(test_list, result_loc, ext='', *args):
    import numpy as np
    import nibabel as nib
    total = np.zeros(args[0].shape)
    temp = np.zeros(total.shape[1:])
    for item in args:
        if item.shape[0] != len(test_list):
            print('Error; Lengh mismatch error: args length: %d, test_list length: %d ' % (item.shape[0], len(test_list)))
            exit()
        for i in range(0,item.shape[0]):
            result_argmax = np.argmax(item[i],axis=-1)
            for i2 in range(0,item[i].shape[-1]):
                loc = np.where(result_argmax==i2)
                temp[...,i2][loc]=1
            total[i] = total[i] + temp
            temp[:]=0

    for i in range(0,len(test_list)):
        img = nib.load(test_list[i])
        new_img = nib.Nifti1Image(np.argmax(total[i],axis=-1), img.affine, img.header)
        filename=test_list[i].split('/')[-1:][0].split('.nii')[0]
        filename=filename+'_deep_argmax'+str(ext)+'.nii.gz'
        #nib.save(new_img, result_loc+str(filename))

def make_sum(axi_filter, cor_filter, sag_filter, input_name, result_loc,output_labels = np.array([1,2,3,4,5,6])):
    import nibabel as nib
    import sys, glob

    # 1-->axi 2-->cor 3-->sag
    axi_list = sorted(glob.glob(axi_filter))
    cor_list = sorted(glob.glob(cor_filter))
    sag_list = sorted(glob.glob(sag_filter))
    axi = nib.load(axi_list[0])
    cor = nib.load(cor_list[0])
    sag = nib.load(sag_list[0])

    bak = np.zeros(np.shape(axi.get_fdata()))
    left_in = np.zeros(np.shape(axi.get_fdata()))
    right_in = np.zeros(np.shape(axi.get_fdata()))    
    left_plate = np.zeros(np.shape(axi.get_fdata()))
    right_plate = np.zeros(np.shape(axi.get_fdata()))
    left_subplate = np.zeros(np.shape(axi.get_fdata()))
    right_subplate = np.zeros(np.shape(axi.get_fdata()))
    total = np.zeros(np.shape(axi.get_fdata()))

    for i in range(len(axi_list)):
        axi_data = nib.load(axi_list[i]).get_fdata()
        cor_data = nib.load(cor_list[i]).get_fdata()
        if len(sag_list) > i:
            sag_data = nib.load(sag_list[i]).get_fdata()

        loc = np.where(axi_data==0)
        bak[loc]=bak[loc]+1
        loc = np.where(cor_data==0)
        bak[loc]=bak[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==0)
            bak[loc]=bak[loc]+1

        loc = np.where(axi_data==1)
        left_in[loc]=left_in[loc]+1
        loc = np.where(cor_data==1)
        left_in[loc]=left_in[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==1)
            left_in[loc]=left_in[loc]+1

        loc = np.where(axi_data==2)
        right_in[loc]=right_in[loc]+1
        loc = np.where(cor_data==2)
        right_in[loc]=right_in[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==1)
            right_in[loc]=right_in[loc]+1

        loc = np.where(axi_data==3)
        left_subplate[loc]=left_subplate[loc]+1
        loc = np.where(cor_data==3)
        left_subplate[loc]=left_subplate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==2)
            left_subplate[loc]=left_subplate[loc]+1

        loc = np.where(axi_data==4)
        right_subplate[loc]=right_subplate[loc]+1
        loc = np.where(cor_data==4)
        right_subplate[loc]=right_subplate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==2)
            right_subplate[loc]=right_subplate[loc]+1

        loc = np.where(axi_data==5)
        left_plate[loc]=left_plate[loc]+1
        loc = np.where(cor_data==5)
        left_plate[loc]=left_plate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==3)
            left_plate[loc]=left_plate[loc]+1

        loc = np.where(axi_data==6)
        right_plate[loc]=right_plate[loc]+1
        loc = np.where(cor_data==6)
        right_plate[loc]=right_plate[loc]+1
        if len(sag_list) > i:
            loc = np.where(sag_data==3)
            right_plate[loc]=right_plate[loc]+1

    result = np.concatenate((bak[np.newaxis,:], left_in[np.newaxis,:], right_in[np.newaxis,:], left_subplate[np.newaxis,:], right_subplate[np.newaxis,:], left_plate[np.newaxis,:], right_plate[np.newaxis,:]),axis=0)
    result = np.argmax(result, axis=0)

    ori_label = np.array([1,2,3,4,5,6])
    relabel = np.array([160,161,160,161,42,1])
    for itr in range(len(ori_label)):
        loc = np.where((result>ori_label[itr]-0.5)&(result<ori_label[itr]+0.5))
        result[loc]=relabel[itr]

    filename=input_name.split('/')[-1:][0].split('.nii')[0]
    filename=filename+'_deep_final.nii.gz'
    new_img = nib.Nifti1Image(result, axi.affine, axi.header)
    nib.save(new_img, result_loc+'/'+filename)

def make_verify(input_path, label_path, result_loc):
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    import sys

    img = nib.load(input_path).get_fdata()
    #label_path = input_path.split('/')[-1:][0].split('.nii')[0]
    #label_path = label_path+'_deep_agg.nii.gz'
    label = nib.load(label_path).get_fdata()

    f,axarr = plt.subplots(3,3,figsize=(9,9))
    f.patch.set_facecolor('k')
    from matplotlib import colors

    f.text(0.4, 0.95, label_path, size="large", color="White")
    cmap = colors.ListedColormap(['None','blue', 'cyan','magenta','orangered','green','yellow'])
    bounds=[0,1,4,4.9,5.1,43,161,162]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    axarr[0,0].imshow(np.rot90(img[:,:,int(img.shape[-1]*0.4)]),cmap='gray')
    axarr[0,0].imshow(np.rot90(label[:,:,int(label.shape[-1]*0.4)]),alpha=0.3, interpolation='nearest' ,cmap=cmap,norm = norm)
    axarr[0,0].axis('off')

    axarr[0,1].imshow(np.rot90(img[:,:,int(img.shape[-1]*0.5)]),cmap='gray')
    axarr[0,1].imshow(np.rot90(label[:,:,int(label.shape[-1]*0.5)]),alpha=0.3, interpolation='nearest' ,cmap=cmap,norm = norm)
    axarr[0,1].axis('off')

    axarr[0,2].imshow(np.rot90(img[:,:,int(img.shape[-1]*0.6)]),cmap='gray')
    axarr[0,2].imshow(np.rot90(label[:,:,int(label.shape[-1]*0.6)]),alpha=0.3, interpolation='nearest' ,cmap=cmap,norm = norm)
    axarr[0,2].axis('off')

    axarr[1,0].imshow(np.rot90(img[:,int(img.shape[-2]*0.4),:]),cmap='gray')
    axarr[1,0].imshow(np.rot90(label[:,int(label.shape[-2]*0.4),:]),alpha=0.3, interpolation='nearest' ,cmap=cmap,norm = norm)
    axarr[1,0].axis('off')

    axarr[1,1].imshow(np.rot90(img[:,int(img.shape[-2]*0.5),:]),cmap='gray')
    axarr[1,1].imshow(np.rot90(label[:,int(label.shape[-2]*0.5),:]),alpha=0.3, interpolation='nearest' ,cmap=cmap,norm = norm)
    axarr[1,1].axis('off')

    axarr[1,2].imshow(np.rot90(img[:,int(img.shape[-2]*0.6),:]),cmap='gray')
    axarr[1,2].imshow(np.rot90(label[:,int(label.shape[-2]*0.6),:]),alpha=0.3, interpolation='nearest' ,cmap=cmap,norm = norm)
    axarr[1,2].axis('off')

    axarr[2,0].imshow(np.rot90(img[int(img.shape[0]*0.4),:,:]),cmap='gray')
    axarr[2,0].imshow(np.rot90(label[int(label.shape[0]*0.4),:,:]),alpha=0.3, interpolation='nearest' ,cmap=cmap,norm = norm)
    axarr[2,0].axis('off')

    axarr[2,1].imshow(np.rot90(img[int(img.shape[0]*0.5),:,:]),cmap='gray')
    axarr[2,1].imshow(np.rot90(label[int(label.shape[0]*0.5),:,:]),alpha=0.3, interpolation='nearest' ,cmap=cmap,norm = norm)
    axarr[2,1].axis('off')

    axarr[2,2].imshow(np.rot90(img[int(img.shape[0]*0.6),:,:]),cmap='gray')
    axarr[2,2].imshow(np.rot90(label[int(label.shape[0]*0.6),:,:]),alpha=0.3, interpolation='nearest' ,cmap=cmap,norm = norm)
    axarr[2,2].axis('off')
    f.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(result_loc+'/'+label_path.split('/')[-1].split('.nii')[0]+'_verify.png', facecolor=f.get_facecolor())
    return 0

def make_callbacks(weight_name, history_name, monitor='val_loss', patience=100, mode='min', save_weights_only=True):
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    import six, io, time, csv, numpy as np, json, warnings
    from collections import deque
    from collections import OrderedDict
    from collections import Iterable
    from collections import defaultdict
    from tensorflow import keras
    from tensorflow.keras.utils import Progbar
    from tensorflow.keras import backend as K
    from tensorflow.python.keras.engine.training_utils_v1 import standardize_input_data
    class CSVLogger_time(Callback):
        """Callback that streams epoch results to a csv file.
        Supports all values that can be represented as a string,
        including 1D iterables such as np.ndarray.
        # Example
        ```python
        csv_logger = CSVLogger('training.log')
        model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```
        # Arguments
            filename: filename of the csv file, e.g. 'run/log.csv'.
            separator: string used to separate elements in the csv file.
            append: True: append if file exists (useful for continuing
                training). False: overwrite existing file,
        """

        def __init__(self, filename, separator=',', append=False):
            self.sep = separator
            self.filename = filename
            self.append = append
            self.writer = None
            self.keys = None
            self.append_header = True
            if six.PY2:
                self.file_flags = 'b'
                self._open_args = {}
            else:
                self.file_flags = ''
                self._open_args = {'newline': '\n'}
            super(CSVLogger_time, self).__init__()

        def on_train_begin(self, logs=None):
            if self.append:
                if os.path.exists(self.filename):
                    with open(self.filename, 'r' + self.file_flags) as f:
                        self.append_header = not bool(len(f.readline()))
                mode = 'a'
            else:
                mode = 'w'
            self.csv_file = io.open(self.filename,
                                    mode + self.file_flags,
                                    **self._open_args)

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_time_start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, six.string_types):
                    return k
                elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                    return '"[%s]"' % (', '.join(map(str, k)))
                else:
                    return k

            if self.keys is None:
                self.keys = sorted(logs.keys())

            if self.model.stop_training:
                # We set NA so that csv parsers do not fail for this last epoch.
                logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

            if not self.writer:
                class CustomDialect(csv.excel):
                    delimiter = self.sep
                fieldnames = ['epoch'] + self.keys +['time']
                if six.PY2:
                    fieldnames = [unicode(x) for x in fieldnames]
                self.writer = csv.DictWriter(self.csv_file,
                                             fieldnames=fieldnames,
                                             dialect=CustomDialect)
                if self.append_header:
                    self.writer.writeheader()

            row_dict = OrderedDict({'epoch': epoch})
            logs['time']=time.time() - self.epoch_time_start
            self.keys.append('time')
            row_dict.update((key, handle_value(logs[key])) for key in self.keys)
            self.writer.writerow(row_dict)
            self.csv_file.flush()

        def on_train_end(self, logs=None):
            self.csv_file.close()
            self.writer = None

        def __del__(self):
            if hasattr(self, 'csv_file') and not self.csv_file.closed:
                self.csv_file.close()
    earlystop=EarlyStopping(monitor=monitor, patience=patience, verbose=0, mode=mode)
    checkpoint=ModelCheckpoint(filepath=weight_name, monitor=monitor, mode=mode, save_best_only=True, save_weights_only=save_weights_only, verbose=0)
    csvlog=CSVLogger_time(history_name, separator='\t')
    return [earlystop, checkpoint, csvlog]
