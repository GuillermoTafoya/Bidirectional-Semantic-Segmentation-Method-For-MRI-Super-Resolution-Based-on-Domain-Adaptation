class Unet_network:
    '''
    This class has all the methods needed for training the model.
    '''
    
    def __init__(self, input_shape, out_ch, loss='dice_loss', metrics=['dice_coef', 'dis_dice_coef'], style='basic', ite=3, depth=4, dim=32, weights='', init='he_normal',acti='elu',lr=1e-4):
        from tensorflow.keras.layers import Input
        self.style=style
        self.input_shape=input_shape
        self.out_ch=out_ch
        self.loss = loss
        self.metrics = metrics
        self.ite=ite
        self.depth=depth
        self.dim=dim
        self.init=init
        self.acti=acti
        self.weight=weights
        self.lr=lr
        self.I = Input(input_shape)
        self.ratio = None

    def conv_block(self,inp,dim):
        from tensorflow.keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2D(dim, (3,3), padding='same', kernel_initializer=self.init)(x)
        return x

    def conv1_block(self,inp,dim):
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2D(dim, (1,1), padding='same', kernel_initializer=self.init)(x)
    def tconv_block(self,inp,dim):
        from tensorflow.keras.layers import BatchNormalization as bn, Activation, Conv2DTranspose, Dropout
        x = bn()(inp)
        x = Activation(self.acti)(x)
        x = Conv2DTranspose(dim, 2, strides=2, padding='same', kernel_initializer=self.init)(x)
        return x

    def basic_block(self, inp, dim):
        for i in range(self.ite):
            inp = self.conv_block(inp,dim)
        return inp

    def res_block(self, inp, dim):
        from tensorflow.keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout, Add
        inp2 = inp
        for i in range(self.ite):
            inp = self.conv_block(inp,dim)
        cb2 = self.conv1_block(inp2,dim)
        return Add()([inp, cb2])

    def dense_block(self, inp, dim):
        from tensorflow.keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout, concatenate
        for i in range(self.ite):
            cb = self.conv_block(inp, dim)
            inp = concatenate([inp,cb])
        inp = self.conv1_block(inp,dim)
        return inp

    def RCL_block(self, inp, dim):
        from tensorflow.keras.layers import BatchNormalization as bn, Activation, Conv2D, Dropout, Add
        RCL=Conv2D(dim, (3,3), padding='same',kernel_initializer=self.init)
        conv=bn()(inp)
        conv=Activation(self.acti)(conv)
        conv=Conv2D(dim,(3,3),padding='same',kernel_initializer=self.init)(conv)
        conv2=bn()(conv)
        conv2=Activation(self.acti)(conv2)
        conv2=RCL(conv2)
        conv2=Add()([conv,conv2])
        for i in range(0, self.ite-2):
            conv2=bn()(conv2)
            conv2=Activation(self.acti)(conv2)
            conv2=Conv2D(dim, (3,3), padding='same',weights=RCL.get_weights())(conv2)
            conv2=Add()([conv,conv2])
        return conv2

    def build_U(self, inp, dim, depth):
        from tensorflow.keras.layers import MaxPooling2D, concatenate, Dropout
        if depth > 0:
            if self.style == 'basic':
                x = self.basic_block(inp, dim)
            elif self.style == 'res':
                x = self.res_block(inp, dim)
            elif self.style == 'dense':
                x = self.dense_block(inp, dim)
            elif self.style == 'RCL':
                x = self.RCL_block(inp, dim)
            else:
                print('Available style : basic, res, dense, RCL')
                exit()
            x2 = MaxPooling2D()(x)
            x2 = self.build_U(x2, int(dim*2), depth-1)
            x2 = self.tconv_block(x2,int(dim*2))
            x2 = concatenate([x,x2])
            if self.style == 'basic':
                x2 = self.basic_block(x2, dim)
            elif self.style == 'res':
                x2 = self.res_block(x2, dim)
            elif self.style == 'dense':
                x2 = self.dense_block(x2, dim)
            elif self.style == 'RCL':
                x2 = self.RCL_block(x2, dim)
            else:
                print('Available style : basic, res, dense, RCL')
                exit()
        else:
            if self.style == 'basic':
                x2 = self.basic_block(inp, dim)
            elif self.style == 'res':
                x2 = self.res_block(inp, dim)
            elif self.style == 'dense':
                x2 = self.dense_block(inp, dim)
            elif self.style == 'RCL':
                x2 = self.RCL_block(inp, dim)
            else:
                print('Available style : basic, res, dense, RCL')
                exit()
        return x2

    def UNet(self):
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        o = self.build_U(self.I, self.dim, self.depth)
        o = Conv2D(self.out_ch, 1, activation='softmax')(o)
        model = Model(inputs=self.I, outputs=o)
        #if len(self.metrics)==2:
        #    model.compile(optimizer=Adam(lr=self.lr), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0]),getattr(self, self.metrics[1])])
        #else:
        #    model.compile(optimizer=Adam(lr=self.lr), loss=getattr(self, self.loss), metrics=[getattr(self, self.metrics[0])])
        if self.weight:
            model.load_weights(self.weight)
        return model

    def build(self):
        return self.UNet()

    def dice_coef(self, y_true, y_pred):
        from tensorflow.keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * K.round(y_pred), axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(K.round(y_pred), axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(dice[1:])

    def dice_loss(self, y_true, y_pred):
        from tensorflow.keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * y_pred, axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(K.pow(-K.log(dice[1:]),0.3))

    def ori_dice_loss(self, y_true, y_pred):
        from tensorflow.keras import backend as K
        smooth = 0.001
        intersection = K.sum(y_true * y_pred, axis=[1,2])
        union = K.sum(y_true, axis=[1,2]) + K.sum(y_pred, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return -K.mean(dice[1:])

    def dis_loss(self, y_true, y_pred):
        from tensorflow.keras import backend as K
        import cv2, numpy as np
        si=K.int_shape(y_pred)[-1]
        riter=2
        smooth = 0.001
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(riter*2+1,riter*2+1))
        kernel=kernel/(np.sum(kernel))
        kernel=np.repeat(kernel[:,:,np.newaxis],si,axis=-1)
        kernel=K.variable(kernel[:,:,:,np.newaxis])
        y_true_s=K.depthwise_conv2d(y_true,kernel,data_format="channels_last",padding="same")
        y_pred_s=K.depthwise_conv2d(y_pred,kernel,data_format="channels_last",padding="same")
        y_true_s = y_true_s > 0.8
        y_pred_s = y_pred_s > 0.8
        y_true_s = y_true - K.cast(y_true_s,'float32')
        y_pred_s = y_pred - K.cast(y_pred_s,'float32')
        #y_true_s = y_true - y_true_s
        #y_pred_s = y_pred - y_pred_s
        intersection = K.sum(y_true_s * y_pred_s, axis=[1,2])
        union = K.sum(y_true_s, axis=[1,2]) + K.sum(y_pred_s, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.sum(K.pow(-K.log(dice[1:]),0.3))

    def dis_dice_coef(self, y_true, y_pred):
        from tensorflow.keras import backend as K
        import cv2, numpy as np
        si=K.int_shape(y_pred)[-1]
        riter=2
        smooth = 0.001
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(riter*2+1,riter*2+1))
        kernel=kernel/(np.sum(kernel))
        kernel=np.repeat(kernel[:,:,np.newaxis],si,axis=-1)
        kernel=K.variable(kernel[:,:,:,np.newaxis])
        y_true_s=K.depthwise_conv2d(y_true,kernel,data_format="channels_last",padding="same")
        y_pred_s=K.depthwise_conv2d(y_pred,kernel,data_format="channels_last",padding="same")
        y_true_s = y_true_s > 0.8
        y_pred_s = y_pred_s > 0.8
        y_true_s = y_true - K.cast(y_true_s,'float32')
        y_pred_s = y_pred - K.cast(y_pred_s,'float32')
        #y_true_s = y_true - y_true_s
        #y_pred_s = y_pred - y_pred_s
        intersection = K.sum(y_true_s * y_pred_s, axis=[1,2])
        union = K.sum(y_true_s, axis=[1,2]) + K.sum(y_pred_s, axis=[1,2])
        dice = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
        return K.mean(dice[1:])

    def hd_loss(self, y_true, y_pred):
        from tensorflow.keras import backend as K
        import cv2, numpy as np
        def in_func(in_tensor,in_kernel,in_f):
            return K.clip(K.depthwise_conv2d(in_tensor,in_kernel,data_format="channels_last",padding="same")-0.5,0,0.5)*in_f
        si=K.int_shape(y_pred)[-1]
        f_qp=K.square(y_true-y_pred)*y_pred
        f_pq=K.square(y_true-y_pred)*y_true
        p_b=K.cast(y_true,'float32')
        p_bc=1-p_b
        q_b=K.cast(y_pred>0.5,'float32')
        q_bc=1-q_b
        rtiter=0
        si=K.int_shape(y_pred)[-1]
        for riter in range(3,19,3):
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(riter*2+1,riter*2+1))
            kernel=kernel/(np.sum(kernel))
            kernel=np.repeat(kernel[:,:,np.newaxis],si,axis=-1)
            kernel=K.variable(kernel[:,:,:,np.newaxis])
            if rtiter == 0:
                loss=K.mean(in_func(p_bc,kernel,f_qp)+in_func(p_b,kernel,f_pq)+in_func(q_bc,kernel,f_pq)+in_func(q_b,kernel,f_qp),axis=0)
            else:
                loss=loss+K.mean(in_func(p_bc,kernel,f_qp)+in_func(p_b,kernel,f_pq)+in_func(q_bc,kernel,f_pq)+in_func(q_b,kernel,f_qp),axis=0)
        return K.mean(loss[1:])

    def hyb_loss(self, y_true, y_pred):
        d_loss=self.dice_loss(y_true, y_pred)
        h_loss=self.dis_loss(y_true, y_pred)
        return 0.1*h_loss + d_loss

    def hyb_loss2(self, y_true, y_pred):
        d_loss=self.dice_loss(y_true, y_pred)
        h_loss=self.hd_loss(y_true, y_pred)
        if self.ratio==None:
            loss = h_loss + d_loss
        else:
            loss = self.ratio * h_loss + (1-self.ratio) * d_loss
        self.ratio = d_loss / h_loss
        return loss