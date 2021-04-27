import tensorflow as tf
from tensorflow.keras import *
from  tensorflow.keras.initializers import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
import numpy as np

from tensorflow.keras.backend import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import *
from loss_metric import loss,f1_score1,bce_dice_loss

def create_model():

    ## Crop and Merge Layers ##
    def CropAndMerge(Input1, Input2):
            """
            Crop input1 so that it matches input2 and then
            return the concatenation of both channels.
            """
            Size1_x = (Input1).shape[1]
            Size2_x = (Input2).shape[1]

            Size1_y = (Input1).shape[2]
            Size2_y = (Input2).shape[2]
      
            diff_x = tf.divide(tf.subtract(Size1_x, Size2_x), 2)
            diff_y = tf.divide(tf.subtract(Size1_y, Size2_y), 2)
            diff_x = tf.cast(diff_x, tf.int32)
            Size2_x = tf.cast(Size2_x, tf.int32)
            diff_y = tf.cast(diff_y, tf.int32)
            Size2_y = tf.cast(Size2_y, tf.int32)
            crop = tf.slice(Input1, [0, diff_x, diff_y, 0], [-1, Size2_x, Size2_y, -1])
            concat = tf.concat([crop, Input2], axis=3)

            return concat
    #-------------------------------------		
    ## Attention Mechanism ##
    def attention_gt(input_x,input_g, fil_las):
        input_size = input_x.shape
        fil_int = fil_las//2
        
        input_g = Conv2D(filters=fil_las,kernel_size=(1,1), strides=(1, 1), activation='relu', padding='same',use_bias=True,
            kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(input_g)
        
        theta_x = Conv2D(filters=fil_int,kernel_size=(3,3), strides=(1, 1), activation='relu', padding='same',use_bias=True, 
                      kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(input_x)
        
        theta_x_size  =  theta_x.shape
        
        phi_g = Conv2D(filters=fil_int,kernel_size=(3,3), strides=(1, 1), activation='relu', padding='same',use_bias=True,
                  kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(input_g)
        
        phi_g_u = UpSampling2D(size=(2, 2), interpolation='bilinear')(phi_g)
        f = relu(add([theta_x,phi_g_u]))
        
        # psi_f = Conv2D(filters=fil_las,kernel_size=(3,3), strides=(1, 1), activation='relu', padding='same',use_bias=True,
                  # kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(f)
        

        psi_f = Conv2D(filters=fil_las, kernel_size=(1,1), strides=(1, 1),activation='relu',padding='same')(f)
        psi_f = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.0),gamma_initializer=Constant(1.0),momentum=0.5)(psi_f)
        
        sigm_psi_f = sigmoid(psi_f)
        
        expand = Reshape(target_shape=input_size[-3:])(sigm_psi_f)


        y = multiply([expand , input_x])

        return y
    ## Conv Block
    def conv_block(input, filters):
        
        x = Conv2D(filters,kernel_size=(3,3), strides=(1, 1), activation='relu', padding='same',use_bias=True, kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(input)
        x = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.0),gamma_initializer=Constant(1.0),momentum=0.5)(x)
        x = Conv2D(filters,kernel_size=(3,3), strides=(1, 1), activation='relu', padding='same',use_bias=True, kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(x)
        x = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.0),gamma_initializer=Constant(1.0),momentum=0.5)(x)
        
        return x
    ## Bottleneck Block
    def bottleneck_block(input, filters):
            
        x = Conv2D(filters,kernel_size=(3,3), strides=(1, 1), activation='relu', padding='same',use_bias=True, 
                  kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(input)
        x = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.0),
                              gamma_initializer=Constant(1.0),momentum=0.5)(x)   
        x = Conv2D(filters,kernel_size=(3,3), strides=(1, 1), activation='relu', padding='same',use_bias=True, 
                  kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(x)
        x = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.0),
                              gamma_initializer=Constant(1.0),momentum=0.5)(x)    
        x = Conv2D(filters,kernel_size=(3,3), strides=(1, 1), activation='relu', padding='same',use_bias=True, 
                  kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(x)
        x = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.0),
                              gamma_initializer=Constant(1.0),momentum=0.5)(x)
      
        return x

    ## Robust Residual block 
    def robust_residual_block(input, filters_inp):
        
        x1 = Conv2D(filters=filters_inp,kernel_size=(3,3), strides=(1, 1), padding='same',activation='relu',
                    use_bias=True, kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(input)
        x1 = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.01),
                                gamma_initializer=Constant(1.0),momentum=0.5)(x1)
      
        x2 = SeparableConv2D(filters=filters_inp,kernel_size=(3,3), strides=(1, 1),activation='relu',padding='same',
                            use_bias=True,depthwise_initializer='glorot_uniform',pointwise_initializer='glorot_uniform', 
                            kernel_initializer='glorot_normal',bias_initializer=Constant(0.1),depth_multiplier=1)(x1)
        x2 = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.01),
                                gamma_initializer=Constant(1.0),momentum=0.5)(x2)
      
        x3 = Conv2D(filters=filters_inp,kernel_size=(3,3), strides=(1, 1), padding='same',activation='relu',
                    use_bias=True, kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(x2)
        x3 = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.01),
                                gamma_initializer=Constant(1.0),momentum=0.5)(x3)
        
        x = concatenate([input,x3],axis=-1)
        x = Conv2D(filters=filters_inp,kernel_size=(3,3), strides=(1, 1),activation='relu', padding='same',use_bias=True, kernel_initializer='glorot_normal',bias_initializer=Constant(0.1))(x)
        x = BatchNormalization(epsilon=1e-3,beta_initializer=Constant(0.01),
                              gamma_initializer=Constant(1.0),momentum=0.5)(x)
      
        return x
    ## Attention Block ##
    def attention_decoder_block(input, filt,conc):
        atten_b = attention_gt(input_x=conc,input_g=input, fil_las=filt)
        x_ct = Conv2DTranspose(filters=filt, kernel_size=(2, 2),activation='relu', strides=(2, 2), padding='same',kernel_initializer='glorot_normal',bias_initializer=Constant(0.1),use_bias=True)(input)
        x = CropAndMerge(Input1=x_ct,Input2=atten_b) 
        return x

    def nuclei_segnet(
        input_shape,
        num_classes=1,
        output_activation='sigmoid'):

        inputs = Input(input_shape)   
        
        filters = [32,64,128,256,512]
    

        # for l in range(num_layers):
        x_conv1 = robust_residual_block(inputs, filters[0])
        x_pool1 = MaxPooling2D((2, 2), strides=(2, 2),padding="same")(x_conv1)
        x_conv2 = robust_residual_block(x_pool1, filters[1])
        x_pool2 = MaxPooling2D((2, 2), strides=(2, 2),padding="same")(x_conv2)  
        x_conv3 = robust_residual_block(x_pool2, filters[2])
        x_pool3 = MaxPooling2D((2, 2), strides=(2, 2),padding="same")(x_conv3)
        x_conv4 = robust_residual_block(x_pool3, filters[3])
        x_pool4 = MaxPooling2D((2, 2), strides=(2, 2),padding="same")(x_conv4)
        x_conv5 = bottleneck_block(x_pool4, filters[4])

    # upsampling in the form of convtranspose

        x_tconv5 = attention_decoder_block(x_conv5, filters[3],x_conv4)
        u_conv4 = conv_block(x_tconv5, filters[3])
        x_tconv4 = attention_decoder_block(u_conv4, filters[2],x_conv3)
        u_conv3 = conv_block(x_tconv4, filters[2])
        x_tconv3 = attention_decoder_block(u_conv3, filters[1],x_conv2)
        u_conv2 = conv_block(x_tconv3, filters[1])
        x_tconv2 = attention_decoder_block(u_conv2, filters[0],x_conv1)
        u_conv1 = conv_block(x_tconv2, filters[0])
                  
        outputs = Conv2D(num_classes, kernel_size=(1,1), strides=(1,1), activation=output_activation, padding='same') (u_conv1)       
        
        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    model = nuclei_segnet(
                  input_shape=(256,256,3),
                  num_classes=1,
                  output_activation='sigmoid')

    adam = optimizers.Adam(lr = 0.001)
    model.compile(optimizer = adam,loss=loss,metrics=[f1_score1])

    return model
