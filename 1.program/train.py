import argparse
import re
import os, glob, datetime
import numpy as np
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam
import data_generator as dg
import keras.backend as K
import cv2
import random

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='../0.data/0.train/', type=str, help='path of train data')
parser.add_argument('--noise_data', default='../0.data/1.noise/', type=str, help='path of train data')
parser.add_argument('--epoch', default=20, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=1, type=int, help='save model at every x epoches')
args = parser.parse_args()


save_dir = os.path.join('models',args.model) 
train_data_out = 0
if not os.path.exists('models'):
    os.mkdir('models')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def DnCNN(depth,filters=64,image_channels=1, use_bnorm=True):
    layer_count = 0
    inpt = Input(shape=(None,None,image_channels),name = 'input'+str(layer_count))
    # 1st layer, Conv+relu
    layer_count += 1
    x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(inpt)
    layer_count += 1
    x = Activation('relu',name = 'relu'+str(layer_count))(x)
    # depth-2 layers, Conv+BN+relu
    for i in range(depth-2):
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
        if use_bnorm:
            layer_count += 1
            x = BatchNormalization(axis=3, momentum=0.9,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
        layer_count += 1
        x = Activation('relu',name = 'relu'+str(layer_count))(x)  
    # last layer, Conv
    layer_count += 1
    x = Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
    layer_count += 1
    x = Subtract(name = 'subtract' + str(layer_count))([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir,'model_*.hdf5'))  # get name list of all .hdf5 files
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*",file_)
            epochs_exist.append(int(result[0]))
        initial_epoch=max(epochs_exist)   
    else:
        initial_epoch = 0
    return initial_epoch

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch<=20:
        lr = initial_lr
    elif epoch<=40:
        lr = initial_lr/10
    elif epoch<=60:
        lr = initial_lr/20 
    else:
        lr = initial_lr/20 
    log('current learning rate is %2.8f' %lr)
    return lr

def extract_noise(inshape,noise_dir):
    noise_num = 2
    patch_size = inshape[1]
    batch_size = inshape[0]
    
    batch = np.zeros((batch_size,patch_size,patch_size,1))
    
    file_list = glob.glob(noise_dir+'/*.norm')
    file_num = random.randint(0,noise_num-1)
    fin = open(file_list[file_num],"rb")
    img = np.fromfile(fin,dtype="float32")
    img = img.reshape(-1,740)
    ww, hh = img.shape
    
    hh = hh-patch_size
    ww = ww-patch_size
    
    for ibatch in range(batch_size):
        hrand = random.randint(0, hh-1)
        wrand = random.randint(0, ww-1)
        noise = img[wrand:wrand+patch_size,hrand:hrand+patch_size]
        noise = noise.reshape(1,patch_size,patch_size)
        if ibatch == 0:
            noise_batch = noise
        elif ibatch > 0:
            noise_batch = np.vstack([noise_batch,noise])

    noise_batch = noise_batch.reshape(batch_size,patch_size,patch_size,1)

    return noise_batch
    
def train_datagen(epoch_iter=1,epoch_num=1,batch_size=128,data_dir=args.train_data,noise_dir=args.noise_data):
    while(True):
        n_count = 0
        if n_count == 0:
            xs = dg.datagenerator(data_dir)
            assert len(xs)%args.batch_size ==0, \
            log('make sure the last iteration has a full batchsize, this is important if you use batch normalization!')
            xs = xs.astype('float32')
            indices = list(range(xs.shape[0]))
            print("shape:", xs.shape) 
            print("shape[0]:",xs.shape[0])
            print("range:",range(xs.shape[0]))
            print("len(indices):",len(indices))
            n_count = 1
            print("batch_size:", batch_size)
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                ratio1 = np.random.uniform(0.7,0.3, batch_size)
                ratio2 = 1.0 - ratio1

                batch_x = xs[indices[i:i+batch_size]]
                noise = extract_noise(batch_x.shape,noise_dir)
  
                for jj in range(batch_size):
                    batch_x[jj] = batch_x[jj]*ratio1[jj]
                    noise[jj] = noise[jj]*ratio2[jj]

                batch_y = (batch_x + noise)
                if(train_data_out):
                    for j in range(0, batch_size):
                        yyy = np.float32(batch_y[j,:,:,0])
                        xxx = np.float32(batch_x[j,:,:,0])
                        nnn = np.float32(noise[j,:,:,0])

                        fout1 = open("./noise_add/noise_extract_%06d.bin"%(i+j),"wb")
                        fout2 = open("./noise_add/seismic_%06d.bin"%(i+j),"wb")
                        fout3 = open("./noise_add/noise_%06d.bin"%(i+j),"wb")
                        yyy.tofile(fout1)
                        xxx.tofile(fout2)
                        nnn.tofile(fout3)
                        fout1.close()
                        fout2.close()
                        fout3.close()

                yield batch_y, batch_x
        
# define loss
def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true))/2
    
if __name__ == '__main__':
    # model selection
    model = DnCNN(depth=17,filters=64,image_channels=1,use_bnorm=True)
    model.summary()
    
    # load the last model in matconvnet style
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:  
        print('resuming by loading epoch %03d'%initial_epoch)
        model = load_model(os.path.join(save_dir,'model_%03d.hdf5'%initial_epoch), compile=False)
    
    # compile the model
    model.compile(optimizer=Adam(0.001), loss=sum_squared_error)
    
    # use call back functions
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'model_{epoch:03d}.hdf5'), 
                verbose=1, save_weights_only=False, period=args.save_every)
    csv_logger = CSVLogger(os.path.join(save_dir,'log.csv'), append=True, separator=',')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    history = model.fit_generator(train_datagen(batch_size=args.batch_size),
                steps_per_epoch=171, epochs=args.epoch, verbose=1, initial_epoch=initial_epoch,
                callbacks=[checkpointer,csv_logger,lr_scheduler])

