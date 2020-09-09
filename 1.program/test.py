# run this to test the model

import argparse
import os, time, datetime
import numpy as np
from keras.models import load_model, model_from_json
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='../0.data/2.test/', type=str, help='directory of test dataset')
    parser.add_argument('--set_name', default='noise_added/', type=str, help='name of test dataset')
#    parser.add_argument('--model_dir', default=os.path.join('models','DnCNN'), type=str, help='directory of the model')
    parser.add_argument('--model_dir', default='./models/DnCNN/', type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model_020.hdf5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--save_result', default=1, type=int, help='save the denoised image, 1 or 0')
    return parser.parse_args()
    
def to_tensor(img):
    if img.ndim == 2:
        return img[np.newaxis,...,np.newaxis]
    elif img.ndim == 3:
        return np.moveaxis(img,2,0)[...,np.newaxis]

def from_tensor(img):
    return np.squeeze(np.moveaxis(img[...,0],0,-1))

def log(*args,**kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"),*args,**kwargs)

def save_result(result,path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt','.dlm'):
        np.savetxt(path,result,fmt='%2.4f')
    else:
        imsave(path,np.clip(result,0,1))


def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


if __name__ == '__main__':    
    
    args = parse_args()
    
    if not os.path.exists(os.path.join(args.model_dir, args.model_name)):
        # load json and create model
        json_file = open(os.path.join(args.model_dir,'model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(os.path.join(args.model_dir,'model.h5'))
        log('load trained model on Train400 dataset by kai')
    else:
        model = load_model(os.path.join(args.model_dir, args.model_name),compile=False)
        log('load trained model')

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
        
    set_name = args.set_name    
    if not os.path.exists(os.path.join(args.result_dir,set_name)):
        os.mkdir(os.path.join(args.result_dir,set_name))
    psnrs = []
    ssims = [] 
    
    for im in os.listdir(os.path.join(args.set_dir,set_name)): 
        if im.endswith(".bin"):
            fin = open(os.path.join(args.set_dir,set_name,im),"rb")
            x = np.fromfile(fin,dtype=np.float32)
            fin.close()

            x = x.reshape(-1,496)
            y = x 
            y = y.astype(np.float32)
            y_  = to_tensor(y)
            start_time = time.time()
            x_ = model.predict(y_) # inference
            elapsed_time = time.time() - start_time
            print('%10s : %10s : %2.4f second'%(set_name,im,elapsed_time))
            x_=from_tensor(x_)
            if args.save_result:
                name, ext = os.path.splitext(im)

                fout1 = open("./%s/%s/%s_dncnn.bin"%(args.result_dir,set_name,name),"wb")
                fout2 = open("./%s/%s/%s_noise.bin"%(args.result_dir,set_name,name),"wb")
                xxx = np.float32(x_)
                yyy = np.float32(y)
                xxx.tofile(fout1)
                yyy.tofile(fout2)
                fout1.close()
                fout2.close()
    

    
        
        


