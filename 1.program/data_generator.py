import glob
import cv2
import numpy as np
from tqdm import tqdm


patch_size, stride = 50, 50
aug_times = 2
scales = [1.1, 1.0, 0.9]
batch_size = 128


def show(x,title=None,cbar=False,figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x,interpolation='nearest',cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()

def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name):

    fin = open(file_name,"rb")
    img = np.fromfile(fin,dtype="float32")
    fin.close()
    n1=300
    img = img.reshape(-1,n1)
    w, h = img.shape
    jj = 0
    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        img_scaled = cv2.resize(img, (h_scaled,w_scaled), interpolation=cv2.INTER_CUBIC)
        for i in range(0, w_scaled-patch_size+1, stride):
            for j in range(0, h_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0,1))
                    
                    if jj == 0:
                        patches = x_aug
                    else:
                        patches = np.vstack([patches,x_aug])
                    jj = jj+1

    patches = patches.reshape(-1,patch_size,patch_size)
                
    return patches

def datagenerator(data_dir='../0.data/0.train/',verbose=False):
    
    file_list = glob.glob(data_dir+'/*.norm')  # get name list of all .png files
    for i in tqdm(range(len(file_list))):
        patch = gen_patches(file_list[i])
    
        if i == 0:
            data = patch
        elif i > 0:
            data = np.vstack([data,patch])
        if verbose:
            print(str(i+1)+'/'+ str(len(file_list)) + ' is done')
    data = data.reshape((data.shape[0],data.shape[1],data.shape[2],1))
    print("data.shape:",data.shape)
    discard_n = len(data)-len(data)//batch_size*batch_size;
    data = np.delete(data,range(discard_n),axis = 0)
    print("data.shape:",data.shape)
    return data

if __name__ == '__main__':   

    data = datagenerator(data_dir='../0.data/0.train/')
