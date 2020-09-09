# Random Noise Attenuation of Sparker Seismic Oceanography Data with Machine Learning

Hyunggu Jun, Hyeong-Tae Jou, Chung-Ho Kim, Sang Hoon Lee, Han-Joon Kim

This repository includes the codes and sample data for the paper
"Random Noise Attenuation of Sparker Seismic Oceanography Data with Machine Learning" in Ocean Science. 
 

The programs are based on the "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (https://github.com/cszn/DnCNN)"

### Requirements:
The version numbers that were used for the program, and newer version would work fine. 

Keras==2.2.4   
Keras-Applications==1.0.8   
Keras-Preprocessing==1.1.0   
numpy==1.16.2   
opencv-python==4.1.1.26   
scikit-learn==0.20.3   
scipy==1.2.1   
tensorboard==1.13.1   
tensorflow==1.13.1   
tensorflow-estimator==1.13.0   
tqdm==4.32.1   


### Data:
0.data/0.train: synthetic training data with 300x300 size   
0.data/1.noise: field noise data with 22431x740 and 22478x780   
0.data/2.test/noise_added: synthetic data with noise for test   
0.data/2.test/original_denoise: synthetic data without noise and synthetic denoised data for comparison   

### Code:
1.program/cube.txt: size of each data    
1.program/train.py: program for training DnCNN model   
1.program/data_generator.py: training data generation code   
1.program/test.py: program for testing the trained model   

### Training: 
To train the DnCNN model, run with augmentations:

```bash
python train.py --model 'DnCNN' --batch_size 128 --train_data '../0.data/0.train/' --noise_data '../0.data/1.noise/' --epoch 20
```

Unless you specify the options, the default options will be used.

### Test:
To test the trained DnCNN model, run with augmentations:

```bash
python test.py --set_dir '../0.data/2.test/' --set_name 'noise_added/' --model_dir './models/DnCNN/' --model_name 'model_020.hdf5' --result_dir 'results'
```

Unless you specify the options, the default options will be used.


### Citation:
Jun, H., Jou, H.-T., Kim, C.-H., Lee, S. H., and Kim, H.-J.: Random Noise Attenuation of Sparker Seismic Oceanography Data with Machine Learning, Ocean Sci. Discuss., https://doi.org/10.5194/os-2020-13, in review, 2020.
