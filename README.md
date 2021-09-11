# Cycle-GAN-Implementation

A simple implementation of [CycleGAN](https://junyanz.github.io/CycleGAN/).

**Environment:**
* Windows 10
* Tensorflow 2.1.0
* OpenCV 3.4.2

**Clone the repository:**

`$ git clone https://github.com/Torocheeze/Cycle-GAN-Implementation.git`

`$ cd Cycle-GAN-Implementation`

## CycleGAN Training&Testing
Download [dataset](https://drive.google.com/file/d/1E4WJ8zHZgfZiA_1QXOL8rchX7RZTiB32/view?usp=sharing), this is **summer2winter_yosemite** mentioned [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md).

Move them to 'Cycle-GAN-Implementation/Cycle_GAN/data'.

* Training

`$ python Cycle_code/cycle_GAN.py`

* Testing

`$ python Cycle_code/test_model.py`


## Simple GUI

`$ python main.py`

Import sample image from 'Cycle-GAN-Implementation/Cycle_code/data/samples/', transform into colour image and save it. 

![GUI](/GUI_sample.JPG)

