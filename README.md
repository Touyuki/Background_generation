This project is used to generate the background of a image. The purpuse is to make one image look like book covers more. The process is like:

![image](https://github.com/Touyuki/Background_generation/blob/main/images/csa.png)

This model is based on a modified CSA-inpainting model. We deleted their CSA layer because our task is quite easier than image inpainting.
original paper:https://arxiv.org/abs/1905.12384

![image](https://github.com/Touyuki/Background_generation/blob/main/images/21.png)


These images shows the images in the last epoch in our training. The upper right one is the ground truth image and the bottom left one is generated one.

![image](https://github.com/Touyuki/Background_generation/blob/main/images/1.jpg)


![image](https://github.com/Touyuki/Background_generation/blob/main/images/2.jpg)


![image](https://github.com/Touyuki/Background_generation/blob/main/images/3.jpg)

### Train
```bash
python train.py
```

We uploaded pre-trained weight files. The verification code is "".
