# ECE-GY-7123-mini-project
Current work:  
1. Using the offered [repository](https://github.com/kuangliu/pytorch-cifar/blob/master/main.py) to walk through the architecture.   
2. Adjust the project into a main file that calling all the functions.   
3. Adjust Resnet model to gain progress.

Other to-dos:  
1. adding path to model. As [this](https://github.com/mayankpoddar/ResNet/blob/main/project1_model.py) line 53 so that trained model can be kept.

References:     
https://github.com/wikibook/keras/blob/master/chapter2-deep-networks/resnet-cifar10-2.2.1.py   
https://github.com/mayankpoddar/ResNet/blob/main/project1_model.py  

-------------------------------------------------------------------------------------------------------------

The most important thing for this mini-project is tunning the hyper parameter. Here are some hyper parameters that we can tune:

- Ci, the number of channels in the i th layer.
- Fi, the filter size in the i th layer
- Ki, the kernel size in the i th skip connection
- P, the pool size in the average pool layer  

Here are the things we can add to our model:

- any optimizer (SGD, ADAM, etc)
-  any data augmentation strategy
-  any regularizer
-  any choice of learning rate, batch size, epochs, etc  

### What should be included in the report:

- A short overview of your project, along with a summary of your findings  
- A methodology section that explains how you went about designing and
  training your models, pros and cons of different architectural choices, what
  lessons you learned during the design process  
- A results section that reports your final test accuracy, model architecture,
  and number of parameters  
- Any relevant citations  