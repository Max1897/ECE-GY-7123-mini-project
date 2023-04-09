# ECE-GY-7123-mini-project
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

(Important things below)

## **Method Design for Tuning:**

**Step One:**

The **First** parameter we tune is the **Channel Size**:

**Four** Channel Size numbers will be tested in:

32 (Original) --->	64 --->	**42 (Best)**	--->	25

- Plot the 4 graphs
- Compare the results

---------------------------------------------------------------------

**Step Two:**

The Second parameter we tune is the **Filter Size**:

The two parameters we need to tune is: **Padding** and **Kernel_Size**

Three sets will be tested:

[padding = 1, padding = 1, kernel_size = 3]

**[padding = 2, padding = 2, kernel_size = 5] (Best)**

[padding = 3, padding = 3, kernel_size = 7]

---------------------------------------------------------------------

**Step Three (important!!!):**

Add a **Learning Rate Decay** parameter for tuning the Learning Rate with two other parameters: **high_speed_lr_decay** and **low_speed_lr_decay**

Here is the Basic Coding Logic for Tuning Learning Rate:

```python
if lr > 0.003:
    lr_decay = high_speed_lr_decay
else:
    lr_decay = low_speed_lr_decay
lr *= lr_decay
```

high_speed_lr_decay should set lower.

low_speed_lr_decay should set higher.

---------------------------------------------------------------------

**Step Four:**

Compare Optimizer: **SGD** vs **Adams**

SGD Parameters setting

- WEIGHT_DECAY
- MOMENTUM
- DAMPENING
- LEARNING RATE

Adams Parameters setting:

- WEIGHT_DECAY
- LEARNING RATE
- PARAMS

-------------------------------------------------------------

**Step Five:**

Compare ResNet 18 with ResNet 34











