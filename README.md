# CDDNet



### Abstract


Recently, there has been increasing attention on single image super resolution (SISR) based on generative adversarial networks (GAN) due to their outstanding performance and wide range of applications. 
However, previous researches have shown that explicit modeling methods have limited utilization of implicit information within images, and are less robust to advanced degradation, often resulting in over-smoothed or visually uncomfortable artifacts. 
Additionally, kernel-based methods can be highly sensitive to minor estimation errors and can produce poor super resolution results. 
For these issues, we propose a cross dropout based dynamic network (CDDNet) for self supervised dynamic blind super resolution.
CDDNet models the degradation of low-resolution (LR) images using degradation weights as the global attention and includes a gyroscope structure as the local attention mechanism to further improve performance. 
CDDNet incorporates our proposed cross dropout module to enhance its performance in handling multiple deep degradation. 
GANloss is used with pixel loss and adversarial loss to improve visual quality. 
We conduct ablation experiments on the effectiveness of cross dropout and also experiment on the effectiveness of the overall network framework. 
Experimental results show that CDDNet is very effective in the field of SISR, achieving higher reconstruction accuracy and acceptable quality of visual perception, with good results on both synthetic and real-world dataset.



Overall pipeline of the CDDNet:

![illustration](total.PNG)

For more details, please refer to our paper.

#### Getting started

- Install dependencies. (Python 3 + NVIDIA GPU + CUDA. Recommend to use Anaconda)
```bash
pip install -r requirements.txt
```

- Prepare the training and testing dataset by following this [instruction](datasets/README.md).
- Prepare the pre-trained models by following this [instruction](experiments/README.md).

#### Training

First, check and adapt the yml file ```options/train/DASR/train_DASR.yml```, then

- Single GPU:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python dasr/train.py -opt options/train/DASR/train_DASR.yml --auto_resume
```

- Distributed Training:
```bash
YTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4335 dasr/train.py -opt options/train/DASR/train_DASR.yml --launcher pytorch --auto_resume

```

Training files (logs, models, training states and visualizations) will be saved in the directory ```./experiments/{name}```

#### Testing

First, check and adapt the yml file ```options/test/DASR/test_DASR.yml```, then run:
```bash
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/DASR/test_DASR.yml
```

Evaluating files (logs and visualizations) will be saved in the directory ```./results/{name}```


### Acknowledgement
This project is built based on the excellent [BasicSR](https://github.com/xinntao/BasicSR) and [DASR](https://github.com/csjliang/DASR) project.

