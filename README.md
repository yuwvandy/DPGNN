# DPGNN
This repository is an official PyTorch(Geometric) implementation of DPGNN(DPGCN) in "Distance-wise Prototypical Graph Neural Network for Node Imbalance Classification"

![](./image/Framework.png)


## Requirements
* PyTorch 1.8.1+cu111
* PyTorch Geometric 1.7.0
* Pytorch-scatter 2.0.7
* NetworkX 2.5.1
* Tqdm 4.61.0
* Sklearn 0.0

Note that the version of PyTorch and PyTorch Geometric should be compatible and PyTorch Geometric is related to other packages, which requires to be installed beforehand. It is recommended to follow the [installation instruction](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).

## Run
* To reproduce the performance comparison and the ablation study in the following Table and the Figure , run
```linux
bash run.sh
```
![](./image/table.png)
![](./image/ablation.png)


* To reproduce our results under different imbalance ratio, run
```linux
bash run_ratio.sh
```
![](./image/ratio.png)
