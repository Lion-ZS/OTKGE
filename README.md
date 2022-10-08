# OTKGE

This is the code of paper 
**OTKGE: Multi-modal Knowledge Graph Embeddings via Optimal Transport**.
## Dependencies
- Python 3.6+
- PyTorch 1.0~1.7
- NumPy 1.17.2+
- tqdm 4.41.1+

## Results
The results of **OTKGE** on **WN9IMG** and **FBIMG** are as follows.


## Reproduce the Results

### 1. Preprocess the Datasets
First we should preprocess the datasets.

```shell script
cd code
python3 process_datasets.py
```
Now, the processed datasets are in the `data` directory.

### 2. Reproduce the Results 

```
CUDA_VISIBLE_DEVICES=0 python3 learn.py --dataset WN9IMG --model OTKGE_wn --rank 500 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 2000 --regularizer N3 --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight

CUDA_VISIBLE_DEVICES=1 python3 learn.py --dataset FBIMG --model OTKGE_fb --rank 500 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 5000 --regularizer N3 --reg 1e-3 --max_epochs 150 \
--valid 5 -train -id 0 -save -weight