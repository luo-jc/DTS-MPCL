# DTS-MPCL

## Usage

### Requirements
* Python 3.7.12
* PyTorch 1.13.1
* Torchvision 0.14.1
* CUDA 11.4 on Ubuntu 18.04

Install the require dependencies:
```

```

### Dataset
1. Download the dataset (include the training set and test set).
2. Move the dataset to `./` and reorganize the directories as follows:
```
./Training
|--001
|  |--262A0898.tif
|  |--262A0899.tif
|  |--262A0900.tif
|  |--exposure.txt
|  |--HDRImg.hdr
|--002
...
./Test (include 15 scenes from `EXTRA` and `PAPER`)
|--001
|  |--262A2615.tif
|  |--262A2616.tif
|  |--262A2617.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
|--BarbequeDay
|  |--262A2943.tif
|  |--262A2944.tif
|  |--262A2945.tif
|  |--exposure.txt
|  |--HDRImg.hdr
...
```

### Training & Evaluaton

To train the model, run:
```
python train.py
python train_MPCL.py
```
To evaluate, run:
```
python test.py --pretrained_model ./Model/DTS.pkl  --save_results --save_dir ./test_result
```
