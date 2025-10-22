# CGSoRec

# Condition-Guided Social Recommendation Model
This is the pytorch implementation of our paper "Balancing User Preferences by Social Networks: A Condition-Guided Social Recommendation Model for  Mitigating Popularity Bias".

<div align="center">
  <img src="https://github.com/hexin5515/CGSoRec/blob/master/Image/CGSoRec.jpg" width="1600px"/>
</div>

## Environment
- python 3.8.10
- pytorch 1.12.0
- numpy 1.22.0

## Usage
### Data
The experimental data are in './datasets' folder, including LastFM, DBook and Ciao.

### Training
```
cd ./CGSoRec
python main.py --cuda --dataset=$1 --data_path=../datasets/$1/ --lr=$2 --weight_decay=$3 --batch_size=$4 --dims=$5 --emb_size=$6 --mean_type=$7 --steps=$8 --noise_scale=$9 --noise_min=${10} --noise_max=${11} --sampling_steps=${12} --reweight=${13} --log_name=${14} --round=${15} --gpu=${16}
```

### Inference

1. Download the checkpoints released by us.
2. Put the 'checkpoints' folder into the current folder.
3. Run inference.py
```
python inference.py --dataset=$1 --gpu=$2
```

### Examples

1. Train CGSoRec on LastFM
```
cd ./CGSoRec
python main.py --cuda --dataset=lastfm_unbias --lr=0.0001 --weight_decay=0 --batch_size=400 --dims=[1000] --emb_size=10 --mean_type=x0 --steps=10 --sampling_steps=0 --log_name=log --gpu=0
```
2. Inference CGSoRec on LastFM
```
cd ./CGSoRec
python inference.py --dataset=lastfm_unbias --gpu=0
```

