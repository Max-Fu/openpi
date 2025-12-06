# Instructions for running yam baselines 

## conversion of data 
```bash
pushd /mnt/amlfs-02/shared/datasets/yam_v5
bash download.sh
popd
bash prep_dataset.sh
```

## Generating training scripts 
```bash 
bash create_train.sh
```

## Launching training
look in the yam_train_scripts folder
launch them as follows
```bash 
cd /mnt/amlfs-01/home/maxf/research/openpi
uv python install 3.11
tmux 
bash yam_train_scripts/train_xdof.a1beta_2025-11-04_19-25-21_USA.sh
```