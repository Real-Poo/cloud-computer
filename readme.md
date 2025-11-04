# Cloud Computing Architecture

## Train
``` python
python train.py \
  --train_hr_dir /path/to/trainHR \
  --val_hr_dir   /path/to/valHR \
  --out_dir      ./checkpoints_unetsr_qat \
  --in_ch 3 --scale 2 --patch 128 --base 64 --bs 8 --epochs 30 --lr 1e-4
```

## Test
``` python
python test.py \
  --ckpt ./checkpoints_unetsr_qat/final_int8_decoder.pth \
  --img  ./some_lr_image.png \
  --in_ch 3 --scale 2 --patch 128 --show
```
