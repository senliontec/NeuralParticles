#!/bin/sh
python3 2D_SPH/scenes/tools/train_sequential.py val_split 0.2 ref 2D_data/patches/highres/ref_sph_2D_v02-01_d%03d_var%02d_%03d time_start 5 features sdf,vel model 2D_data/model/sph_2D_v04 var 1 time_end 15 epochs 250 data_end 18 fig 2D_data/model/sph_2D_v04_loss checkpoint_intervall 10 data_start 0 src 2D_data/patches/lowres/sph_2D_v02-01_d%03d_var%02d_%03d log_intervall 10 batch 32
