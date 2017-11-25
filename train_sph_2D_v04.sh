#!/bin/sh
python3 2D_SPH/scenes/tools/train_sequential.py val_split 0.2 model models/2D_data/sph_2D_v04 checkpoint_intervall 10 features sdf,vel src 2D_data/patches/lowres/sph_2D_v02-01_d%03d_var%02d_%03d log_intervall 10 ref 2D_data/patches/highres/ref_sph_2D_v02-01_d%03d_var%02d_%03d batch 32 time_start 5 fig models/2D_data/sph_2D_v04_loss time_end 15 data_start 0 epochs 250 data_end 18 var 1
