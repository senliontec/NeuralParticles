#!/bin/sh
python3 2D_SPH/scenes/tools/train_sequential.py ref 2D_data/patches/highres/ref_sph_2D_v02-05_d%03d_var%02d_%03d time_end 15 features ps,dens,vel data_end 18 fig models/2D/sph_2D_v10_loss var 1 checkpoint_intervall 10 model models/2D/sph_2D_v10 val_split 0.2 src 2D_data/patches/lowres/sph_2D_v02-05_d%03d_var%02d_%03d log_intervall 10 epochs 250 time_start 5 batch 32 data_start 0
