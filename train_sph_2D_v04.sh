#!/bin/sh
python3 2D_SPH/scenes/tools/train_sequential.py epochs 250 fig models/2D/sph_2D_v04_loss log_intervall 10 data_start 0 val_split 0.2 data_end 18 time_end 15 batch 32 ref 2D_data/patches/highres/ref_sph_2D_v02-01_d%03d_var%02d_%03d src 2D_data/patches/lowres/sph_2D_v02-01_d%03d_var%02d_%03d features sdf,vel checkpoint_intervall 10 var 1 model models/2D/sph_2D_v04 time_start 5
