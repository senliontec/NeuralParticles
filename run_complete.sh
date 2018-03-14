#!/bin/sh

python gen_ref.py config $1 data $2 manta $3
python gen_src.py config $1 data $2 manta $3
python train.py config $1 data $2 manta $3
python run.py config $1 data $2
python show_data.py config $1 data $2 manta $3 type "res"