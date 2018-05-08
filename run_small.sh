#!/bin/sh

python -m neuralparticles.scripts.gen_src config $1 data $2 manta "neuralparticles/build/"
python -m neuralparticles.scripts.train config $1 data $2 manta "neuralparticles/build/"
python -m neuralparticles.scripts.run config $1 data $2
python -m neuralparticles.scripts.show_data config $1 data $2 manta "neuralparticles/build/" type "res"