#!/bin/bash

rm -r ./hpctoolkit-vectorAdd-measurements/

hpcrun -e gpu=nvidia ./vectorAdd
hpcstruct --gpucfg yes ./hpctoolkit-vectorAdd-measurements/

