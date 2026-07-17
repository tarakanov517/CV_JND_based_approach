#!/bin/bash

mkdir -p /scratch/$USER/data
cd /scratch/$USER/data
wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip -q -o tiny-imagenet-200.zip