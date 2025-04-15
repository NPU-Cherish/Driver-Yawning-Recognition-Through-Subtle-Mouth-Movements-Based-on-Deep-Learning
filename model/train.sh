#!/bin/bash

cur_date=`date +%Y-%m-%d-%H-%M-%S`
log_file_name=./log/${cur_date}
caffe train \
    -solver ./solver.prototxt \
    -snapshot=
