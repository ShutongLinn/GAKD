#!/bin/bash

BASE_PATH=${1:-"/home/GAKD"}

export TF_CPP_MIN_LOG_LEVEL=3

BASE_PATH_LIST="${BASE_PATH}/data/dolly:${BASE_PATH}/data/self-inst:${BASE_PATH}/data/sinst/0_2:${BASE_PATH}/data/sinst/3_6:${BASE_PATH}/data/sinst/6_10:${BASE_PATH}/data/sinst/11_:${BASE_PATH}/data/uinst/0_2:${BASE_PATH}/data/uinst/3_5:${BASE_PATH}/data/uinst/6_10:${BASE_PATH}/data/uinst/11_"


export BASE_PATH_LIST

PYTHONPATH=${BASE_PATH} python ${BASE_PATH}/tools/process_data_eval.py
