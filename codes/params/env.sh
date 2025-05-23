#!/bin/bash 

lqcd_data_path='/lqcd_data_path'
parent="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"/.. && pwd )"
export PYTHONPATH="$parent:$PYTHONPATH"

mkdir -p "$lqcd_data_path/figures"
mkdir -p "$lqcd_data_path/final_results"
