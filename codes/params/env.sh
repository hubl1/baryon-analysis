#!/bin/bash 

lqcd_data_path='/lqcd_data_path'
parent="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )"/.. && pwd )"
export PYTHONPATH="$parent:$PYTHONPATH"
