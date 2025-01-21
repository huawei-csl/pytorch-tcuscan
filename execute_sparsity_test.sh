#!/bin/bash

densities=("0.0" "0.001" "0.01" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7")

for density in "${densities[@]}"; do
    make profile_fp16_seg_scan_sc DENSITY="$density"
done
