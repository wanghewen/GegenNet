#!/bin/bash

epochs=300
declare -a DataSets=("house1to10" "amazon-book" "bonanza" "mlm" "review" "senate1to10")
declare -a Scripts=("./new_train_jacobi.py")
declare -a Eigen_methods=("SA_LA")
declare -a Spectral_transforms=("Jacobi_paper")
set -x
for dataset in "${DataSets[@]}"; do
  for script in "${Scripts[@]}"; do
    for eigen_method in "${Eigen_methods[@]}"; do
      for spectral_transform in "${Spectral_transforms[@]}"; do
        if [ "$spectral_transform" != "None" ];
        then
          declare -a Spectral_transform_layers=(3)
        else
          declare -a Spectral_transform_layers=(1)
        fi
        for spectral_transform_layer in "${Spectral_transform_layers[@]}"; do
          start=$(date +%s.%N)
          export PYTHONPATH=.;export CUBLAS_WORKSPACE_CONFIG=:16:8;python -u $script --dataset $dataset --epochs $epochs --eigen_method $eigen_method --spectral_transform "$spectral_transform" --spectral_transform_layer $spectral_transform_layer --use_cache_eigen_results
          dur=$(echo "$(date +%s.%N) - $start" | bc)
          { printf "\nExecution time: %.6f seconds\n" $dur; } 2>/dev/null
        done
      done
    done
  done
done
