#!/bin/bash

python protein_mpnn_run.py \
    --model_name v_48_020 \
    --path_to_model_weights "C:/Users/humza/Desktop/ProteinMPNN-main/ProteinMPNN-main/vanilla_model_weights" \
    --out_folder "D:/NLP-with-Antimicrobials/ProteinMPNN_AMPs/output" \
    --pdb_path "D:/NLP-with-Antimicrobials/ProteinMPNN_AMPs/AMP99.pdb" \
    --seed 37