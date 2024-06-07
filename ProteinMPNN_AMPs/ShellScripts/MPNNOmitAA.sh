#!/bin/bash

python protein_mpnn_run.py \
    --model_name v_48_020 \
    --path_to_model_weights "//ProteinMPNN-main/ProteinMPNN-main/vanilla_model_weights" \
    --out_folder "D:/NLP-with-Antimicrobials/ProteinMPNN_AMPs/output/OmitAA" \
    --pdb_path "D:/NLP-with-Antimicrobials/ProteinMPNN_AMPs/PDB_Files/AMP99.pdb" \
    --seed 37 \
    --omit_AAs YW