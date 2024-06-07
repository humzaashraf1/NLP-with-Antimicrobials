#!/bin/bash

# Define the output folder
output_parent_folder="D:/NLP-with-Antimicrobials/ProteinMPNN_AMPs/output/VanillaLoop"

# Define the number of times to run the script
num_runs=5

# Loop through the specified number of runs
for ((i=1; i<=$num_runs; i++)); do
    # Create a subfolder for the current iteration
    output_folder="$output_parent_folder/Run_$i"
    mkdir -p "$output_folder"  # Create the folder if it doesn't exist

    echo "Running iteration $i..."
    # Call your Python script with the specified output folder
    python //ProteinMPNN-main/ProteinMPNN-main/protein_mpnn_run.py \
        --model_name v_48_020 \
        --path_to_model_weights "//ProteinMPNN-main/ProteinMPNN-main/vanilla_model_weights" \
        --out_folder "$output_folder" \
        --pdb_path "D:/NLP-with-Antimicrobials/ProteinMPNN_AMPs/PDB_Files/AMP99.pdb"
done

echo "All iterations completed."
