import numpy as np
import json

# Define the sequence
sequence = "MGAIAKLVAKFGWPFIKKFYKQIMQFIGQGWTIDQIEKWLKRH"
num_positions = len(sequence)
num_amino_acids = 21  # 20 standard amino acids

# Simulate frequencies for each amino acid at each position
frequencies = np.random.dirichlet(np.ones(num_amino_acids), size=num_positions)

# Normalize frequencies to probabilities
probabilities = frequencies / frequencies.sum(axis=1, keepdims=True)

# Transform probabilities to bias values (using log transformation)
bias_values = np.interp(probabilities, (probabilities.min(), probabilities.max()), (-2, 2))

# Amino acid order (columns in the bias matrix)
amino_acids = "ACDEFGHIKLMNPQRSTVWYX"

# Create the JSON structure
bias_dict = {"AMP99": {"A": bias_values.tolist()}}

# Save to JSONL file
with open('bias_by_res.jsonl', 'w') as jsonl_file:
    jsonl_file.write(json.dumps(bias_dict) + "\n")

print("Bias matrix generated and saved to 'bias_by_res.jsonl'")