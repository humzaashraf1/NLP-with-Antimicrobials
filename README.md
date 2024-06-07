# Natural Language Processing with Antimicrobial Peptides
In this repository we will be playing with sequences of antimicrobial peptides (AMPs), which are short sequences of proteins that have therapeutic potential for killing bacterial cells. 
The dataset we will use comes from the following paper: "**AMPlify: attentive deep learning model for discovery of novel antimicrobial peptides effective against WHO priority pathogens**" (Li, C., Sutherland, D., Hammond, S.A. et al. AMPlify: attentive deep learning model for discovery of novel antimicrobial peptides effective against WHO priority pathogens. BMC Genomics 23, 77 (2022). https://doi.org/10.1186/s12864-022-08310-4). 

While antimicrobial peptides are a salient solution for combating threats of antibiotic resistance (due to their biochemical diversity), it remains unclear what sequence properties characterize both optimal specificity (E.g., low toxicity for mammalian cells) as well as biopharmaceutical manufacturability (E.g., good biophysical/ biochemical properties for isolating and purifying the drug). The sequence space for peptides of up to 200 residues in length for the 20 naturally occurring amnio acids is 20<sup>200</sup> (or 1.6 x 10<sup>260</sup>). For reference, there are only 10<sup>80</sup> atoms in the observable universe. As a result, there are essentially infinitely many sequences that could conceivably have anti-microbial activity. However, in practice, the true fitness landscape for naturally occurring sequences is on the order of tens to hundreds of thousands. To date, the Database of Antimicrobial Activity and Structure of Peptides (https://www.dbaasp.org/home) has documented approximately 20,000 peptides to have some form of antimicrobial activity based on many thousands of aggregated research publications. Innovations in natural language processing in recent years has made it possible to "read" drug sequences as plain text in order to derive "context". This context may represent the model's ability to "learn" the fundamental chemical laws that underlie structure, activity, and other biophysical properties relevant to the drug development process. Furthermore, the ability to contextualize protein sequences by both their structure and activity has far-reaching potential even beyond antibiotic resistance--like towards the development of cancer-targeting antibody therapies, an increasingly popular approach for highly specific targeting of malignant cells.

## Understanding the Dataset
"We used publicly available AMP sequences to train and test AMP predictors. In order to build a non-redundant AMP dataset, we first downloaded all available sequences from two manually curated databases: Antimicrobial Peptide Database [44] (APD3, http://aps.unmc.edu/AP) and Database of Anuran Defense Peptides [39] (DADP, http://split4.pmfst.hr/dadp). Since APD3 is being frequently updated, we used a static version that was scraped from the website on March 20, 2019 comprising 3061 sequences. Version 1.6 of DADP contains 1923 distinct mature AMPs. We concatenated these two sets and removed duplicate sequences, producing a non-redundant (positive) set of 4173 distinct, mature AMP sequences, all 200 amino acid residues in length or shorter...We designed a rigorous selection strategy for our non-AMP sequences (Supplementary Fig. S3), using sequences from the UniProtKB/Swiss-Prot database [46] (2019_02 release), which only contains manually annotated and reviewed records from the UniProt database. First, we downloaded sequences that are 200 amino acid residues or shorter in length (matching the maximum peptide length in the AMP set), excluding those with annotations containing any of the 16 following keywords related to antimicrobial activities: {antimicrobial, antibiotic, antibacterial, antiviral, antifungal, antimalarial, antiparasitic, anti-protist, anticancer, defense, defensin, cathelicidin, histatin, bacteriocin, microbicidal, fungicide}. Second, duplicates and sequences with residues other than the 20 standard amino acids were removed. Third, a set of potential AMP sequences annotated with any of the 16 selected keywords were downloaded and compared with our candidate negative set. We noted instances where a sequence with multiple functions was annotated separately in multiple records within the database, and removed sequences in common between candidate non-AMPs and potential AMPs. The candidate non-AMP sequences were also checked against the positive set to remove AMP sequences that lack the annotation in UniProtKB/Swiss-Prot. Finally, 4173 sequences were sampled from the remaining set of 128,445 non-AMPs, matching the number and length distribution of sequences in the positive set" (Li et al., 2022). 

## Classifying Peptides with Long-Short-Term-Memory (LSTM)
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) designed to effectively capture long-term dependencies in sequential data. Traditional RNNs struggle with long-term dependencies due to issues like vanishing and exploding gradients. LSTMs address these issues through a more complex architecture that includes mechanisms specifically intended to manage long-term dependencies.
### How LSTMs Work:
**Forget Gate**: Looks at the previous hidden state and the current input, and outputs a number between 0 and 1 for each number in the cell state. A value of 1 means "completely keep this," while a value of 0 means "completely get rid of this."  

**Input Gate**: Decides which values from the input will update the cell state. It consists of two parts:  
A sigmoid layer that decides which values to update and a tanh layer that creates a vector of new candidate values.

**Cell State Update**: The cell state is updated by combining the old cell state (scaled by the forget gate) and the new candidate values (scaled by the input gate).  

**Output Gate**: Determines the next hidden state based on the updated cell state and the current input.

<img src="https://github.com/humzaashraf1/NLP-with-Antimicrobials/assets/121640997/3ae2fa3a-c8cb-4ea3-a1b7-5e7ba150fea0" alt="LSTM" width="550" height="200">

Graphic courtesey of StatQuest (https://statquest.org/) 

### Implementing LSTM for Classification:
LSTM was able to achieve ~85% accuracy for classifying peptide sequences as AMPs vs. non-AMPs on an evaluation set. Model parameters were saved along with the notebook under <ins>**LSTM_AMP_Classification**</ins>.

**Example Usage**  
From the command line:  
PS FOLDER> & //WindowsApps/python3.10.exe //predict_sequence.py  
Enter the amino acid sequence: MNAELIVQLGSLALITVAGPAIIVLLFLKQGNL  
Prediction for sequence 'MNAELIVQLGSLALITVAGPAIIVLLFLKQGNL': [0.02103578 0.9789484 ]  
Ground truth: [0 1], where the left-side class is AMP and the right-side class is non-AMPs.

## Classifying Peptides with Bidirectional Encoder Representations from Transformers (BERT)  
BERT is a powerful natural language processing (NLP) model introduced by researchers at Google in 2018. BERT is based on the Transformer architecture, which relies on self-attention mechanisms to capture the relationship between words in a sentence. This architecture allows BERT to process and understand context from both directions. BERT randomly masks some of the words in the input sentence and trains the model to predict the masked words based on the context provided by the surrounding words. After pretraining, BERT can be fine-tuned on downstream NLP tasks with labeled data. In this example, we will fine-tune a pre-trained BERT model that is publically available for sequence classification (https://huggingface.co/Rostlab/prot_bert). The pre-trained model, also called ProtTrans, was trained on nearly 400 billion amino acids from UniRef and BFD (see the publication: doi: https://doi.org/10.1101/2020.07.12.199554).

### How Transformers are used in BERT  
**Self-Attention Mechanism**: The core component of transformers is the self-attention mechanism, which allows the model to weigh the importance of each word in a sentence based on its relevance to other words. This mechanism enables the model to capture long-range dependencies and understand the contextual relationships between words (https://arxiv.org/abs/1706.03762).   

**Input Representation**: BERT tokenizes input text into subword units using WordPiece tokenization. Each token is then represented as an embedding vector, which is the input to the transformer.  

**Multi-Head Self-Attention**: This sub-module computes multiple attention scores in parallel, allowing the model to capture different aspects of the input sequence. Each attention head learns different patterns and dependencies in the data.  

**Feed-Forward Neural Network**: After computing attention scores, the model passes the output through a feed-forward neural network to capture nonlinear relationships between words.  

**Contextualized Representations**: The output of the transformer layers is a set of contextualized representations for each token in the input sequence. These representations capture the contextual information of each token within the context of the entire sentence. 

<img src="https://github.com/humzaashraf1/NLP-with-Antimicrobials/assets/121640997/57da8e73-5753-4b7f-b53b-101fd62c5ff6" alt="Transformer" width="450" height="400">  

Graphic courtesey of StatQuest (https://statquest.org/)

### Implementing BERT for Classification:  
BERT was able to achieve ~88% accuracy for classifying peptide sequences as AMPs vs. non-AMPs on an evaluation set. While this is a relatively minor improvement compared to LSTM, the BERT model has greater sequence context, which we can visualize with dimensionality reduction. Model parameters were not saved due to storage constraints, but the .pth file can be easily generated on Google Colab. All code is located within <ins>**BERT_AMP_Classification**</ins>.

**Example Usage**  
From the command line:  
PS FOLDER> & //WindowsApps/python3.10.exe //predict_sequence.py  
Enter a protein sequence: MNAELIVQLGSLALITVAGPAIIVLLFLKQGNL  
The predicted label for the sequence is: non-AMP    
Ground truth: [0 1], where the left-side class is AMP and the right-side class is non-AMPs.

### Visualizing BERT Embeddings with t-Distributed Stochastic Neighbor Embedding (t-SNE)  
t-SNE is a dimensionality reduction technique that allows n-dimensional vectors to be projected down into 2-dimensional space based on relative similarity in high-dimensional space. Since BERT essentially produces high dimensional embedding vectors, we can visualize "similarity" between peptide sequences by comparing the relative distances between data points in the projected space. To achieve this, we simply pass our tokenized sequences to the model in small batches and then compute the mean of the entire tokenized sequence (using "outputs.last_hidden_state.mean(dim=1)") in the last hidden layer, rather than proceeding to the final fully-connected layer (see visualize_embeddings.py).    

**BERT Before Fine Tuning**  
<img src="https://github.com/humzaashraf1/NLP-with-Antimicrobials/assets/121640997/c7f4333c-4dfa-4c5e-a45f-c8d5e0c75b2f" alt="tSNE Before Fine Tuning" width="400" height="300">  
**BERT After Fine Tuning**  
<img src="https://github.com/humzaashraf1/NLP-with-Antimicrobials/assets/121640997/975fdda8-1662-49fc-b062-9c1e36c4c0fc" alt="tSNE Before Fine Tuning" width="400" height="300">  
As can be seen in the above scatter plots, fine tuning the pre-trained model increased the separation between embeddings in both the training and testing sets, suggesting our training was successful!

## Folding Peptides with Evolutionary-Scale-Modeling-Fold (ESM-Fold) 
ESM-Fold, developed by researchers at Meta in 2022, is a protein folding model that stands out from its Google counterpart AlphaFold by not relying on multiple sequence alignments for its predictions. This approach allows ESM-Fold to efficiently fold hundreds to thousands of protein sequences without compromising accuracy. The model's remarkable performance is driven by ESM-2, a large language model similar to BERT, with approximately 15 billion parameters. At this scale, ESM-2's attention scores effectively correlate with contact maps between protein residues, thereby enabling accurate structure prediction purely from masked-language training (https://www.science.org/doi/abs/10.1126/science.ade2574).   

### Bulk Folding Peptides
One straightforward approach to scanning many thousands of peptide sequences for their structure is to extract relevant information from each folded sequence and then to write the metadata to a new spreadsheet. In this case, we can extract information about per-residue stability as well as per-residue secondary structure. In ESM_Fold_Bulk.ipynb within <ins>**ESMFold_AMPs**</ins> (an adaptation of a colab notebook from https://github.com/facebookresearch/esm?tab=readme-ov-file) we simply pass an excel spreadsheet in the following format:  

| ID | Sequence                                   |
|----|--------------------------------------------|
| 99 | MGAIAKLVAKFGWPFIKKFYKQIMQFIGQGWTIDQIEKWLKRH |

The code then returns the following:  
| ID | Folded                                   |
|----|------------------------------------------|
| 99 | -FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF |  

| ID | Secondary Structures                       |
|----|--------------------------------------------|
| 99 | -HHHHHHHHHH-HHHHHHTHHHHHHHHHTT--HHHHHHHHHT- |

Secondary Structure Key (dssp):
| Symbol | Structure                   |
|--------|-----------------------------|
| H      | Alpha Helix                 |
| G      | 3-10 Helix                  |
| I      | Pi Helix                    |
| E      | Beta Strand                 |
| B      | Beta Bridge                 |
| T      | Turn                        |
| S      | Bend                        |
| -      | Coil (no defined structure)|  

Furthermore, we can save the resulting .pdb file and then display it with PyMol in Python:  
<img src="https://github.com/humzaashraf1/NLP-with-Antimicrobials/assets/121640997/768ce098-35d1-45f1-9831-d62b16888688" alt="AMP # 99" width="250" height="250">  

The structure predicted was color-coded based on the per-residue pLDDT score, which serves as an indicator of confidence in the local structure. Higher confidence is represented by more blue coloring relative to experimental structures, while lower confidence is indicated by more red. In the Folded column of the ESM_Fold_Bulk output, 'F' denotes a per-residue confidence score of 0.7 or higher, while '-' signifies a score below 0.7. Thus, the pLDDT score can be likened to a quasi-temperature score (or B-factor) in crystallography, providing a measure of the degree of thermal motion or disorder of atoms within a crystal. Furthermore, the structure was predicted to fold into a series of alpha-helices, as indicated by our Secondary Structure prediction column, which follows a similar naming convention as the Folded column, where 'H' represents alpha-helices and '-' represents no annotated structural elements.  

### Cross Checking ESM-Fold with AlphaFold-3  
One way to validate our structure prediction is to cross check with AlphaFold. Since AlphaFold-3 is now available for non-commercial use (https://golgi.sandbox.google.com/about), I simply pasted the same sequence into their server to generate a structure:  
<img src="https://github.com/humzaashraf1/NLP-with-Antimicrobials/assets/121640997/16f78351-2607-4406-aa63-1c024eeb5cc1" alt="AlphaFold" width="550" height="250">  

## Generative Protein Design with a Message-Passing-Neural-Network (ProteinMPNN)  
ProteinMPNN is a deep learning model designed to generate protein sequences that fold into specific three-dimensional structures. The model uses an attention-based message-passing mechanism, where proteins are represented as graphs, with nodes corresponding to amino acid residues and edges representing bonds between residues. In this type of network, nodes exchange information with their neighbors through edges in an iterative process called message passing. During each iteration, nodes update their states based on the information received from their neighboring nodes. The attention mechanism enhances the message passing by assigning different weights to the incoming messages from neighboring nodes. This allows the network to focus on more relevant neighbors when updating each node's representation. For a more detailed description, you can watch this excellent YouTube video by DeepFindr (https://www.youtube.com/watch?v=A-yKQamf2Fc).  
 ![image](https://github.com/humzaashraf1/NLP-with-Antimicrobials/assets/121640997/31334d79-4052-4f79-8e27-0aef348c7251)  
 (Graphic courtesey of DeepFindr).  

### Constraints on Sequence Generation with ProteinMPNN  
When passing an amino-acid sequence to ProteinMPNN, the model generates a new protein that folds into the same 3D shape. For further customizability, there are several optional input arguments that constrain the output sequence. Here are three interesting constraints to consider from the argparser:

1) AA composition bias (E.g., make certain residues more or less likely to occur).
2) Bias by residue (E.g., positional biases that make certain residues more or less likely at specific positions in the sequence).
3) Omit AA bias (E.g., eliminate certain residues altogether in the designed chain).

The encoder uses unmasked self-attention, while the decoder uses masked self-attention (thus the generation of each residue is auto-regressive--where it occurs sequentially from left to right). For each residue, the amino acid with the highest probability is selected. Probabilities for all the amino acids add up to 1.0 because the output logits are run through a softmax function (softmax(z)<sub>i</sub> = e<sup>z<sub>i</sub></sup> / ∑<sub>j</sub>e<sup>z<sub>j</sub></sup>). The optional inputs essentially just modify the probabilities for each residue during sequence generation: probs = F.softmax((temperature * logits - OmitBias * 10^8 + CompBias + temperature * ResidueBias) / temperature). Thus, we can restrict the model to generate sequences from either rational design or experimentally derived insights. 

### Running ProteinMPNN  
I cloned the official GitHub repo (https://github.com/dauparas/ProteinMPNN) and created a bash script to call on "protein_mpnn_run.py". In this case, we will use the same AMP from the ESM-Fold example: MGAIAKLVAKFGWPFIKKFYKQIMQFIGQGWTIDQIEKWLKRH. I generated the corresponding .pdb file from ESM-Fold and named it AMP_99. To get started, we only need to pass the following input arguments to the command line: --model_name --path_to_model_weights --out_folder --pdb_path. I additionally included --seed 37, since the authors of the paper seem to always use that in their examples (if no seed argument is given, a random seed is chosen).
 
<sub> Portions of code in this repository were generated with the assistance of ChatGPT, a LLM developed by OpenAI.</sub>
