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
BERT is a powerful natural language processing (NLP) model introduced by researchers at Google in 2018. BERT is based on the Transformer architecture, which relies on self-attention mechanisms to capture the relationship between words in a sentence. This architecture allows BERT to process and understand context from both directions. BERT randomly masks some of the words in the input sentence and trains the model to predict the masked words based on the context provided by the surrounding words. After pretraining, BERT can be fine-tuned on downstream NLP tasks with labeled data. In this example, we will fine-tune a pre-trained BERT model tha tis publically available for sequence classification (https://huggingface.co/Rostlab/prot_bert). The pre-trained model, also called ProtTrans, was trained on nearly 400 billion amino acids from UniRef and BFD (see the publication: doi: https://doi.org/10.1101/2020.07.12.199554).

### How Transformers are used in BERT  
**Self-Attention Mechanism**: The core component of transformers is the self-attention mechanism, which allows the model to weigh the importance of each word in a sentence based on its relevance to other words. This mechanism enables the model to capture long-range dependencies and understand the contextual relationships between words.  

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
t-SNE is a dimensionality reduction technique that allows n-dimensional vectors to be projected down into 2-dimensional space based on relative similarity in high-dimensional space. Since BERT essentially produces high dimensional embedding vectors, we can visualize "similarity" between peptide sequences by comparing the relative distances between data points in the projected space.    

**BERT Before Fine Tuning**  
<img src="https://github.com/humzaashraf1/NLP-with-Antimicrobials/assets/121640997/c7f4333c-4dfa-4c5e-a45f-c8d5e0c75b2f" alt="tSNE Before Fine Tuning" width="400" height="300">  
**BERT After Fine Tuning**  
<img src="https://github.com/humzaashraf1/NLP-with-Antimicrobials/assets/121640997/975fdda8-1662-49fc-b062-9c1e36c4c0fc" alt="tSNE Before Fine Tuning" width="400" height="300">  
As can be seen in the above scatter plots, fine tuning the pre-trained model increased the separation between embeddings in both the training and testing sets, suggesting our training was successful!


<sub> Portions of code in this repository were generated with the assistance of ChatGPT, a LLM developed by OpenAI.</sub>
