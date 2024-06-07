import torch
from transformers import AutoTokenizer, BertModel
import torch.nn as nn
import torch.nn.init as init

PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert_bfd_localization'

class ProteinClassifier(nn.Module):
    def __init__(self, n_classes):
        super(ProteinClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.classifier = nn.Sequential(nn.Dropout(p=0.4),
                                        nn.Linear(self.bert.config.hidden_size, n_classes),
                                        nn.Tanh())
        self.init_weights()
        
    def init_weights(self):
        init.xavier_uniform_(self.classifier[1].weight)
        init.constant_(self.classifier[1].bias, 0)
        
    def forward(self, input_ids, attention_mask):
        output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        return self.classifier(output.pooler_output)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

# Load the model architecture
model = ProteinClassifier(1)  # Initialize the model architecture
model.load_state_dict(torch.load("protein_classifier_model.pth"))  # Load the trained model weights

def preprocess_sequence(sequence, tokenizer, max_length):
    # Add space between characters
    sequence_with_space = ' '.join(sequence)
    encoded_sequence = tokenizer(sequence_with_space, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return encoded_sequence

def predict(sequence, model, tokenizer, max_length):
    model.eval()
    with torch.no_grad():
        encoded_sequence = preprocess_sequence(sequence, tokenizer, max_length)
        input_ids = encoded_sequence["input_ids"]
        attention_mask = encoded_sequence["attention_mask"]
        outputs = model(input_ids, attention_mask)
        probabilities = torch.sigmoid(outputs)
        probabilities = (probabilities > 0.5).float()
    
    predicted_class = "AMP" if probabilities[0] >= 0.5 else "non-AMP"
    return predicted_class

def main():
    sequence = input("Enter a protein sequence: ")
    max_length = 60  # Should match the max_length used during training
    
    # Get the predicted class label
    predicted_label = predict(sequence, model, tokenizer, max_length)
    
    print(f"The predicted label for the sequence is: {predicted_label}")

if __name__ == "__main__":
    main()