from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import shap
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

model_path = "./models/BERT_Multi-Label_classification"  # Path to your saved model
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
label_map = {'toxic':0, 'severe_toxic':1, 'obscene':2, 'threat':3, 'insult':4, 'identity_hate':5}
# Create a pipeline for multi-label classification
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device,  # Use GPU if available
    return_all_scores=True  # Return scores for all labels
)

# def predict_proba(texts):
#     encoded_inputs = [
#         tokenizer.encode(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
#         for text in texts
#     ]
#     tokens = {
#         'input_ids': torch.cat([inp for inp in encoded_inputs]),
#         'attention_mask': torch.cat([torch.tensor(inp != 0, dtype=torch.int64) for inp in encoded_inputs])
#     }

#     with torch.no_grad():
#         logits = model(**tokens).logits
#         probs = torch.sigmoid(logits).cpu().numpy()  # shape: (batch_size, num_labels)
#     return probs

masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(classifier, output_names=list(label_map.keys()))