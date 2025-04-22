from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import shap
import onnxruntime as ort
import onnx
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

label_map = {'toxic':0, 'severe_toxic':1, 'obscene':2, 'threat':3, 'insult':4, 'identity_hate':5}

vanilla_multi_label_classification = f"./models/BERT_Multi-Label_classification"  # Path to your saved model
vanilla_model = AutoModelForSequenceClassification.from_pretrained(vanilla_multi_label_classification, num_labels=len(label_map.keys()),hidden_dropout_prob=0.1)
vanilla_tokenizer = AutoTokenizer.from_pretrained(vanilla_multi_label_classification)
#vanilla_binary_classification = "./models/BERT_Binary_classification"  # Path to your saved model
simCSE_multi_label_classification = "simCSE_models\simcse_multi"
simCSE_model = AutoModelForSequenceClassification.from_pretrained(simCSE_multi_label_classification, num_labels=len(label_map.keys()), hidden_dropout_prob=0.1)
simCSE_multi_tokenizer = AutoTokenizer.from_pretrained(simCSE_multi_label_classification)
simCSE_model.resize_token_embeddings(len(simCSE_multi_tokenizer)) 

simCSE_binary_classification = "simCSE_models\simcse_binary"
simCSE_binary_model = AutoModelForSequenceClassification.from_pretrained(simCSE_binary_classification, num_labels=2, hidden_dropout_prob=0.1)
simCSE_binary_tokenizer = AutoTokenizer.from_pretrained(simCSE_binary_classification)
simCSE_binary_model.resize_token_embeddings(len(simCSE_binary_tokenizer))

simCSE_model.to(device)
vanilla_model.to(device)
simCSE_binary_model.to(device)

simCSE_model.eval()
vanilla_model.eval()
simCSE_binary_model.eval()

vanilla_classifier = pipeline("text-classification", model=vanilla_model, tokenizer=vanilla_tokenizer, 
                              device=device, return_all_scores=True)
simCSE_classifier = pipeline("text-classification", model=simCSE_model, tokenizer=simCSE_multi_tokenizer, 
                             device=device, return_all_scores=True)
simCSE_binary_classifier = pipeline("text-classification", model=simCSE_binary_model, tokenizer=simCSE_binary_tokenizer, 
                                     device=device, return_all_scores=True)

vanilla_explainer = shap.Explainer(vanilla_classifier,output_names=list(label_map.keys()))
simCSE_explainer = shap.Explainer(simCSE_classifier,output_names=list(label_map.keys()))
simCSE_binary_explainer = shap.Explainer(simCSE_binary_classifier,output_names=["normal","cyberbullying"])
