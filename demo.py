import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Chargement du modèle et du tokenizer
model_name = "bert-base-uncased"  # remplace-le si tu as un autre modèle
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
model.eval()

# Fonction de prédiction des sentiments
def classify_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Adapte l’étiquette selon ton modèle
    labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    return labels[predicted_class]

# Interface de Gradio
demo = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Entrer un tweet..."),
    outputs="text",
    title="Analyse de sentiment sur les tweets réalisés relativement au Corona",
    description="Ce modèle BERT prédit le sentiment d’un tweet en sortie."
)

if __name__ == "__main__":
    demo.launch()
