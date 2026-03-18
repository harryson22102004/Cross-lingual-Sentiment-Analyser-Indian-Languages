import torch, torch.nn as nn
from transformers import BertTokenizer, BertModel
 
class CrossLingualSentiment(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(768, 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, n_classes))
 
    def forward(self, input_ids, attention_mask):
        _, pooled = self.bert(input_ids, attention_mask=attention_mask, return_dict=False)
        return self.classifier(pooled)
 
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = CrossLingualSentiment(3)
LABELS = ['negative', 'neutral', 'positive']
 
def predict_sentiment(text):
    enc = tokenizer(text, return_tensors='pt', truncation=True,
                    padding='max_length', max_length=128)
    with torch.no_grad():
        out = model(enc['input_ids'], enc['attention_mask'])
    return LABELS[out.argmax(-1).item()]
 
# Zero-shot cross-lingual transfer examples
samples = [
    ("I love this product, it's amazing!", "English"),
    ("यह उत्पाद बहुत अच्छा है", "Hindi"),
    ("এই পণ্যটি দারুণ", "Bengali"),
    ("ఈ ఉత్పత్తి చాలా అద్భుతంగా ఉంది", "Telugu"),
    ("இந்த தயாரிப்பு மிகவும் நன்றாக உள்ளது", "Tamil"),
    ("This is terrible quality.", "English"),
]
print("Cross-lingual sentiment analysis:")
for text, lang in samples:
    pred = predict_sentiment(text)
    print(f"  [{lang:8s}] {text[:40]:40s} → {pred}")
