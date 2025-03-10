from torch.utils.data import Dataset, DataLoader
from Tokenizer import Tokenizer
from model_loader import load_model
import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluation(y_pred, y_true):
    labels = [0, 1] 
    confusion_m = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Confusion Matrix: {confusion_m}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    print(f'Accuracy: {accuracy}')
    print(f'Macro Precision: {precision.mean()}')
    print(f'Macro Recall: {recall.mean()}')
    print(f'Macro F1: {f1.mean()}')


special_tokens = ["<TRG>", "</TRG>", "<ARG>", "</ARG>", "<TIME>", "</TIME>", "<LOC>", "</LOC>"]
tokenizer_name = 'roberta-base'

# Gobal variables
train_data_path = 'data/event_pairs.train'
dev_data_path = 'data/event_pairs.dev'
test_data_path = 'data/event_pairs.test'

train_srl = 'data/train_srl.json'
dev_srl = 'data/dev_srl.json'
test_srl = 'data/test_srl.json'

pretrained_model_name = 'roberta-base'
tokenizer_name = 'roberta-base'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
batch_size = 32
learning_rate = 2e-5
epochs = 1

'''
# Create dataset instances from original data without SRL
train_dataset = Tokenizer(train_data_path, load_data=False)
dev_dataset = Tokenizer(dev_data_path, load_data=False)
test_dataset = Tokenizer(test_data_path, test=True, load_data=False)

# Create dataset instances from original data with SRL (not recommended)
train_dataset = Tokenizer(train_data_path, srl=True, load_data=False)
dev_dataset = Tokenizer(dev_data_path, srl=True, load_data=False)
test_dataset = Tokenizer(test_data_path, test=True, srl=True, load_data=False)
#
'''

# load pre-process srl data from file
train_dataset = Tokenizer(train_srl, load_data=True)
dev_dataset = Tokenizer(dev_srl, load_data=True)
test_dataset = Tokenizer(test_srl, test=True, load_data=True)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# load model
model = load_model(pretrained_model_name=pretrained_model_name, fine_tune=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
model.train()
for epoch in range(epochs):
    print("Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0.0
    for i, batch in enumerate(tqdm(train_loader, desc="Training", unit="batch")):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device).squeeze(dim=1)
        attention_mask = batch['attention_mask'].to(device).squeeze(dim=1)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % 300 == 0:
            print(f"Step {i}, Loss: {loss.item()}")
    
    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")

# Save the final model
torch.save(model.state_dict(), 'final_model.pth')

# dev loop and evaluation
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    dev_loss = 0.0
    for i, batch in enumerate(tqdm(dev_loader, desc="Dev", unit="batch")):
        input_ids = batch['input_ids'].to(device).squeeze(dim=1)
        attention_mask = batch['attention_mask'].to(device).squeeze(dim=1)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        dev_loss += outputs.loss.item()
        if i % 300 == 0:
            print(f"Step {i}, Loss: {outputs.loss.item()}")

        preds = torch.argmax(outputs.logits, dim=-1)
        all_labels.extend(batch['label'].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    print(f"Dev Loss: {dev_loss / len(dev_loader)}")

# Evaluation
evaluation(all_preds, all_labels)

# test loop and evaluation
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    test_loss = 0.0
    for i, batch in enumerate(tqdm(test_loader, desc="Test", unit="batch")):
        input_ids = batch['input_ids'].to(device).squeeze(dim=1)
        attention_mask = batch['attention_mask'].to(device).squeeze(dim=1)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        test_loss += outputs.loss.item()
        if i % 300 == 0:
            print(f"Step {i}, Loss: {outputs.loss.item()}")

        preds = torch.argmax(outputs.logits, dim=-1)
        all_labels.extend(batch['label'].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    print(f"Dev Loss: {test_loss / len(test_loader)}")

# Evaluation
evaluation(all_preds, all_labels)