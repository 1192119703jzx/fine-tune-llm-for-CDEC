import torch
from transformers import RobertaForSequenceClassification
from Tokenizer import len_tokenizer, tokenizer_used
from peft import LoraConfig, get_peft_model

special_tokens = ["<TRG>", "</TRG>", "<ARG>", "</ARG>", "<TIME>", "</TIME>", "<LOC>", "</LOC>"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(pretrained_model_name, fine_tune=False):
    # Create model
    if fine_tune:
        try:
            model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2, load_in_8bit=True)
        except RuntimeError as e:
            print(f"Error using 8-bit quantization: {e}")
            print("Falling back to standard precision.")
            model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)
    else:
        model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=2)

    model.to(device)

    # Resize token embeddings to accommodate new special tokens
    model.resize_token_embeddings(len_tokenizer)

    # Get the model's embedding layer
    embedding_layer = model.roberta.embeddings.word_embeddings
    # Get token IDs for new special tokens
    special_tokens_ids = tokenizer_used.convert_tokens_to_ids(special_tokens)
    # Initialize special token embeddings
    with torch.no_grad():
        for token_id in special_tokens_ids:
            embedding_layer.weight[token_id] = torch.mean(embedding_layer.weight, dim=0)

    if fine_tune:
        lora_config = LoraConfig(
            r=8,                # LoRA rank (tradeoff between efficiency and performance)
            lora_alpha=32,      # LoRA scaling factor
            lora_dropout=0.05,   # Dropout to prevent overfitting
            target_modules=["query", "value"],  # Apply LoRA to attention layers
            bias="none"
        )

        model = get_peft_model(model, lora_config)
        #model.print_trainable_parameters()

    return model
    # Save the model
    #model.save_pretrained('../models/roberta-base-special-tokens')

#

