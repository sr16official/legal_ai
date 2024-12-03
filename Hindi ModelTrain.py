#Packages To Import
import numpy as np
import pandas as pd
import os
import transformers
import torch
# from IPython import get_ipython
# from IPython.display import display
from datasets import load_dataset
from huggingface_hub import login
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import inspect
from evaluate import load
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn import functional as F
import torch._inductor.config as config
from transformers import AutoTokenizer

from mpi4py import MPI
import deepspeed
import gradio as gr

from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from indicnlp.tokenize import indic_tokenize
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, BitsAndBytesConfig, AutoTokenizer

from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from accelerate import infer_auto_device_map

## Checking the colab Gpu availability 
if 'COLAB_GPU' in os.environ:
    from google.colab import output
    output.enable_custom_widget_manager()

"""
Make sure to login into the hugging face library 
#login()
"""

"""
Downloading the model weights  from hugging face to finetune the model weigths on my data since the model itsellf is of 32 gb the amount 
og gpu required is high and i do not have that much of gpu available so to make use of the most of the resources i have, i downlaoded the 
model into the 8 bit config by using the transformers BitsAndBytesConfig library, by making tradeoff between the accuracy and the size
"""
## Model configuration and download 
from huggingface_hub import hf_hub_download
model_path = "MBZUAI/Llama-3-Nanda-10B-Chat"

# Simplified BitsAndBytesConfig without unrecognized parameters
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.bfloat16
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
)

"""
Since i have been training the model on the google colab and using the A100 connection type i have predefined the gpu distribution.

"""

device_map = infer_auto_device_map(
     model,
     max_memory={0: "20GB", "cpu": "20GB"}  # Reserve ~10GB on GPU for training operations
)

max_memory = {0: "20GB", "cpu": "20GB"}
device_map = infer_auto_device_map(
    model,
    max_memory=max_memory

"""
Checking the size of the quantized model 
"""
model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
model_size_gb = model_size_bytes / (1024 * 1024 * 1024)
print(f"Model size: {model_size_gb:.2f} GB")
print(model_size_gb)

"""
Loading the dataset from the hugging face library 
"""
ds = load_dataset("Exploration-Lab/IL-TUR", "bail")


## splitting the dataset into train test and validation split for the training 
train_specific = ds['train_specific'].to_pandas()
dev_specific = ds['dev_specific'].to_pandas()
test_specific = ds['test_specific'].to_pandas()

train_all = ds['train_all'].to_pandas()
dev_all = ds['dev_all'].to_pandas()
test_all = ds['test_all'].to_pandas()

train_df = pd.concat([train_all, train_specific], ignore_index=True)
validation_df = pd.concat([dev_all, dev_specific], ignore_index=True)
test_df = pd.concat([test_all, test_specific], ignore_index=True)

"""
Low-Rank Adaptation.
It's a technique used in deep learning, particularly for fine-tuning large language models (LLMs).
"""
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)

"""
Preprocessing the dataset
"""

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):  # Updated to accept encodings and labels
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        try:
            idx = int(idx)  # Convert idx to an integer
        except (TypeError, ValueError):
            raise TypeError(f"Invalid index type: {type(idx)}. Index must be an integer.")

        # Accessing the data using its keys (e.g., 'input_ids', 'attention_mask', 'labels')
        if idx < 0 or idx >= len(self.labels): # Updated to use len(self.labels)
            raise IndexError("Index out of range")

        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long) # Ensure labels are tensors

        return item

    def __len__(self):
        return len(self.labels)  # Updated to use len(self.labels)

from torch.nn.utils.rnn import pad_sequence


def collate_fn(features):
    """Collate function to handle padding and batching of data."""

    # Pad input sequences (text) to maximum length
    text_features = [feature['input_ids'] for feature in features]
    max_length = max((len(text) if isinstance(text, list) and text else text.shape[0] if isinstance(text, torch.Tensor) else 0) for text in text_features)

    if max_length == 0: # Handle the case where max_length is 0 due to empty/0-d tensors
        max_length = 1

    # Pad text features with -100
    padded_text = [
        torch.cat([torch.tensor(text), torch.tensor([-100] * (max_length - (len(text) if isinstance(text, list) and text else text.shape[0] if isinstance(text, torch.Tensor) else 0)))])
        if isinstance(text, (list, torch.Tensor)) and (len(text) if isinstance(text, list) else text.shape[0] > 0) # If not empty, pad as usual
        else torch.tensor([-100] * max_length)  # Otherwise, pad with -100s
        for text in text_features
    ]

    # Pad labels to match the padded text length and use -100 for padding
    label_features = [feature['labels'] for feature in features]
    padded_labels = [
        torch.cat([torch.tensor(labels), torch.tensor([-100] * (max_length - len(labels)))])
        if isinstance(labels, list) and labels  # Pad only if labels is a non-empty list
        else torch.tensor([-100] * max_length)  # Otherwise, pad with -100s
        for labels in label_features
    ]

    # Stack padded features into tensors
    input_ids = torch.stack(padded_text).long()
    labels = torch.stack(padded_labels)

    # Return a dictionary with padded features
    return {'input_ids': input_ids, 'labels': labels}



# Preprocessing function
def preprocess_and_tokenize(text):
    if isinstance(text, dict):
        text = text.get('content', '')  # Extract 'content' or use empty string if not found
    tokens = indic_tokenize.trivial_tokenize(text)
    tokenized_text = " ".join(tokens)
    return tokenized_text
def prepare_dataset(df, column_text, column_label):

    df['tokenized_text'] = df[column_text].apply(preprocess_and_tokenize)
    encodings = tokenizer(
        df['tokenized_text'].tolist(),
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    labels = torch.tensor(df[column_label].tolist(), dtype=torch.long)

    # Handle missing or empty labels
    if labels.numel() == 0:
        raise ValueError("Labels tensor is empty after preprocessing.")

    return TextClassificationDataset(encodings, labels)
def validate_data(df, column_text, column_label):
    # Check for missing or empty rows
    valid_rows = df[column_text].notna() & df[column_text].str.strip().astype(bool)
    valid_labels = df[column_label].notna()
    df = df[valid_rows & valid_labels]

    if df.empty:
        raise ValueError("No valid data found after validation.")

    return df

# Apply validation before preparing datasets
train_df = validate_data(train_df, column_text="text", column_label="label")
validation_df = validate_data(validation_df, column_text="text", column_label="label")
test_df = validate_data(test_df, column_text="text", column_label="label")




# Prepare datasets
train_dataset = prepare_dataset(train_df, column_text="text", column_label="label")
validation_dataset = prepare_dataset(validation_df, column_text="text", column_label="label")
test_dataset = prepare_dataset(test_df, column_text="text", column_label="label")

# DataLoaders with custom collate function
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

"""
Lora Training setup 
"""
from peft import LoraConfig, get_peft_model


# Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of the LoRA update matrices
    lora_alpha=32,  # Scaling factor for the LoRA updates
    lora_dropout=0.05,  # Dropout probability for the LoRA layers
    bias="none",  # Bias type for the LoRA layers
    task_type="CAUSAL_LM",  # Task type for the LoRA layers
)

# Attach LoRA adapters to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def compute_loss(self, model, inputs, return_outputs, num_items_in_batch):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits

    # Ensure labels are not empty and have the correct shape
    batch_size, seq_length = logits.shape[:2]
    if labels.numel() == 0:
        raise ValueError("Labels tensor is empty.")

    if labels.shape[0] != batch_size or labels.shape[1] != seq_length:
        raise ValueError(f"Mismatch: logits shape {logits.shape}, labels shape {labels.shape}")

    # Compute loss using CrossEntropyLoss
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    return loss

def compute_metrics(eval_pred):
    """
    Computes and returns a dictionary of metrics for binary classification.
    Handles empty predictions/labels to avoid NaN values.
    """
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # Calculate predictions
    predictions = np.argmax(logits, axis=-1)

    # Filter out instances with missing data (label -100)
    valid_indices = np.where((labels != -100).all(axis=1))[0]
    filtered_predictions = predictions[valid_indices]
    filtered_labels = labels[valid_indices]

    # Handle empty predictions/labels
    if len(filtered_predictions) == 0 or len(filtered_labels) == 0:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

    # Calculate metrics using evaluate library
    accuracy = evaluate.load("accuracy").compute(predictions=filtered_predictions, references=filtered_labels)
    precision = evaluate.load("precision").compute(predictions=filtered_predictions, references=filtered_labels, average="binary")
    recall = evaluate.load("recall").compute(predictions=filtered_predictions, references=filtered_labels, average="binary")
    f1 = evaluate.load("f1").compute(predictions=filtered_predictions, references=filtered_labels, average="binary")

    # Return the metrics
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"],
    }
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract labels and ensure they are long tensors
        labels = inputs.pop("labels").long()

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # Assuming the model's output has 'logits'

        # Ensure logits and labels are reshaped correctly for CrossEntropyLoss
        num_classes = logits.shape[-1]
        logits = logits.view(-1, num_classes)
        labels = labels.view(-1)

        # Compute loss
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits, labels)
        loss.requires_grad = True
        # Return loss (and optionally outputs)
        return (loss, outputs) if return_outputs else loss

    def patched_get_global_norm_of_tensors(input_tensors: List[torch.Tensor], norm_type=2, mpu=None, use_graph=False, moe_ep_group=None):
        if mpu is None:
            mpu = dist  # Assuming 'dist' is your distributed module (e.g., torch.distributed)
        compute_buffer = []

        if norm_type == float('inf'):
            global_grad_norm = torch.zeros(1, dtype=torch.float32, device=input_tensors[0].device)
            for tensor in input_tensors:
                local_grad_norm = torch.max(tensor.abs()).float()  # type: ignore
                global_grad_norm = torch.max(global_grad_norm, local_grad_norm)
            if mpu.get_data_parallel_world_size() > 1:
                mpu.all_reduce(global_grad_norm, op=torch.distributed.ReduceOp.MAX)
            return global_grad_norm[0]
        if use_graph:
            global_grad_norm = torch.zeros(1, dtype=torch.float32, device=input_tensors[0].device)
            for tensor in input_tensors:
                if tensor.data_ptr() == 0 or tensor.numel() == 0:
                    continue
                local_grad_norm = torch.norm(tensor, norm_type).float()  # type: ignore
                if moe_ep_group is not None:
                    local_grad_norm = local_grad_norm.half()  # type: ignore
                    local_grad_norm = moe_ep_group.reduce(local_grad_norm)  # type: ignore
                    local_grad_norm = local_grad_norm.float()  # type: ignore
                global_grad_norm += local_grad_norm ** norm_type
            if mpu.get_data_parallel_world_size() > 1:
                mpu.all_reduce(global_grad_norm, op=torch.distributed.ReduceOp.SUM)
            return global_grad_norm[0] ** (1.0 / norm_type)
        else:
            # The original get_global_norm_of_tensors function calls _norm_tensors before checking
            # if compute_buffer is empty.  If any of the input_tensors is empty, then this
            # results in an empty compute_buffer.  To avoid this, let's check if the input_tensors
            # are empty first and return 0 if so.
            if not input_tensors:
                return torch.tensor(0., dtype=torch.float32, device=torch.device('cpu'))
            try:
                original_norm_tensors(input_tensors, compute_buffer, norm_type, mpu, use_graph, moe_ep_group=None)
            except Exception as e:  # Catch any exceptions and print
                print(f"An error occurred in get_global_norm_of_tensors: {e}")
                return torch.tensor(0., dtype=torch.float32, device=torch.device('cpu'))  # Handle appropriately

"""
Defining the training args 
"""
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/output",
    per_device_train_batch_size=1, # change the batch size according to you gpu capability 
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,

    bf16=True,
    # deepspeed="/content/ds_config.json", # in the configure optimizer i have also written the code for the deepspeed optimization the zero redundency one you can check it out  
    optim="adamw_torch_fused",
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    adam_beta1=0.9,
    logging_strategy = "epoch",
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    label_smoothing_factor=0.1,
    logging_dir="/content/drive/MyDrive/logging_dir",
    gradient_checkpointing=True,
    resume_from_checkpoint="/content/drive/MyDrive/output/epoch_"
)
torch.set_autocast_enabled(False)

# Initialize GradScaler
scaler = torch.cuda.amp.GradScaler()



def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
    # Create an empty list to store parameter groups
    optim_groups = []

    # Manually add parameters to optimizer groups if they require gradients
    for n, p in self.named_parameters():
        if p.requires_grad:
            group = {'params': [p]}
            if p.dim() >= 2:  # Apply weight decay based on dimension
                group['weight_decay'] = weight_decay
            else:
                group['weight_decay'] = 0.0
            optim_groups.append(group)

    # Check if any parameters require gradients
    if not optim_groups:
        print("Warning: No parameters require gradients. Returning DeepSpeedCPUAdam optimizer.")
        # Handle the case where optim_groups is empty by returning a default optimizer
        return DeepSpeedCPUAdam(self.parameters(), lr=learning_rate, betas=betas)


    # Calculate the number of parameters in each group (optional)
    num_decay_params = sum(p.numel() for p in [g['params'][0] for g in optim_groups if 'weight_decay' in g and g['weight_decay'] > 0])
    num_nodecay_params = sum(p.numel() for p in [g['params'][0] for g in optim_groups if 'weight_decay' in g and g['weight_decay'] == 0])

    # Create and return the optimizer (AdamW in this case)
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    print(f"using fused AdamW: {use_fused}")


# def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
#     # Create an empty list to store parameter groups
#     optim_groups = []

#     # Manually add parameters to optimizer groups if they require gradients
#     for n, p in self.named_parameters():
#         if p.requires_grad:
#             group = {'params': [p]}
#             if p.dim() >= 2:  # Apply weight decay based on dimension
#                 group['weight_decay'] = weight_decay
#             else:
#                 group['weight_decay'] = 0.0
#             optim_groups.append(group)

#     # Check if any parameters require gradients
#     if not optim_groups:
#         print("Warning: No parameters require gradients. Returning DeepSpeedCPUAdam optimizer.")
#         # Handle the case where optim_groups is empty by returning a default optimizer
#         return DeepSpeedCPUAdam(self.parameters(), lr=learning_rate, betas=betas)


#     # Calculate the number of parameters in each group (optional)
#     num_decay_params = sum(p.numel() for p in [g['params'][0] for g in optim_groups if 'weight_decay' in g and g['weight_decay'] > 0])
#     num_nodecay_params = sum(p.numel() for p in [g['params'][0] for g in optim_groups if 'weight_decay' in g and g['weight_decay'] == 0])

#     # Create and return the optimizer (AdamW in this case)
#     fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
#     use_fused = fused_available and device_type == 'cuda'
#     print(f"using fused AdamW: {use_fused}")

#     # **Disable DeepSpeed ZeRO stage 3 for this optimizer**
#     zero_stage = 0  # Set zero_stage to 0 or a lower stage as needed

#     # Choose optimizer based on zero_stage
#     if zero_stage == 1:
#         print("using ZeroRedundancyOptimizer")
#         optimizer = ZeroRedundancyOptimizer(
#             **optim_groups[0],  # Pass the first group as kwargs
#             optimizer_class=torch.optim.AdamW,
#             lr=learning_rate,
#             betas=betas,
#             fused=use_fused  # Pass fused argument
#         )
#         # Add the remaining groups
#         for group in optim_groups[1:]:
#             optimizer.add_param_group(group)
#     else:
#         print("using regular AdamW")
#         optimizer = torch.optim.AdamW(
#             optim_groups,
#             lr=learning_rate,
#             betas=betas,
#             fused=use_fused  # Pass fused argument
#         )

#     return optimizer
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and being used.")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU instead.")

configure_optimizers= configure_optimizers(model, training_args.weight_decay, training_args.learning_rate, (training_args.adam_beta1, training_args.adam_beta2), training_args.device, 0)
trainer =CustomTrainer(
    model=model.to(device),
    args=training_args,
    train_dataset=train_dataset ,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
    optimizers=(configure_optimizers, None),
    tokenizer=tokenizer,
    data_collator=collate_fn,
    callbacks=[SaveToDriveCallback()],
)


trainer.train()



