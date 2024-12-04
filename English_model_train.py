## Rest of the code remains the same only the pretrained model and data processing changes from the Hindi_train.py 
"""
Loading the pretrained llama model in quantized form so as to make it compatiable with the gpu 
"""
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Configure BitsAndBytes for 8-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit quantization
    bnb_4bit_quant_type="nf4",  # Use nf4 quantization type
    bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for computation
)

# Load the processor and model
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
model = AutoModelForImageTextToText.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    quantization_config=bnb_config,  # Apply the quantization configuration
    device_map="auto",  # Automatically assign the model to the appropriate device
    trust_remote_code=True,  # Allow loading models with custom code
    torch_dtype=torch.bfloat16,  # Use bfloat16 data type for the model
)

"""
preparing the dataset
"""
from huggingface_hub import login
login()
from datasets import load_dataset
ds = load_dataset("santoshtyss/indian_courts_cases")
train_data=train_data.to_pandas()
validation_data=validation_data.to_pandas()
"""
Preprocessing the dataset
"""
# tokeinizing the model and preprocessing the input data 
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
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
