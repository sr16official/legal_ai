"""
Testing the Model
"""
Tokenizing the dataset 
test_encodings = tokenizer(
    list(test_df['text']),  # Assuming 'text' column contains the input text
    truncation=True,
    padding=True,
    return_tensors="pt"  # Return PyTorch tensors
)

# Create a TensorDataset for test data
test_dataset = torch.utils.data.TensorDataset(test_encodings.input_ids, test_encodings.attention_mask)

# Evaluate on test data using the Trainer
test_results = trainer.predict(test_dataset)  # Pass the preprocessed test dataset

# Print or process the test results
print(test_results.metrics)
