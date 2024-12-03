"""
Code for evaluating the model performance 
"""
@torch.no_grad()
def evaluate_model(model, validation_df, compute_metrics, tokenizer, device="cuda"):
    model = model.to(device)

    encodings = tokenizer(
        list(validation_df['text']),
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    encodings = {key: val.to(device) for key, val in encodings.items()}

    dataset = torch.utils.data.TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(validation_df['label']).to(device)
    )

    dataloader = DataLoader(dataset, batch_size=8)

    all_predictions = []
    all_labels = []

    # Iterate through batches
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.argmax(dim=-1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Calculate metrics
    eval_pred = {'predictions': torch.tensor(all_predictions).to(device), 'label_ids': torch.tensor(all_labels).to(device)} # Changed to dictionary

    metrics = compute_metrics(eval_pred)

    return metrics
