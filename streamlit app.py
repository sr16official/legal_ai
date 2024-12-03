
"""
I have assumed the user will give a copy of is chargesheet filed by the police  officials and the case proceedings and that will be very 
much lengthy so instead of directly giving the chargesheet as an input to the model i will be using a bigger model specific in summariation
which will enusre that the importand tokens are not lost while proecssing the model 
"""
# Load Hindi summarization model
summarizer_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/IndicBART")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/IndicBART")
summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)

"""
For making the streamlit app loading the pretrained model from the drive 
"""
# Load your trained bail prediction model and tokenizer
model_path = "/content/drive/MyDrive/trained_model"  # Replace with your model path
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Streamlit app
st.title("Bail Prediction App")

# Text input for case description
case_description = st.text_area("Enter Case Description (in Hindi):", "")

# Summarization
if st.button("Summarize"):
    summary = summarizer(case_description, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    st.write("**Summary:**", summary)

# Prediction
if st.button("Predict Bail"):
    if case_description:  # Check if input is provided
        # Tokenize and preprocess the summary or original text
        inputs = tokenizer(summary if summary else case_description, return_tensors="pt", truncation=True, padding=True)

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits).item()

        # Display prediction
        if predicted_class == 1:  # Assuming 1 represents "Bail Granted"
            st.write("**Prediction:** Bail Granted")
        else:
            st.write("**Prediction:** Bail Denied")
    else:
        st.write("Please enter a case description.")
"""
Making this repository live and public in hope for a better world 
"""
