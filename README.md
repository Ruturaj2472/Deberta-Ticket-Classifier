# Customer Support Ticket Tagger

Fine-tunes a **DeBERTa-v3** transformer model to automatically classify customer support tickets into one of **10 categories** — such as Billing and Payments, Technical Support, Returns and Exchanges, and more — using Hugging Face `transformers`, `datasets`, and the `Trainer` API.

Unlike prompt-based LLM classification (calling GPT/Gemini/Claude with a prompt), this project **trains its own dedicated classification model** from a labeled dataset, producing a small, fast, self-contained model that can be deployed without depending on any external API.

## What It Does

1. Loads a labeled dataset of customer support tickets (text + category)
2. Encodes text category labels into numeric IDs
3. Splits the data into train/test sets
4. Loads a pretrained `microsoft/deberta-v3-base` model and tokenizer
5. Tokenizes the ticket text
6. Fine-tunes the model on the labeled tickets using Hugging Face's `Trainer`
7. Evaluates the fine-tuned model
8. Saves the trained model and tokenizer locally
9. Loads the saved model back through a `text-classification` pipeline
10. Runs predictions on sample tickets and compares predicted vs. actual labels

## Ticket Categories

The model classifies tickets into these 10 categories:

```
Billing and Payments
Customer Service
General Inquiry
Human Resources
IT Support
Product Support
Returns and Exchanges
Sales and Pre-Sales
Service Outages and Maintenance
Technical Support
```

## Dataset

The notebook expects a CSV file (`customer_tickets.csv`) with two columns: raw ticket text and a category label.

> **Note:** The dataset is not included in this repo. Provide your own CSV in the same two-column format and point the notebook's file path to it.

## Project Structure

```
customer-support-ticket-tagger/
├── README.md
├── requirements.txt
├── .gitignore
└── customer-support-ticket-tagger.ipynb
```

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/customer-support-ticket-tagger.git
   cd customer-support-ticket-tagger
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Obtain the dataset (`customer_tickets.csv`) and update the file path in the notebook to point to your local copy:
   ```python
   df = pd.read_csv("path/to/your/customer_tickets.csv")
   ```
   Expected format: a CSV with two columns — ticket text and category label.

4. A GPU is strongly recommended for training — fine-tuning a transformer on CPU alone will be significantly slower.

5. Run the notebook cells from top to bottom.

## How It Works

### 1. Label Encoding

```python
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['labels'])
```
Converts text category names (e.g. `"Billing and Payments"`) into numeric IDs the model can train on, while `label_encoder.classes_` keeps the mapping so predictions can be decoded back into readable labels later.

### 2. Model and Tokenizer

```python
model_name = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))
```
Loads a pretrained DeBERTa-v3 base model with a classification head sized to match the number of ticket categories.

### 3. Fine-Tuning

```python
training_args = TrainingArguments(
    learning_rate=0.00002,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.05,
    ...
)
trainer = Trainer(model=model, args=training_args, train_dataset=..., eval_dataset=...)
trainer.train()
```
Fine-tunes the model on the tokenized training set for 10 epochs, with weight decay applied to reduce overfitting.

### 4. Saving and Reloading

```python
trainer.save_model("./text-classification-model")
tokenizer.save_pretrained("./text-classification-model")

classifier = pipeline("text-classification", model="./text-classification-model", tokenizer=tokenizer)
```
Saves the fine-tuned model to disk, then reloads it through a simple `pipeline()` call for easy inference — this is the artifact you'd actually deploy in a real application.

### 5. Prediction

```python
def predictor(input_ticket, org_label):
    result = classifier(input_ticket)
    decoded_label = label_encoder.inverse_transform([int(result[0]['label'].split("_")[-1])])[0]
    ...
```
Runs a ticket through the classifier and decodes both the predicted and original labels back into human-readable category names for easy comparison.

## Example Usage

```python
predictor(
    "Hi, I've noticed performance issues with my Dell XPS 13 9310 after the latest update. Please assist.",
    org_label=9
)
```

```
Input Ticket: Hi, I've noticed performance issues with my Dell XPS 13 9310 after the latest update. Please assist.

Original Label: 9, Original Label Decoded: Technical Support
Predicted Label: 9, Predicted label Decoded: Technical Support
```

## Requirements

- Python 3.10+
- GPU strongly recommended for training (CPU-only training will be significantly slower)
- See `requirements.txt` for exact packages

## Notes & Limitations

- The train/test split (`test_size=0.145`) is relatively small — results should be interpreted with that in mind; a larger, more balanced dataset would give more reliable evaluation metrics.
- `num_train_epochs=10` with a base-sized model can overfit on a small dataset; the `weight_decay=0.05` setting is a partial safeguard, but watch the eval loss/accuracy across epochs if you adjust this.
- This model classifies **single-label** categories only. If a ticket genuinely spans two categories (e.g. billing + technical), the model must still pick just one.
- The `<name>`, `<tel_num>`, `<acc_num>`, `<t_num>` placeholders visible in ticket examples indicate the dataset has already been anonymized/redacted upstream.

## License

MIT
