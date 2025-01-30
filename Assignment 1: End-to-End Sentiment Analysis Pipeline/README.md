# Sentiment Analysis Pipeline (BERT + Flask)

## Overview
This project implements an end-to-end sentiment analysis pipeline using a fine-tuned BERT model. The model is trained on the IMDB dataset for binary classification (positive/negative sentiments). A Flask-based API serves predictions, and a test script validates the API endpoints.

## Project Features
- Fine-tunes `BertForSequenceClassification` on the IMDB dataset.
- Performs Exploratory Data Analysis (EDA) and visualization.
- Stores processed data in an SQLite database.
- Deploys a Flask API for sentiment prediction.
- Includes a test script for API validation.

---
## Project Setup
### 1. Install Dependencies
Run the following command to install all required libraries:
```sh
pip install -r requirements.txt
```

### 2. Set Up the Database
The dataset is stored in an SQLite database. The training script creates and populates the database automatically. Ensure the script is executed before running the Flask app.

```sh
python train_model.py  # This will create 'imdb_dataset.db'
```

---
## Data Acquisition
We use the IMDB dataset for training and testing:
- The dataset is loaded using the `datasets` library:
  ```python
  from datasets import load_dataset
  dataset_dict = load_dataset("imdb")
  ```
- The dataset contains 25,000 training and 25,000 test samples.
- Data is cleaned and stored in an SQLite database (`imdb_dataset.db`).

---
## Run Instructions
### 1. Train the Model
Execute the following command to fine-tune the BERT model:
```sh
python train_model.py
```
The script performs data cleaning, EDA, training, and evaluation.

### 2. Start the Flask Server
Run the following command to launch the Flask API:
```sh
python app.py
```
The server will be accessible at `http://127.0.0.1:5000/`.

### 3. Test the API
Use the provided test script to verify the API:
```sh
python test_api.py
```
Alternatively, you can send a manual request using `curl`:
```sh
curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d '{"review_text": "This movie was amazing!"}'
```
Expected response:
```json
{
  "sentiment_prediction": "positive"
}
```

---
## Model Information
### Model Architecture
- **Base Model:** `textattack/bert-base-uncased-yelp-polarity`
- **Fine-Tuned for:** Binary sentiment classification (positive/negative)
- **Optimizer:** AdamW
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 16
- **Epochs:** 3

### Performance Metrics
| Metric     | Value  |
|------------|--------|
| Accuracy   | 92.4%  |
| Precision  | 89.2%  |
| Recall     | 95.3%  |
| F1 Score   | 92.1%  |

---
## Folder Structure
```
├── data/                     # Dataset and database
│   ├── imdb_dataset.db        # SQLite database
├── models/                   # Trained models
│   ├── bert_model/           # Fine-tuned BERT model
├── scripts/                  # Training and testing scripts
│   ├── train_model.py        # Model training script
│   ├── test_api.py           # API testing script
├── app.py                     # Flask API server
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
```

---
## Future Improvements
- Enhance model performance with additional training epochs.
- Deploy API to a cloud platform (AWS/GCP/Azure).
- Improve text preprocessing with advanced NLP techniques.
- Expand dataset to include more diverse sentiment categories.

---
## Contributors
- **Your Name**  
- Feel free to contribute via pull requests!

## License
MIT License

