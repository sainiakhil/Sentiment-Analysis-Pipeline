# Sentiment Analysis Pipeline (BERT + Flask)

## Overview
This project implements an end-to-end sentiment analysis pipeline using a fine-tuned BERT model. The model is trained on the IMDB dataset for binary classification (positive/negative sentiments). A Flask-based API serves predictions, and a test script validates the API endpoints.

## Project Features
- Fine-tunes `BertForSequenceClassification` on the IMDB dataset.
- Performs Exploratory Data Analysis (EDA) and visualization.
- Stores processed data in an SQLite database.
- Deploys a Flask API for sentiment prediction.
- Includes a test script for API validation.
- 
### Install Dependencies
Run the following command to install all required libraries:
```sh
!pip install transformers
!pip install datasets
!pip install evaluate
!pip install Flask
!pip install pyngrok
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
First, run the [`Model training Script | Sentiment Analysis Pipeline.ipynb`](#[section-name](https://github.com/sainiakhil/Sentiment-Analysis-Pipeline/blob/main/Assignment%201%3A%20End-to-End%20Sentiment%20Analysis%20Pipeline/Model_training_Script___Sentiment_Analysis_Pipeline.ipynb)) file. This will generate a fine-tuned model. Save this model locally or in a drive using `save_pretrained`. 

### 2. Start the Flask Server
After training, run `Flask_app.ipynb` and provide the saved fine-tuned model directory path to use it.

Run the following command to launch the Flask API:
```sh
    ngrok.connect(5000)

    # Get the public URL
    tunnels = ngrok.get_tunnels()
    ngrok_url = tunnels[0].public_url
    print(f" * Public URL: {ngrok_url}")

    # Run Flask app
    app.run(port=5000)
```
The server will be accessible at `http://127.0.0.1:5000/`.

### 3. Test the API
Finally, run `Test Script for Flask Endpoints | Sentiment Analysis Pipeline.ipynb` to check if the Flask endpoints are working correctly. Provide ngrock generated public URL and review Text.

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
| F1 Score   | 92.1%  |![data visialization 1](https://github.com/user-attachments/assets/577c4c48-43b3-4ea8-93ee-cea38824bcf4)


---
## Exploratory Data Analysis (EDA)
![EDA](https://github.com/user-attachments/assets/bc8c6fd4-1db3-479e-ae94-f242c85042d1)

## Data Visualization

### Bar Plot: Sentiment Distribution
![data visialization 1](https://github.com/user-attachments/assets/4d7f299a-01df-4968-98a3-6ebd0900857d)




### Box Plot: Review Length by Sentiment
![data visualization 2](https://github.com/user-attachments/assets/a4e33b49-8579-47ed-8654-dda7b467dac1)




