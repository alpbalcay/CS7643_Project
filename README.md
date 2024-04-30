# Time Series Forecasting Project

This project aims to perform time series forecasting using various models including Darts' TFT model, N-BEATS (Nhits) model, and LSTM model. The project utilizes publicly available sales data from the retail industry to demonstrate the effectiveness of these models in predicting future sales.

## Dataset

The dataset used in this project consists of historical sales data for multiple products across different stores. You can download the dataset from the provided Google Drive link: [Sales Data](https://drive.google.com/drive/folders/1tuMYXW6TrJo95FKUMu5VGiahwFzoqf8P?usp=sharing)

The dataset is organized into CSV files containing sales data for each product and store combination.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository:
``
git clone git@github.com:alpbalcay/CS7643_Project.git``

2. Download the dataset from the provided Google Drive link and place it in the `data` directory within the project root folder.

3. Create a Conda environment using the provided `environment.yaml` file:
``
conda (mamba) env create -f environment.yaml``

4. Activate the Conda environment:
``
conda (mamba) activate cs7643-fp``

5. Open the provided Jupyter notebook (`ModelRuns.ipynb`) in Google Colab or any Jupyter-compatible environment.

6. Follow the instructions in the notebook to run the code and train the forecasting models.

## Project Structure
```
.
├── ModelRuns.ipynb
├── README.md
├── data
│   ├── LSTM_covariates.pkl
│   ├── LSTM_target.pkl
│   ├── LSTM_train.pkl
│   ├── NHiTS_covariates.pkl
│   ├── NHiTS_train.pkl
│   ├── TFT_fut_cov.pkl
│   ├── TFT_past_cov.pkl
│   ├── TFT_train.pkl
│   ├── actual_series.pkl
│   ├── holidays_events.csv
│   ├── oil.csv
│   ├── sample_submission.csv
│   ├── stores.csv
│   ├── test.csv
│   ├── train.csv
│   ├── train_pipeline.pkl
│   ├── training_transformed.pkl
│   └── transactions.csv
├── data_process.py
├── environment.yaml
├── models
│   ├── __init__.py
│   ├── __pycache__
│   ├── lstm.py
│   ├── nhits.py
│   └── tft.py
├── plots
├── preprocess
│   ├── __init__.py
│   ├── holiday_cov.py
│   ├── oil_cov.py
│   ├── promotion_cov.py
│   ├── sales.py
│   ├── time_cov.py
│   ├── train_transform.py
│   └── transaction_cov.py
├── run_lstm.py
├── run_nhits.py
├── run_tft.py
└── utils
    └── __init__.py
```
## Models Used

### 1. TFT Model

The Temporal Fusion Transformer (TFT) model, available in the Darts library, is a powerful neural network-based model specifically designed for time series forecasting tasks. It leverages attention mechanisms and autoregressive components to capture temporal patterns and make accurate predictions.

Key Features:

*    Transformer Architecture: TFT model utilizes the transformer architecture, which consists of encoder and decoder layers equipped with self-attention mechanisms. This architecture allows the model to effectively capture temporal dependencies in time series data.

*    Multi-Horizon Forecasting: TFT model supports multi-horizon forecasting, allowing users to predict multiple future time steps simultaneously. This makes it suitable for forecasting tasks with varying prediction horizons.

*    Categorical Embeddings: TFT model handles categorical variables by embedding them into continuous vector representations, enabling the model to incorporate categorical information in the forecasting process.

### 2. N-HiTS Model

The N-Hits forecasting model, available in the Darts library, is a variant of the N-BEATS architecture tailored to optimize the number of predictions (hits) within a specified tolerance level.
Key Features:

* Optimized Predictions: N-Hits model focuses on generating accurate predictions while controlling the number of hits within a predefined tolerance range.

* Tolerance Level: Users can specify a tolerance level within which the predictions should fall. This allows for fine-tuning the model's performance based on specific accuracy requirements.

* Scalable Architecture: Built upon the N-BEATS architecture, the N-Hits model inherits its scalability and flexibility, making it suitable for various forecasting tasks across different domains.

### 3. LSTM Model

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that is well-suited for time series forecasting tasks. It is capable of learning long-term dependencies and capturing sequential patterns in the data.
