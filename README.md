# Domain-Specific-LLM-Finetuning-on-Finance-Dataset-


# FinancialDataUsingLLM

# AI-Driven Question Answering System

## Overview
This repository contains the implementation of an AI-driven Question Answering System that leverages state-of-the-art machine learning techniques and NLP models to efficiently retrieve accurate answers from a large corpus of documents. This system integrates FAISS for efficient similarity search in high-dimensional spaces and utilizes models like OpenAI's GPT-3 and fine-tuned transformers for generating responses.

## Project Structure
- `FAISS_INDEX.py`: Script for managing the FAISS index and saving the vector database.
- `QnA.py`: Core script containing the model logic, including functions to generate answers and find relevant document chunks.
- `evaluation.py`: Script for generating predictions on a sample set and saving the results for metric calculations.
- `extract_urls.py`: Utility for scraping URLs from the Moneycontrol website.
- `faiss_index.pkl`: Pickle file containing preprocessed document chunks stored in a vector database format.
- `file_info.json`: JSON file containing metadata about the document chunks such as file counts and URLs.
- `finetuning.ipynb`: Jupyter notebook detailing the fine-tuning process of the models used in the project.
- `load_documents.py`: Script to display the contents of the pickle file containing the document chunks.
- `main.py`: Streamlit application script for the user interface, including functionality to display metrics graphs.
- `metrics.json`: JSON file containing computed metrics from model evaluations.
- `metrics_evaluation.py`: Script that computes metrics based on model predictions and reference answers.
- `record_metrics_data_huggingface.json`: Output data containing predictions for sample questions using the Hugging Face model.
- `record_metrics_data_openai.json`: Output data containing predictions for sample questions using OpenAI's model.
- `requirements.txt`: List of Python libraries required to run the project.
- `sample.json`: Sample data file containing questions, reference answers, and contexts used for metric calculations.

## Setup and Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ManojKumarKolli/FinancialDataUsingLLM.git
   cd FinancialDataUsingLLM/QuestionAndAnswerUsingLangchainInFinance
2. **Install Dependencies**:
Make sure Python 3.8+ is installed on your system, then run:
   ```bash
   pip install -r requirements.txt

3. **Running the Streamlit Application**:
Execute the following command to start the Streamlit interface:
   ```bash
   streamlit run main.py

### Setting Up API Keys

To ensure the project functions correctly, you'll need to obtain API keys from OpenAI and Weights & Biases. Follow these steps to set up your environment:

#### OpenAI API Key
1. **Obtain Key**:
   - Visit [OpenAI](https://platform.openai.com/signup) and sign up or log in.
   - Navigate to the API section and generate a new API key.

#### Weights & Biases API Key
1. **Obtain Key**:
   - Sign up or log in at [Weights & Biases](https://wandb.ai/).
   - Access your user settings and navigate to API keys to generate a new key.

#### Storing API Keys Securely
1. **Create a `.env` File** in your project's root directory.
2. Wand api key is used in fine-tuning. 
3. **Add the API Keys** to the `.env` file:
   ```plaintext
   open_api_key=YOUR_OPENAI_API_KEY_HERE
   WANDB_API_KEY=YOUR_WANDB_API_KEY_HERE 


## Contributions
We extend our deepest gratitude to each team member whose dedicated efforts have significantly shaped this project:

- **Manoj Kumar** ([@manojkumar](https://github.com/ManojKumarKolli)): Led the integration and optimization of the OpenAI model, managed the model fine-tuning processes, developed evaluation metrics, and contributed to human-centric metrics analysis.
- **Meghana Reddy** ([@meghanaDasireddy](https://github.com/DasireddyMeghana)): Responsible for data scraping and preprocessing, managed the FAISS indexing system, and played a key role in the evaluation of human-centric metrics.
- **Krishna Sai** ([@ananthakrishna](https://github.com/ananthakrishna4747)): Developed the Streamlit application, integrated the Hugging Face model, implemented metric plotting in the UI, and handled various evaluation processes including human-centric metrics.

## Acknowledgments
Special thanks to all mentors and advisors who provided guidance throughout the project. Their insights were invaluable in navigating the complexities of advanced NLP and system integration.

