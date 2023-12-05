#### Model_Preprocessing.py

# Copyright (c) 2023 Landon Dahle
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# For more information on licensing, see https://opensource.org/licenses/MIT

# -----------------------------------------------------------------------------
# Author: Landon Dahle
# Date: 2023
# Project: Progress - BMEN 351 Final Project
# License: MIT
# -----------------------------------------------------------------------------

# =============================================================================
# Input Variables
# =============================================================================
import os
import time
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import re


# =============================================================================
# Function to Vectorize Text
# =============================================================================

def preprocess_text(text):
    # Ensure text is a string
    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'\W+', ' ', text)

    # Additional preprocessing steps can be added here
    return text

def vectorize_text(text):
    try:
        processed_text = preprocess_text(text)
        inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
    except Exception as e:
        print(f"Error vectorizing text: {e}")
        return None  # Return None in case of an error

# =============================================================================
# Combine CSV Files and Vectorize Text
# =============================================================================
def combine_and_vectorize(csv_directory, vector_directory):
    all_data = pd.DataFrame()
    os.makedirs(vector_directory, exist_ok=True)  # Ensure the main vector directory exists

    for file in os.listdir(csv_directory):
        if file.endswith('.csv'):
            file_path = os.path.join(csv_directory, file)
            csv_data = pd.read_csv(file_path)

            # Create a specific directory for each paper
            paper_vector_directory = os.path.join(vector_directory, os.path.splitext(file)[0])
            os.makedirs(paper_vector_directory, exist_ok=True)

            for idx, row in csv_data.iterrows():
                vector_filename = f'vector_{idx}.npy'
                vector_file_path = os.path.join(paper_vector_directory, vector_filename)

                # Vectorize and save only if it hasn't been done yet
                if 'vector_file' not in csv_data.columns or pd.isna(row.get('vector_file', None)):
                    text = row.get('Text', '')
                    vector = vectorize_text(text)

                    if vector is not None:
                        np.save(vector_file_path, vector)
                        csv_data.at[idx, 'vector_file'] = vector_file_path
                    else:
                        csv_data.at[idx, 'vector_file'] = "Error"

            all_data = pd.concat([all_data, csv_data], ignore_index=True)
            all_data.to_csv(updated_csv_path, index=False)  # Save progress

    return all_data

# =============================================================================
# Main Processing
# =============================================================================
start = time.time()

# Paths
data_directory = 'F:\\OneDrive - Texas A&M University\\BMEN 351\\Project 2\\Data'
csv_directory = 'F:\\OneDrive - Texas A&M University\\BMEN 351\\Project 2\\Data\\CSVs'
model_path = r'F:\OneDrive - Texas A&M University\BMEN 351\Project 2\Data\Training\biobert_v1.1_pubmed'
vector_directory = os.path.join(data_directory, 'Training\\Vectors')
updated_csv_path = os.path.join(data_directory, 'Training\\combined_data.csv')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Combine CSV files and vectorize texts
combined_data = combine_and_vectorize(csv_directory, vector_directory)

# Fill NaN values in 'OpenAI_Bool' with 0
combined_data['OpenAI_Bool'] = combined_data['OpenAI_Bool'].fillna(0)

# Save the DataFrame with references to vector files
combined_data.to_csv(updated_csv_path, index=False)

# =============================================================================
# Outputs
# =============================================================================
end = time.time()
print("The total runtime of the above code was", (end - start), "seconds")

