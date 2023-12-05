#### Stats.py

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

# Modules
import time
import pandas as pd
import time

# =============================================================================
# Code
# =============================================================================
start = time.time()

# Input csv with pd dataframe
# r"C:/Users/IOX20/OneDrive - Texas A&M University/BMEN 351/Project 2/combined_data.csv"
# Header as follows: "Text	Research_Advice	Temporal_Position	OpenAI_Bool	OpenAI_Subjects	vector_file"
# Drop Text, Temporal_Position, OpenAI_Subjects, and vector_file to save processing


# Model statistics:
# Take Length of dataframe
# Count number of 1s in research_advice
# Count number of 0s in OpenAI_Bool
# Count number of 1s in OpenAI_Bool
# if value in OpenAI_bool != bool (if it is text most likely) add to new counter for openAI errors

# Load the CSV file into a DataFrame
df = pd.read_csv(r"C:/Users/IOX20/OneDrive - Texas A&M University/BMEN 351/Project 2/combined_data.csv")

# Drop unnecessary columns
df = df.drop(['Text', 'Temporal_Position', 'OpenAI_Subjects', 'vector_file'], axis=1)

def convert_to_bool_or_int(value):
    try:
        # Try to convert the value to an integer
        return int(float(value))
    except (ValueError, TypeError):
        # If conversion fails, return the value as it is (string)
        return value
    
def convert_to_int_or_drop(value):
    try:
        # Try to convert the value to an integer
        return int(float(value))
    except (ValueError, TypeError):
        # If conversion fails, signal to drop this row
        return 'drop'

# Attempt to convert 'Research_Advice' values to int, flag rows to drop if not convertible
df['Research_Advice'] = df['Research_Advice'].apply(convert_to_int_or_drop)

# Drop rows where 'Research_Advice' could not be converted (flagged as 'drop')
df = df[df['Research_Advice'] != 'drop']

df['OpenAI_Bool'] = df['OpenAI_Bool'].apply(convert_to_bool_or_int)

df_openai_bool_1 = df[df['OpenAI_Bool'] == 1]
df_openai_bool_0 = df[df['OpenAI_Bool'] == 0]
df_openai_bool_errors = df[df['OpenAI_Bool'].apply(lambda x: isinstance(x, str))]




# Model statistics
total_entries = len(df)
num_research_advice_1s = df['Research_Advice'].sum()  # Assuming Research_Advice column is binary (0/1)
# num_openai_bool_0s = (df['OpenAI_Bool'] == 0).sum()
# num_openai_bool_1s = (df['OpenAI_Bool'] == 1).sum()
# num_openai_errors = df['OpenAI_Bool'].apply(lambda x: isinstance(x, str)).sum()
num_openai_bool_1s = len(df_openai_bool_1)
num_openai_errors = len(df_openai_bool_errors)
num_openai_bool_0s = num_research_advice_1s-(num_openai_bool_1s+num_openai_errors)

# =============================================================================
# Outputs
# =============================================================================
# Print statistics
print(f"Total number of entries: {total_entries}")
print(f"Number of positive research advice entries: {num_research_advice_1s}")
print(f"Number of OpenAI_Bool entries with value 0: {num_openai_bool_0s}")
print(f"Number of OpenAI_Bool entries with value 1: {num_openai_bool_1s}")
print(f"Number of OpenAI_Bool entries with errors (non-boolean values): {num_openai_errors}")


end = time.time()
print("The total runtime of the above code was",(end-start), "seconds")
