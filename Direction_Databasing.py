#### Direction_Databasing.py

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


#### Modules
# General
import time
import os
import sys

# CSV Parsing
import pandas as pd
import logging
import multiprocessing
from multiprocessing import Pool



# Inputs

# =============================================================================
# Direction_Databasing
# =============================================================================
def direction_collector(args):
    """
    Process a single CSV and classify its text content, then save to a new CSV file.
    
    :param args: Tuple containing input CSV file path and output CSV file path.
    """
    try:
        csv_file_path, directions_csv_path = args
        data = pd.read_csv(csv_file_path)
        # Check if 'OpenAI_Output' column exists
        if 'OpenAI_Output' in data.columns:
            # Create a new DataFrame for appending
            append_data = pd.DataFrame()

            for index, row in data.iterrows():
                if pd.notna(row['OpenAI_Output']) and row['OpenAI_Output']:  # Check for non-empty cells
                    # Ensure the data in 'OpenAI_Output' is treated as a string
                    openai_output_str = str(row['OpenAI_Output'])

                    # Splitting the OpenAI_Output into two columns: 'OpenAI_Bool' and 'OpenAI_Subjects'
                    openai_bool, openai_subjects = openai_output_str.split(',', 1)

                    # Append the relevant data to the new DataFrame
                    # append_data = append_data.append({
                    #     'CSV_Title': os.path.basename(csv_file_path),
                    #     'Text': row['Text'],
                    #     'Research_Advice': row['Research_Advice'],
                    #     'OpenAI_Bool': openai_bool.strip(),
                    #     'OpenAI_Subjects': openai_subjects.strip()
                    # }, ignore_index=True)
                    pd.concat([append_data,pd.DataFrame({
                        'CSV_Title': os.path.basename(csv_file_path),
                        'Text': row['Text'],
                        'Research_Advice': row['Research_Advice'],
                        'OpenAI_Bool': openai_bool.strip(),
                        'OpenAI_Subjects': openai_subjects.strip()
                    })])

            # Append to the master CSV file, if there's data to append
            if not append_data.empty:
                append_data.to_csv(directions_csv_path, mode='a', index=False, header=not os.path.exists(directions_csv_path))

    except Exception as e:
        logging.error(f"Error occurred while processing CSV: {csv_file_path}. Error: {e}")

def database_directions(CSV_input_directory, directions_csv_directory, user_topic):
    """
    Process all CSVs in the input directory and classify text content, 
    then save to corresponding new CSV files in the output directory.
    
    :param CSV_input_directory: Path to the directory containing input CSVs.
    :param directions_output_directory: Path to the directory where output CSVs will be saved.
    """
    try:
        tasks = []
        for csv_file in os.listdir(CSV_input_directory):
            if csv_file.lower().endswith('.csv'):
                csv_file_path = os.path.join(CSV_input_directory, csv_file)
                directions_csv_path = os.path.join(directions_csv_directory, 
                                               '{}.csv'.format(user_topic))
                tasks.append((csv_file_path, directions_csv_path))

        with Pool() as pool:
            pool.map(direction_collector, tasks)

    except Exception as e:
        logging.error(f"Error occurred while processing CSVs: {e}")

# =============================================================================
# Outputs
# =============================================================================
if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) != 2:
        print("Usage: python Direction_Databasing.py input_directory")
        sys.exit(1)

    data_directory = sys.argv[1] + r"\Data"
    
    
    #### Paper_Databasing

    # Define the output directory where CSVs will be saved
    CSV_input_directory = os.path.join(data_directory, "CSVs")
    directions_csv_directory = os.path.join(data_directory, "Directions")
    
    # User topic
    user_topic = "Tetrahedral_DNA_Nanocages"
    
    # Call Function
    database_directions(CSV_input_directory, directions_csv_directory, user_topic)
    
    # Time
    end = time.time()
    print("The total runtime of the above code was",(end-start), "seconds")
