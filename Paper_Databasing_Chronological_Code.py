# Imports
import os
import sys
import csv
import re
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import logging

# Inputs
input_pdf_path = "C:/Users/IOX20/OneDrive - Texas A&M University/BMEN 351/Project 2/Data/PDFs/Self-Assembled Multivalent DNA Nanostructures for Noninvasive Intracellular Delivery of Immunostimulatory CpG Oligonucleotides.pdf"  # Replace with the path to your PDF file
output_csv_path = "C:/Users/IOX20/OneDrive - Texas A&M University/BMEN 351/Project 2/Data/CSVs/Self-Assembled Multivalent DNA Nanostructures for Noninvasive Intracellular Delivery of Immunostimulatory CpG Oligonucleotides.csv"  # Replace with the desired output CSV path


# Extract Text
try:
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)  # Remove 'codec' argument
    
    with open(input_pdf_path, 'rb') as fp:
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()
        
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, 
                                      password=password, caching=caching, 
                                      check_extractable=True):
            interpreter.process_page(page)
        
        text = retstr.getvalue()
    
    device.close()
    retstr.close()
    
    # Clean the extracted text
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Remove new lines
    cleaned_text = re.sub(r'[\r\n]+', ' ', cleaned_text)
    
    # Split into sentences
    sentences = re.split(r'(?<=\.|!|\?)(?<!\d\.\d)(?!\s*[a-z]) *', cleaned_text)
    
    # Merge small sentences
    min_char_count = 40  # Minimum character count for a sentence to be considered complete
    min_space_count = 5  # Minimum space count for a sentence to be considered complete
    
    # Function to check if a sentence is below the thresholds
    def is_sentence_short(sentence):
        return len(sentence) < min_char_count or sentence.count(' ') < min_space_count
    
    # Iterate through sentences and append short sentences to the previous one
    i = 0
    while i < len(sentences):
        if i > 0 and is_sentence_short(sentences[i]):
            sentences[i-1] += ' ' + sentences.pop(i)
        else:
            i += 1

    # Function to check if a sentence has more than one semicolon
    def has_multiple_semicolons(sentence):
        return sentence.count(';') > 1
    
    # Iterate through sentences and remove those with multiple semicolons
    sentences = [sentence for sentence in sentences if not has_multiple_semicolons(sentence)]

    # Create rolling window text chunks
    rolling_window_size = 3
    text_chunks = []
    for i in range(len(sentences) - rolling_window_size + 1):
        chunk = ' '.join(sentences[i:i + rolling_window_size])
        text_chunks.append(chunk)

    # Write the text chunks to the CSV file
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Text"])  # Header row
        csv_writer.writerows([[chunk] for chunk in text_chunks])
except Exception as e:
    logging.error(f"Error occurred while processing PDF to CSV with rolling window: {e}")