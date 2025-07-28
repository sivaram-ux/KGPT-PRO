import fitz  # PyMuPDF
import camelot
from variables import file_name
import pdfplumber

# Text extraction
if False: 
    doc = fitz.open(file_name)
    for page in doc:
        text = page.get_text()
        print(len(text))
else:
    # Table extraction
    with pdfplumber.open(file_name) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                print(f"Page {i+1} table:\n", tables[0])
