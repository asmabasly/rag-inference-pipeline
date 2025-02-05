import pdfplumber
import re

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ''
        for page in pdf.pages:
            all_text += page.extract_text()
    return all_text

pdf_path = r'C:\Users\Tifa\Downloads\rag\src\data\circulaire.pdf'
pdf_text = extract_text_from_pdf(pdf_path)
print(pdf_text)


def clean_text(text, is_table=False):
    """
    Clean text by removing unwanted characters and spaces, with different handling based on text context.
    """
    if is_table:
        # Replace multiple spaces or tabs with a single comma (for CSV-like formatting)
        text = re.sub(r'\s{2,}', ',', text)
    else:
        # Normalize spaces and remove special characters in regular text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9é\s]', ' ', text)
        text = re.sub(r'[^\w.,\':;\s]', ' ', text)

    # Replace 'ROWEND' with actual newlines if needed or handle as you see fit
        text = re.sub(r' ROWEND ', '\n', text)
        text = re.sub(r' +', ' ', text)
    text = text.strip()
    return text

def process_pdf(file_path):
    all_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract text from tables
            for table in page.extract_tables():
                for row in table:
                    cleaned_row = [clean_text(cell, is_table=True) for cell in row]
                    all_text.append(','.join(cleaned_row))
            
            # Extract regular text
            text = page.extract_text()
            if text:
                cleaned_text = clean_text(text)
                all_text.append(cleaned_text)
    
    return '\n'.join(all_text)

# Example usage
processed_text = process_pdf(pdf_path)
print(processed_text)