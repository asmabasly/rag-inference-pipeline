import PyPDF2

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

#pdf_text = extract_text_from_pdf("path/to/your/document.pdf")
#text_chunks = pdf_text.split("\n\n")