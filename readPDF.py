#from pdfminer.high_level import extract_text as extract_text_from_pdf
from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_docx(file_obj):
    doc = Document(file_obj)  # file_obj can be a path or a file-like object
    text = " ".join([para.text for para in doc.paragraphs])
    return text


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text= " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text


t=extract_text_from_pdf("AI.pdf")
print(t)
doc_text = extract_text_from_docx("AI for All.docx")
#print(doc_text)
#print(clean_text)

