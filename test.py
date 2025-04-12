
from PyPDF2 import PdfReader
pdf_path="Reglement_des_Etudes_2023-2024.pdf"
reader = PdfReader(pdf_path)
full_text = ""
i=0
for page in reader.pages:
    i+=1
print(i)