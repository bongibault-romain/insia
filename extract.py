import nltk
from PyPDF2 import PdfReader
#nltk.download('punkt_tab')
# Télécharger le tokenizer de phrases si nécessaire
#nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def pdf_to_sentences(pdf_path, txt_output_path):
    # Lire le PDF
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    
    
    paragraphs = full_text.split("\n")

    paragraphs=[i for i in paragraphs if len(i)>15]
    paragraphs = [
        "".join(paragraphs[i:i+3])
        for i in range(0, len(paragraphs), 3)
    ]

    # Écrire chaque phrase dans un fichier texte
    with open(txt_output_path, "w", encoding="utf-8") as f:
        for i in range(len(paragraphs)):
            f.write(paragraphs[i].strip()+"\n")



# Exemple d'utilisation
pdf_to_sentences("Reglement_des_Etudes_2023-2024.pdf", "reglement.txt")
