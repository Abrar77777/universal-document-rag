from utils.pdf_loader import load_pdf
from utils.text_cleaner import clean_text
from utils.text_splitter import split_text

raw_text = load_pdf("data/raw/company_policy.pdf")
cleaned_text = clean_text(raw_text)

chunks = split_text(cleaned_text)

print(f"Total chunks: {len(chunks)}")
print("\n--- SAMPLE CHUNK ---\n")
print(chunks[0][:500])
print("\n--- END SAMPLE CHUNK ---\n")

