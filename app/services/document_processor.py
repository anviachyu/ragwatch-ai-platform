from io import BytesIO

from pypdf import PdfReader


def extract_text_from_pdf(file_content: bytes) -> tuple[str, int]:
    pdf_reader = PdfReader(BytesIO(file_content))
    extracted_pages = []

    for page in pdf_reader.pages:
        page_text = page.extract_text() or ""
        extracted_pages.append(page_text)

    complete_text = "\n".join(extracted_pages).strip()

    return complete_text, len(pdf_reader.pages)