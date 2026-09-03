import fitz


def extract_text(pdf_path, max_chars=1_500_000):
    text_parts = []
    character_count = 0

    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            remaining = max_chars - character_count
            if remaining <= 0:
                break

            page_text = page.get_text()[:remaining]
            text_parts.append(page_text)
            character_count += len(page_text)

    return "".join(text_parts)