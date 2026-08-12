from pathlib import Path
import fitz

p = Path(r"C:\Users\Sai Krishna\Downloads\front-end-developer_resume_sample.pdf")
data = p.read_bytes()
print("bytes", len(data))
doc = fitz.open(stream=data, filetype="pdf")
print("pages", doc.page_count)
for i, page in enumerate(doc):
    text = page.get_text("text") or ""
    imgs = page.get_images(full=True)
    print(f"page {i}: text_len={len(text.strip())} images={len(imgs)}")
    print("preview:", repr(text[:500]))
doc.close()
