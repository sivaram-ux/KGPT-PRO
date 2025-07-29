import markdown
from bs4 import BeautifulSoup

def parse_markdown_chunks(markdown_path):
    with open(markdown_path, "r", encoding="utf-8") as f:
        raw_md = f.read()
    html = markdown.markdown(raw_md, extensions=["tables"])
    soup = BeautifulSoup(html, "html.parser")

    chunks = []
    current_title, current_content, chunk_id = "", [], 0

    def flush_chunk():
        nonlocal chunk_id, current_content
        if current_content:
            full_content = f"{current_title}:\n" + "\n".join(current_content)
            chunks.append({"id": f"chunk_{chunk_id}", "title": current_title, "content": full_content})
            chunk_id += 1
            current_content = []

    for elem in soup.find_all(["h1", "h2", "h3", "p", "ul", "table"]):
        if elem.name in ["h1", "h2", "h3"]:
            flush_chunk()
            current_title = elem.text.strip()
        elif elem.name == "p":
            text = elem.get_text(strip=True)
            if text:
                current_content.append(text)
        elif elem.name == "ul":
            items = [li.get_text(strip=True) for li in elem.find_all("li")]
            if items:
                current_content.append("• " + "\n• ".join(items))
        elif elem.name == "table":
            print(f"[Table] Found at chunk: {chunk_id}")
            rows = elem.find_all("tr")
            if rows:
                headers = [td.get_text(strip=True) for td in rows[0].find_all(["th", "td"])]
                sentences = []
                for row in rows[1:]:
                    values = [td.get_text(strip=True) for td in row.find_all("td")]
                    flat = ", ".join([f"{k} is {v}" for k, v in zip(headers, values)])
                    sentences.append(flat)
                current_content.append(" ".join(sentences))
    flush_chunk()
    return chunks
