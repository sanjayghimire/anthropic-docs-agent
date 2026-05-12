from ingest import download_llms_txt, parse_pages, chunk_text, embed_texts, upsert_chunks

content    = download_llms_txt()
pages      = parse_pages(content)[:3]

for page in pages:
    chunks = chunk_text(page["content"])
    
    if not chunks:
        print(f"Skipped: {page['title']}")
        continue
    
    embeddings = embed_texts(chunks)
    upsert_chunks(chunks, embeddings, page["url"], page["title"])
    print(f"Stored: {page['title']} — {len(chunks)} chunks")