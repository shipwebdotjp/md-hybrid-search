import sqlite3
import re

def normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

conn = sqlite3.connect(":memory:")
conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, collection_name UNINDEXED, content)")

# Index some content
content = normalize_text("The quick-brown fox jumps over the apple:banana.")
print(f"Indexed content: {content}")
conn.execute("INSERT INTO chunks_fts (chunk_id, collection_name, content) VALUES (?, ?, ?)", ("1", "test", content))

def search(query):
    normalized_query = normalize_text(query)
    print(f"Searching for: {query} (normalized: {normalized_query})")
    try:
        cursor = conn.execute("SELECT chunk_id FROM chunks_fts WHERE content MATCH ?", (normalized_query,))
        results = cursor.fetchall()
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error: {e}")

search("fox")
search("quick-brown")
search("apple:banana")
search("jumps over")
search("-fox")

def tokenize_query(query):
    # Simple tokenization: extract alphanumeric words and quote them
    # This is a common pattern to avoid FTS5 syntax errors
    tokens = re.findall(r'\w+', query)
    return ' '.join(f'"{token}"' for token in tokens)

def search_v2(query):
    tokenized_query = tokenize_query(query)
    print(f"Searching (v2) for: {query} (tokenized: {tokenized_query})")
    try:
        cursor = conn.execute("SELECT chunk_id FROM chunks_fts WHERE content MATCH ?", (tokenized_query,))
        results = cursor.fetchall()
        print(f"Results: {results}")
    except Exception as e:
        print(f"Error: {e}")

search_v2("quick-brown")
search_v2("apple:banana")
search_v2("-fox")
