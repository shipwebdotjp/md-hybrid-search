"""Vector Search MCP Server.

This MCP (Model Context Protocol) server provides tools for semantic and keyword search
against a vector database of memory entries.

The server exposes one main tool:
1. search: Performs semantic and keyword search with configurable modes (hybrid, similarity-only, fulltext-only).

Both tools return document metadata and content that can be used by LLMs to provide
contextually relevant responses to user queries.
"""

from mcp.server.fastmcp import FastMCP
import main

# Initialize the MCP server with a descriptive name
# This name will appear in Raycast or other tools that connect to this server
mcp = FastMCP(name="Memory Search (Japanese)")

@mcp.tool()
def search(query: str, k: int = 10, mode: str = "hybrid") -> list:
    """Search the User's personal memory database sourced from their vault.

    The corpus contains Japanese-language notes and documents from the user's vault, which may include meeting notes, personal reflections, research materials, and more.

    Args:
        query (str): The search query string. Japanese queries recommended.
        k (int, optional): Maximum number of results to return. Defaults to 10.
        mode (str, optional): Search mode. One of:
            - "hybrid"     : Combines semantic and keyword search with reranking (default)
            - "similarity" : Semantic vector search only
            - "keyword"   : Keyword (BM25) search only

    Returns:
        list: List of document dictionaries containing:
            - page_content (str): The document content/text (Japanese)
            - metadata (dict): Document metadata including source file, title, publication date, etc.
    """
    documents, summary = main.query(query, k=int(k), collection_name="documents", search_mode=mode)
    # metadataからfile_directory,filenameを削除
    cleaned = []
    for doc in documents:
        # dict の場合はそのまま操作
        if isinstance(doc, dict):
            metadata = (doc.get("metadata") or {}).copy()
            metadata.pop("file_directory", None)
            metadata.pop("filename", None)
            doc["metadata"] = metadata
            cleaned.append(doc)
        else:
            # langchain Document などのオブジェクトの場合は属性アクセスで取得して辞書に変換して返す
            metadata = getattr(doc, "metadata", None) or {}
            # コピーして不要なフィールドを削除
            try:
                metadata = dict(metadata)
            except Exception:
                metadata = {}
            metadata.pop("file_directory", None)
            metadata.pop("filename", None)
            page_content = getattr(doc, "page_content", None) or getattr(doc, "pageContent", None)
            cleaned.append({"page_content": page_content, "metadata": metadata})
    return cleaned


@mcp.tool()
def read_file(file_path: str) -> str:
    """Read a file as UTF-8 text and return its full contents.

    Args:
        file_path (str): Path to the file to read.

    Returns:
        str: File contents decoded as UTF-8.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Main execution block
# When run directly, starts the MCP server in stdio mode
# Stdio mode allows the server to communicate with clients through standard input/output
# This is the recommended mode for integrating with Raycast and most AI coding agents
if __name__ == "__main__":
    mcp.run(transport="stdio")
