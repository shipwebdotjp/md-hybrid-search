import os
import datetime
from typing import List
from langchain_core.documents import Document


class CustomTextLoader:
    def __init__(self, file_path: str):
        """
        Initialize the loader with a file path to a text file.
        :param file_path: Path to the text file to load.
        """
        self.file_path = file_path

    def load(self) -> List[Document]:
        """
        Load the text file and return a list of Document objects.
        If a read error occurs, fall back to reading with errors='replace'.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            # Catch any loader errors and fall back to reading with errors='replace'
            msg = str(e)
            print(
                f"Warning: loader failed for {self.file_path}: {msg}\nFalling back to plain-text read."
            )
            try:
                with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                content = ""
        doc = Document(page_content=content, metadata={})
        doc.metadata["source"] = self.file_path
        doc.metadata["filename"] = os.path.basename(self.file_path)
        doc.metadata["last_modified"] = datetime.datetime.fromtimestamp(
            os.path.getmtime(self.file_path)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        return [doc]
