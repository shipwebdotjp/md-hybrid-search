import glob
import re
import os
import datetime
from typing import List
from langchain_core.documents import Document
from FileIndex import FileIndex
from loader.custom_file_loader import CustomFileLoader

class CustomDirectoryLoader:
    def __init__(self, directory_path: str, glob_pattern: str = "*.*", mode: str = "single", db = None, collection_name = None):
        """
        Initialize the loader with a directory path and a glob pattern.
        :param directory_path: Path to the directory containing files to load.
        :param glob_pattern: Glob pattern to match files within the directory.
        :param mode: Mode to use with UnstructuredFileLoader ('single', 'elements', or 'paged').
        """
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.mode = mode
        self.db = db
        self.collection_name = collection_name

    def load(self) -> dict:
        """
        Load all files matching the glob pattern in the directory using UnstructuredFileLoader.
        :return: List of Document objects loaded from the files.
        """
        total_tokens = 0
        total_docs = 0

        fileIndex = FileIndex(db=self.db, directory_path=self.directory_path, glob_pattern=self.glob_pattern, collection_name=self.collection_name)
        newFiles = fileIndex.getNewFiles()
        # print(f'newFiles: {newFiles}')

        # Construct the full glob pattern
        # full_glob_pattern = f"{self.directory_path}/**/*.*"
        # print(f"Using glob pattern: {full_glob_pattern}")
        # Iterate over all files matched by the glob pattern
        for file_path in newFiles:
            loader = CustomFileLoader(file_path, db=self.db, fileIndex=fileIndex, directory_path=self.directory_path, mode=self.mode)
            result = loader.load()
            total_docs += result['count']
            total_tokens += result['total_tokens']
            # documents.extend(docs)
        # self.deletedFiles(self.db, allFiles)
        fileIndex.checkDeletedFiles()
        return {'count': total_docs, 'total_tokens': total_tokens}
    
    # def checkDBduplicate(self, db, file_path):
    #     collection = db.get(where={"source": file_path})
    #     if len(collection['documents']) > 0:
    #         last_modified = collection['metadatas'][0]['last_modified']
    #         file_modified = os.path.getmtime(file_path)
    #         datetime_file = datetime.datetime.fromtimestamp(file_modified)
    #         # format to 2023-06-07T20:52:39
    #         formatted_date = datetime_file.strftime('%Y-%m-%dT%H:%M:%S')

    #         # print(f'last_modified: {last_modified}')
    #         # print(f'formatted_date: {formatted_date}')
    #         if last_modified != formatted_date:
    #             self.delete_document(db, file_path)
    #             return False
    #         else:
    #             return True
    #     else:
    #         return False
    
    # def delete_document(self, db, file_path):
    #     collection = db.get(where={"source": file_path})
    #     if len(collection['documents']) > 0:
    #         ids = collection['ids']
    #         db.delete(ids)

    # def deletedFiles(self, db, allFiles):
    #     # If file were deleted, then remove from collections
    #     collection = db.get(where={"file_directory": self.directory_path})
    #     allFilesInCollection = []
    #     if len(collection['documents']) > 0:
    #         for metadata in collection['metadatas']:
    #             allFilesInCollection.append(metadata['source'])
    #         # remove duplicates
    #         allFilesInCollection = list(set(allFilesInCollection))
    #         for file in allFilesInCollection:
    #             if file not in allFiles:
    #                 print(f'Deleting file: {file}')
    #                 self.delete_document(db, file)
