from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from unstructured.cleaners.core import group_broken_paragraphs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
import re
import os
import datetime
import glob
from bs4 import BeautifulSoup
from src.loader.CustomHTMLLoader import CustomHTMLLoader
import tiktoken
import sqlite3
import json
from src.search.fts_sync import get_fts_sync_manager


class FileIndex:
    def __init__(
        self,
        directory_path: str,
        glob_pattern: str = "*.*",
        db=None,
        collection_name=None,
    ):
        self.db = db
        self.directory_path = directory_path
        self.glob_pattern = glob_pattern
        self.collection_name = collection_name

    def getAllFilesInCollection(self):
        con = sqlite3.connect("db/settings.sqlite3")
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            """
            SELECT file_path FROM file
            WHERE collection_name = ? AND file_directory = ?
        """,
            (self.collection_name, self.directory_path),
        )
        allFilesInCollection = [row[0] for row in cur.fetchall()]
        con.close()

        # collection_name = self.collection_name
        # print(f'collection name: {collection_name}')
        # collection = self.db.get(where={"file_directory": self.directory_path})
        # print(f'directory: {self.directory_path}')
        # print("Number of files in collection: ", len(collection['documents']))
        # allFilesInCollection = []
        # if len(collection['documents']) > 0:
        #     for metadata in collection['metadatas']:
        #         allFilesInCollection.append(metadata['source'])
        # allFilesInCollection = list(set(allFilesInCollection))

        # difference between allFilesInCollection and allFilesInDirectory
        # files in collection that are not in directory
        # deletedFiles = list(set(allFilesInCollection) - set(allFilesInDirectory))
        # print("Number of files deleted: ", len(deletedFiles))
        # print("Files deleted: ", deletedFiles)
        # files in directory that are not in collection
        # newFiles = list(set(allFilesInDirectory) - set(allFilesInCollection))
        # print("Number of new files: ", len(newFiles))
        # print("New files: ", newFiles)
        return allFilesInCollection

    def getAllFilesInDirectory(self):
        allFilesInDirectory = []
        # glob pattern for all files in directory
        full_glob_pattern = f"{self.directory_path}/**/*.*"
        for file_path in glob.glob(full_glob_pattern, recursive=True):
            if not re.search(self.glob_pattern, file_path):
                continue
            allFilesInDirectory.append(file_path)
        return allFilesInDirectory

    def getNewFiles(self):
        allFilesInDirectory = self.getAllFilesInDirectory()
        allFilesInCollection = self.getAllFilesInCollection()
        newFiles = list(set(allFilesInDirectory) - set(allFilesInCollection))
        con = sqlite3.connect("db/settings.sqlite3")
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        for file_in_directory in allFilesInDirectory:
            if file_in_directory not in newFiles:
                file_last_modified = os.path.getmtime(file_in_directory)
                datetime_file = datetime.datetime.fromtimestamp(file_last_modified)
                formatted_date = datetime_file.strftime("%Y-%m-%dT%H:%M:%S")
                cur.execute(
                    """
                    SELECT file_path, document_ids FROM file
                    WHERE collection_name = ? AND file_directory = ? AND file_path = ? AND last_modified < ?
                """,
                    (
                        self.collection_name,
                        self.directory_path,
                        file_in_directory,
                        formatted_date,
                    ),
                )
                row = cur.fetchone()
                if row is not None:
                    self.deleteFile(file_in_directory, json.loads(row["document_ids"]))
                    newFiles.append(file_in_directory)
        return newFiles

    def checkDeletedFiles(self):
        allFilesInDirectory = self.getAllFilesInDirectory()
        allFilesInCollection = self.getAllFilesInCollection()
        deletedFiles = list(set(allFilesInCollection) - set(allFilesInDirectory))
        if len(deletedFiles) > 0:
            print("Files deleted: ", deletedFiles)
            con = sqlite3.connect("db/settings.sqlite3")
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            for file_to_delete in deletedFiles:
                cur.execute(
                    """
                    SELECT file_path, document_ids FROM file
                    WHERE collection_name = ? AND file_directory = ? AND file_path = ?
                """,
                    (self.collection_name, self.directory_path, file_to_delete),
                )
                row = cur.fetchone()
                if row is not None:
                    self.deleteFile(file_to_delete, json.loads(row["document_ids"]))
            con.close()

    def deleteFile(self, file_path, document_ids):
        con = sqlite3.connect("db/settings.sqlite3")
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            """
            DELETE FROM file
            WHERE collection_name = ? AND file_directory = ? AND file_path = ?
        """,
            (self.collection_name, self.directory_path, file_path),
        )
        con.commit()
        con.close()
        if len(document_ids) > 0 and self.db is not None:
            # ChromaDBから削除
            self.db.delete(document_ids)
            # FTS5からも削除
            fts_sync = get_fts_sync_manager()
            for doc_id in document_ids:
                fts_sync.sync_delete(doc_id)

    def addFile(self, file_path, document_ids):
        file_name = os.path.basename(file_path)
        last_modified = datetime.datetime.fromtimestamp(
            os.path.getmtime(file_path)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        added_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        con = sqlite3.connect("db/settings.sqlite3")
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO file
            (collection_name, file_directory, file_path, file_name, document_ids, last_modified, added_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                self.collection_name,
                self.directory_path,
                file_path,
                file_name,
                json.dumps(document_ids),
                last_modified,
                added_date,
            ),
        )
        con.commit()
        con.close()
        print(f"Added file: {file_path}")

    def addOrMergeFile(self, file_path, document_ids):
        """
        Insert a new file record or merge document_ids into an existing record.

        Behavior:
        - If no record exists for (collection_name, file_directory, file_path):
            Insert a new row with document_ids, last_modified (from filesystem) and added_date (now).
        - If a record exists:
            Merge the provided document_ids into the existing document_ids (preserve existing order,
            append new ids that are not already present). Do NOT update last_modified or added_date.
        """
        file_name = os.path.basename(file_path)
        # last_modified is set on insert only (do not update on merge)
        last_modified = datetime.datetime.fromtimestamp(
            os.path.getmtime(file_path)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        added_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        con = sqlite3.connect("db/settings.sqlite3")
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # Check for existing record
        cur.execute(
            """
            SELECT document_ids FROM file
            WHERE collection_name = ? AND file_directory = ? AND file_path = ?
        """,
            (self.collection_name, self.directory_path, file_path),
        )
        row = cur.fetchone()

        if row is None:
            # No existing record: insert new
            cur.execute(
                """
                INSERT INTO file (collection_name, file_directory, file_path, file_name, document_ids, last_modified, added_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    self.collection_name,
                    self.directory_path,
                    file_path,
                    file_name,
                    json.dumps(document_ids),
                    last_modified,
                    added_date,
                ),
            )
            con.commit()
            con.close()
            # print(f'Added file: {file_path}')
            return

        # Existing record: merge document_ids (document_ids are 1D string ID arrays)
        try:
            existing_raw = row["document_ids"]
            existing_ids = json.loads(existing_raw) if existing_raw else []
        except Exception:
            existing_ids = []

        # Preserve existing order, append only new IDs
        merged = existing_ids.copy()
        added_count = 0
        for nid in document_ids:
            if nid not in merged:
                merged.append(nid)
                added_count += 1

        if added_count > 0:
            cur.execute(
                """
                UPDATE file
                SET document_ids = ?
                WHERE collection_name = ? AND file_directory = ? AND file_path = ?
            """,
                (
                    json.dumps(merged),
                    self.collection_name,
                    self.directory_path,
                    file_path,
                ),
            )
            con.commit()
            print(
                f"Merged file: {file_path} (existing: {len(existing_ids)}, added: {added_count}, merged: {len(merged)})"
            )

        con.close()
