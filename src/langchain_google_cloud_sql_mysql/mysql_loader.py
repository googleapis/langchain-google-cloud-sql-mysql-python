# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Sequence, cast

import sqlalchemy
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from langchain_google_cloud_sql_mysql.mysql_engine import MySQLEngine

DEFAULT_METADATA_COL = "langchain_metadata"


def _parse_doc_from_table(
    content_columns: Iterable[str],
    metadata_columns: Iterable[str],
    column_names: Iterable[str],
    rows: Sequence[Any],
) -> List[Document]:
    docs = []
    for row in rows:
        page_content = " ".join(
            str(getattr(row, column))
            for column in content_columns
            if column in column_names
        )
        metadata = {
            column: getattr(row, column)
            for column in metadata_columns
            if column in column_names
        }
        # load extra metadata from langchain_metadata column
        if DEFAULT_METADATA_COL in metadata:
            extra_metadata = json.loads(metadata[DEFAULT_METADATA_COL])
            del metadata[DEFAULT_METADATA_COL]
            metadata |= extra_metadata
        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)
    return docs


def _parse_row_from_doc(metadata_columns: Iterable[str], doc: Document) -> Dict:
    doc_metadata = doc.metadata.copy()
    row = {"page_content": doc.page_content}
    for entry in doc.metadata:
        if entry in metadata_columns:
            row[entry] = doc_metadata[entry]
            del doc_metadata[entry]
    # store extra metadata in langchain_metadata column in json format
    if LANGCHAIN_METADATA in metadata_columns and len(doc_metadata) > 0:
        row[LANGCHAIN_METADATA] = json.dumps(doc_metadata)
    return row


class MySQLLoader(BaseLoader):
    """A class for loading langchain documents from a Cloud SQL MySQL database."""

    def __init__(
        self,
        engine: MySQLEngine,
        table_name: str = "",
        query: str = "",
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        """
        Args:
          engine (MySQLEngine): MySQLEngine object to connect to the MySQL database.
          table_name (str): The MySQL database table name. (OneOf: table_name, query).
          query (str): The query to execute in MySQL format.  (OneOf: table_name, query).
          content_columns (List[str]): The columns to write into the `page_content`
             of the document. Optional.
          metadata_columns (List[str]): The columns to write into the `metadata` of the document.
             Optional.
        """
        self.engine = engine
        self.table_name = table_name
        self.query = query
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns
        if self.table_name == "" and self.query == "":
            raise ValueError("One of table_name or query needs be specified.")
        if self.table_name and self.query:
            raise ValueError("Cannot specify both table_name and query.")

    def load(self) -> List[Document]:
        """
        Load langchain documents from a Cloud SQL MySQL database.

        Document page content defaults to the first columns present in the query or table and
        metadata defaults to all other columns. Use with content_columns to overwrite the column
        used for page content. Use metadata_columns to select specific metadata columns rather
        than using all remaining columns.

        If multiple content columns are specified, page_content’s string format will default to
        space-separated string concatenation.

        Returns:
            (List[langchain_core.documents.Document]): a list of Documents with metadata from
                specific columns.
        """
        stmt = None
        if self.query:
            stmt = sqlalchemy.text(self.query)
        if self.table_name:
            stmt = sqlalchemy.text(f"select * from `{self.table_name}`;")
        with self.engine.connect() as connection:
            result_proxy = connection.execute(stmt)
            column_names = list(result_proxy.keys())
            results = result_proxy.fetchall()
            content_columns = self.content_columns or [column_names[0]]
            metadata_columns = self.metadata_columns or [
                col for col in column_names if col not in content_columns
            ]
            return _parse_doc_from_table(
                content_columns,
                metadata_columns,
                column_names,
                results,
            )


class MySQLDocumentSaver:
    """A class for saving langchain documents into a Cloud SQL MySQL database table."""

    def __init__(
        self,
        engine: MySQLEngine,
        table_name: str,
    ):
        """
        Args:
          engine: MySQLEngine object to connect to the MySQL database.
          table_name: The name of table for saving documents.
        """
        self.engine = engine
        self.table_name = table_name
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS `{self.table_name}` (
                page_content LONGTEXT NOT NULL,
                langchain_metadata LONGTEXT NOT NULL
            );
        """

        with self.engine.connect() as conn:
            conn.execute(sqlalchemy.text(create_table_query))
            conn.commit()

        self._table = self.engine.load_document_table(self.table_name)
        self._columns = self._table.columns.keys()

    def add_documents(self, docs: List[Document]):
        with self.engine.connect() as conn:
            for doc in docs:
                row = _parse_row_from_doc(self._columns, doc)
                conn.execute(sqlalchemy.insert(self._table).values(row))
            conn.commit()

    def delete(self, docs: List[Document]):
        with self.engine.connect() as conn:
            for doc in docs:
                row = _parse_row_from_doc(self._columns, doc)
                conn.execute(sqlalchemy.delete(self._table).values(row))
            conn.commit()
