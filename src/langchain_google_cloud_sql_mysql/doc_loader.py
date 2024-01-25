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
from typing import Dict, List, Optional, cast

import sqlalchemy
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from . import CloudSQLMySQLEngine


def _parse_doc_from_table(page_content_columns, metadata_columns, column_names, rows):
    docs = []
    for row in rows:
        page_content = "\n".join(
            f"{column}: {getattr(row, column)}"
            for column in page_content_columns
            if column in column_names
        )
        metadata = {
            column: getattr(row, column)
            for column in metadata_columns
            if column in column_names
        }
        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)
    return docs


class CloudSQLMySQLDBLoader(BaseLoader):
    """A class for loading langchain documents from a Cloud SQL MySQL database."""

    def __init__(
        self,
        engine: CloudSQLMySQLEngine,  # CloudSQLMySQLEngine for the connection
        query: str,  # Query in MySQL format
        page_content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        """
        Args:
          engine: CloudSQLMySQLEngine object to connect to the MySQL database.
          query: The query to execute in MySQL format.
          page_content_columns: The columns to write into the `page_content`
             of the document. Optional.
          metadata_columns: The columns to write into the `metadata` of the document.
             Optional.
        """
        self.engine = engine
        self.query = query
        self.page_content_columns = page_content_columns
        self.metadata_columns = metadata_columns

    def load(self) -> List[Document]:
        """
        Load langchain documents from a Cloud SQL MySQL database.

        Returns:
            a list of Documents with metadata from specific columns.
        """
        with self.engine.connect() as connection:
            result_proxy = connection.execute(sqlalchemy.text(self.query))
            column_names = result_proxy.keys()
            results = result_proxy.fetchall()
            return _parse_doc_from_table(
                self.page_content_columns or column_names,
                self.metadata_columns or [],
                column_names,
                results,
            )
