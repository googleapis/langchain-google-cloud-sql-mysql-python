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
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Sequence, cast

import sqlalchemy
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from langchain_google_cloud_sql_mysql.mysql_engine import MySQLEngine


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
        doc = Document(page_content=page_content, metadata=metadata)
        docs.append(doc)
    return docs


class MySQLLoader(BaseLoader):
    """A class for loading langchain documents from a Cloud SQL MySQL database."""

    def __init__(
        self,
        engine: MySQLEngine,
        query: str,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
    ):
        """
        Args:
          engine (MySQLEngine): MySQLEngine object to connect to the MySQL database.
          query (str): The query to execute in MySQL format.
          content_columns (List[str]): The columns to write into the `page_content`
             of the document. Optional.
          metadata_columns (List[str]): The columns to write into the `metadata` of the document.
             Optional.
        """
        self.engine = engine
        self.query = query
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns

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
        with self.engine.connect() as connection:
            result_proxy = connection.execute(sqlalchemy.text(self.query))
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
