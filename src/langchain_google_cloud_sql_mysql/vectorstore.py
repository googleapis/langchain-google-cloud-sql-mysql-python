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

# TODO: Remove below import when minimum supported Python version is 3.10
from __future__ import annotations

import json
from typing import Any, Iterable, List, Optional, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .engine import MySQLEngine
from .indexes import DEFAULT_QUERY_OPTIONS, QueryOptions, SearchType, VectorIndex

DEFAULT_INDEX_NAME_SUFFIX = "langchainvectorindex"


class MySQLVectorStore(VectorStore):
    def __init__(
        self,
        engine: MySQLEngine,
        embedding_service: Embeddings,
        table_name: str,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: Optional[str] = "langchain_metadata",
        query_options: QueryOptions = DEFAULT_QUERY_OPTIONS,
    ):
        """Constructor for MySQLVectorStore.
        Args:
            engine (MySQLEngine): Connection pool engine for managing
                connections to Cloud SQL for MySQL database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of an existing table or table to be created.
            content_column (str): Column that represent a Document's
                page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding
                is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (List[str]): Column(s) to ignore in
                pre-existing tables for a document's metadata. Can not be used
                with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id.
                Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON.
                Defaults to "langchain_metadata".
        """
        if metadata_columns and ignore_metadata_columns:
            raise ValueError(
                "Can not use both metadata_columns and ignore_metadata_columns."
            )
        # Get field type information
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"

        results = engine._fetch(stmt)
        columns = {}
        for field in results:
            columns[field["COLUMN_NAME"]] = field["DATA_TYPE"]

        # Check columns
        if id_column not in columns:
            raise ValueError(f"Id column, {id_column}, does not exist.")
        if content_column not in columns:
            raise ValueError(f"Content column, {content_column}, does not exist.")
        content_type = columns[content_column]
        if content_type != "text" and "char" not in content_type:
            raise ValueError(
                f"Content column, {content_column}, is type, {content_type}. It must be a type of character string."
            )
        if embedding_column not in columns:
            raise ValueError(f"Embedding column, {embedding_column}, does not exist.")
        if columns[embedding_column] != "varbinary":
            raise ValueError(
                f"Embedding column, {embedding_column}, is not type Vector (varbinary)."
            )

        metadata_json_column = (
            None if metadata_json_column not in columns else metadata_json_column
        )

        # If using metadata_columns check to make sure column exists
        for column in metadata_columns:
            if column not in columns:
                raise ValueError(f"Metadata column, {column}, does not exist.")

        # If using ignore_metadata_columns, filter out known columns and set known metadata columns
        all_columns = columns
        if ignore_metadata_columns:
            for column in ignore_metadata_columns:
                del all_columns[column]

            del all_columns[id_column]
            del all_columns[content_column]
            del all_columns[embedding_column]
            metadata_columns = [key for key, _ in all_columns.keys()]

        # set all class attributes
        self.engine = engine
        self.embedding_service = embedding_service
        self.table_name = table_name
        self.content_column = content_column
        self.embedding_column = embedding_column
        self.metadata_columns = metadata_columns
        self.id_column = id_column
        self.metadata_json_column = metadata_json_column
        self.query_options = query_options
        self.db_name = self.__get_db_name()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_service

    def __get_db_name(self) -> str:
        result = self.engine._fetch("SELECT DATABASE();")
        return result[0]["DATABASE()"]

    def _add_embeddings(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        if not ids:
            ids = ["NULL" for _ in texts]
        if not metadatas:
            metadatas = [{} for _ in texts]
        # Insert embeddings
        for id, content, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            metadata_col_names = (
                ", " + ", ".join(self.metadata_columns)
                if len(self.metadata_columns) > 0
                else ""
            )
            insert_stmt = f"INSERT INTO `{self.table_name}`(`{self.id_column}`, `{self.content_column}`, `{self.embedding_column}`{metadata_col_names}"
            values = {"id": id, "content": content, "embedding": str(embedding)}
            values_stmt = "VALUES (:id, :content, string_to_vector(:embedding)"

            # Add metadata
            extra = metadata
            for metadata_column in self.metadata_columns:
                if metadata_column in metadata:
                    values_stmt += f", :{metadata_column}"
                    values[metadata_column] = metadata[metadata_column]
                    del extra[metadata_column]
                else:
                    values_stmt += ",null"

            # Add JSON column and/or close statement
            insert_stmt += (
                f", {self.metadata_json_column})" if self.metadata_json_column else ")"
            )
            if self.metadata_json_column:
                values_stmt += ", :extra)"
                values["extra"] = json.dumps(extra)
            else:
                values_stmt += ")"

            query = insert_stmt + values_stmt
            self.engine._execute(query, values)

        return ids

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        embeddings = self.embedding_service.embed_documents(list(texts))
        ids = self._add_embeddings(
            texts, embeddings, metadatas=metadatas, ids=ids, **kwargs
        )
        return ids

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = self.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return ids

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> bool:
        if not ids:
            return False

        id_list = ", ".join([f"'{id}'" for id in ids])
        query = (
            f"DELETE FROM `{self.table_name}` WHERE `{self.id_column}` in ({id_list})"
        )
        self.engine._execute(query)
        return True

    def apply_vector_index(self, vector_index: VectorIndex):
        # Construct the default index name
        if not vector_index.name:
            vector_index.name = f"{self.table_name}_{DEFAULT_INDEX_NAME_SUFFIX}"
        query_template = f"CALL mysql.create_vector_index('{vector_index.name}', '{self.db_name}.{self.table_name}', '{self.embedding_column}', '{{}}');"
        self.__exec_apply_vector_index(query_template, vector_index)
        # After applying an index to the table, set the query option search type to be ANN
        self.query_options.search_type = SearchType.ANN

    def alter_vector_index(self, vector_index: VectorIndex):
        existing_index_name = self._get_vector_index_name()
        if not existing_index_name:
            raise ValueError("No existing vector index found.")
        if not vector_index.name:
            vector_index.name = existing_index_name.split(".")[1]
        if existing_index_name.split(".")[1] != vector_index.name:
            raise ValueError(
                f"Existing index name {existing_index_name} does not match the new index name {vector_index.name}."
            )
        query_template = (
            f"CALL mysql.alter_vector_index('{existing_index_name}', '{{}}');"
        )
        self.__exec_apply_vector_index(query_template, vector_index)

    def __exec_apply_vector_index(self, query_template: str, vector_index: VectorIndex):
        index_options = []
        if vector_index.index_type:
            index_options.append(f"index_type={vector_index.index_type.value}")
        if vector_index.distance_measure:
            index_options.append(
                f"distance_measure={vector_index.distance_measure.value}"
            )
        if vector_index.num_partitions:
            index_options.append(f"num_partitions={vector_index.num_partitions}")
        if vector_index.num_neighbors:
            index_options.append(f"num_neighbors={vector_index.num_neighbors}")
        index_options_query = ",".join(index_options)

        stmt = query_template.format(index_options_query)
        self.engine._execute_outside_tx(stmt)

    def _get_vector_index_name(self):
        query = f"SELECT index_name FROM mysql.vector_indexes WHERE table_name='{self.db_name}.{self.table_name}';"
        result = self.engine._fetch(query)
        if result:
            return result[0]["index_name"]
        else:
            return None

    def drop_vector_index(self):
        existing_index_name = self._get_vector_index_name()
        if existing_index_name:
            self.engine._execute_outside_tx(
                f"CALL mysql.drop_vector_index('{existing_index_name}');"
            )
        self.query_options.search_type = SearchType.KNN
        return existing_index_name

    @classmethod
    def from_texts(  # type: ignore[override]
        cls: Type[MySQLVectorStore],
        texts: List[str],
        embedding: Embeddings,
        engine: MySQLEngine,
        table_name: str,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        query_options: QueryOptions = DEFAULT_QUERY_OPTIONS,
        **kwargs: Any,
    ):
        vs = cls(
            engine=engine,
            embedding_service=embedding,
            table_name=table_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            id_column=id_column,
            metadata_json_column=metadata_json_column,
            query_options=query_options,
        )
        vs.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return vs

    @classmethod
    def from_documents(  # type: ignore[override]
        cls: Type[MySQLVectorStore],
        documents: List[Document],
        embedding: Embeddings,
        engine: MySQLEngine,
        table_name: str,
        ids: Optional[List[str]] = None,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[str] = [],
        ignore_metadata_columns: Optional[List[str]] = None,
        id_column: str = "langchain_id",
        metadata_json_column: str = "langchain_metadata",
        query_options: QueryOptions = DEFAULT_QUERY_OPTIONS,
        **kwargs: Any,
    ) -> MySQLVectorStore:
        vs = cls(
            engine=engine,
            embedding_service=embedding,
            table_name=table_name,
            content_column=content_column,
            embedding_column=embedding_column,
            metadata_columns=metadata_columns,
            ignore_metadata_columns=ignore_metadata_columns,
            id_column=id_column,
            metadata_json_column=metadata_json_column,
            query_options=query_options,
        )
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        vs.add_texts(texts, metadatas=metadatas, ids=ids, **kwargs)
        return vs

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        **kwargs: Any,
    ):
        raise NotImplementedError
