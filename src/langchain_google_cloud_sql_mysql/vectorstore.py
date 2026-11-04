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
from typing import Any, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .engine import MySQLEngine
from .indexes import (
    DEFAULT_QUERY_OPTIONS,
    DistanceMeasure,
    QueryOptions,
    SearchType,
    VectorIndex,
)
from .loader import _parse_doc_from_row

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
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        query_options: QueryOptions = DEFAULT_QUERY_OPTIONS,
    ):
        """Constructor for MySQLVectorStore.
        Args:
            engine (MySQLEngine): Connection pool engine for managing connections to Cloud SQL for MySQL database.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of an existing table or table to be created.
            content_column (str): Column that represent a Document's page_content. Defaults to "content".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the document value. Defaults to "embedding".
            metadata_columns (List[str]): Column(s) that represent a document's metadata.
            ignore_metadata_columns (List[str]): Column(s) to ignore in pre-existing tables for a document's metadata. Can not be used with metadata_columns. Defaults to None.
            id_column (str): Column that represents the Document's id. Defaults to "langchain_id".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "langchain_metadata".
            k (int): The number of documents to return as the final result of a similarity search. Defaults to 4.
            fetch_k (int): The number of documents to initially retrieve from the database during a similarity search. These documents are then re-ranked using MMR to select the final `k` documents. Defaults to 20.
            lambda_mult (float): The weight used to balance relevance and diversity in the MMR algorithm. A higher value emphasizes diversity more, while a lower value prioritizes relevance. Defaults to 0.5.
            query_options: Additional query options.
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
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.query_options = query_options
        self.db_name = self.__get_db_name()

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_service

    def __get_db_name(self) -> str:
        result = self.engine._fetch("SELECT DATABASE();")
        return result[0]["DATABASE()"]

    def __get_column_names(self) -> List[str]:
        results = self.engine._fetch(
            f"SELECT COLUMN_NAME FROM `INFORMATION_SCHEMA`.`COLUMNS` WHERE `TABLE_SCHEMA` = '{self.db_name}' AND `TABLE_NAME` = '{self.table_name}'"
        )
        return [r["COLUMN_NAME"] for r in results]

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
    ) -> List[Document]:
        """Searches for similar documents based on a text query.

        Args:
            query: The text query to search for.
            k: The number of similar documents to return.
            filter: A filter expression to apply to the search results.
            **kwargs: Additional keyword arguments to pass to the search function.

        Returns:
            A list of similar documents.
        """
        embedding = self.embedding_service.embed_query(query)
        docs = self.similarity_search_by_vector(
            embedding=embedding, k=k, filter=filter, **kwargs
        )
        return docs

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        query_options: Optional[QueryOptions] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Searches for similar documents based on a vector embedding.

        Args:
            embedding: The vector embedding to search for.
            k: The number of similar documents to return.
            filter: A filter expression to apply to the search results.
            query_options: Additional query options.
            **kwargs: Additional keyword arguments to pass to the search function.

        Returns:
            A list of similar documents.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            query_options=query_options,
            **kwargs,
        )

        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[str] = None,
        query_options: Optional[QueryOptions] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Searches for similar documents based on a text query and returns their scores.

        Args:
            query: The text query to search for.
            k: The number of similar documents to return.
            filter: A filter expression to apply to the search results.
            query_options: Additional query options.
            **kwargs: Additional keyword arguments to pass to the search function.

        Returns:
            A list of tuples, where each tuple contains a document and its similarity score.
        """
        embedding = self.embedding_service.embed_query(query)
        docs_with_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            query_options=query_options,
            **kwargs,
        )
        return docs_with_scores

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        query_options: Optional[QueryOptions] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Searches for similar documents based on a vector embedding and returns their scores.

        Args:
            embedding: The vector embedding to search for.
            k: The number of similar documents to return.
            filter: A filter expression to apply to the search results.
            query_options: Additional query options.
            **kwargs: Additional keyword arguments to pass to the search function.

        Returns:
            A list of tuples, where each tuple contains a document and its similarity score.
        """
        results = self._query_collection(
            embedding=embedding,
            k=k,
            filter=filter,
            map_results=False,
            query_options=query_options,
            **kwargs,
        )

        documents_with_scores = []

        for row in results:
            row = row._asdict()
            if row.get(self.metadata_json_column):
                row[self.metadata_json_column] = json.loads(
                    row[self.metadata_json_column]
                )
            document = _parse_doc_from_row(
                content_columns=[self.content_column],
                metadata_columns=self.metadata_columns,
                row=row,
                metadata_json_column=self.metadata_json_column,
            )

            documents_with_scores.append(
                (
                    document,
                    row["distance"],
                )
            )

        return documents_with_scores

    def max_marginal_relevance_search(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        query_options: Optional[QueryOptions] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Performs Maximal Marginal Relevance (MMR) search based on a text query.

        Args:
            query: The text query to search for.
            k: The number of documents to return.
            fetch_k: The number of documents to initially retrieve.
            lambda_mult: The weight for balancing relevance and diversity.
            filter: A filter expression to apply to the search results.
            query_options: Additional query options.
            **kwargs: Additional keyword arguments to pass to the search function.

        Returns:
            A list of documents selected using MMR.
        """
        embedding = self.embedding_service.embed_query(text=query)

        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            query_options=query_options,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        query_options: Optional[QueryOptions] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Performs Maximal Marginal Relevance (MMR) search based on a vector embedding.

        Args:
            embedding: The vector embedding to search for.
            k: The number of documents to return.
            fetch_k: The number of documents to initially retrieve.
            lambda_mult: The weight for balancing relevance and diversity.
            filter: A filter expression to apply to the search results.
            query_options: Additional query options.
            **kwargs: Additional keyword arguments to pass to the search function.

        Returns:
            A list of documents selected using MMR.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            query_options=query_options,
            **kwargs,
        )

        return [result[0] for result in docs_and_scores]

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[str] = None,
        query_options: Optional[QueryOptions] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Performs Maximal Marginal Relevance (MMR) search based on a vector embedding and returns documents with scores.

        Args:
            embedding: The vector embedding to search for.
            k: The number of documents to return.
            fetch_k: The number of documents to initially retrieve.
            lambda_mult: The weight for balancing relevance and diversity.
            filter: A filter expression to apply to the search results.
            query_options: Additional query options.
            **kwargs: Additional keyword arguments to pass to the search function.

        Returns:
            A list of tuples, where each tuple contains a document and its similarity score, selected using MMR.
        """
        results = self._query_collection(
            embedding=embedding,
            k=fetch_k,
            filter=filter,
            map_results=False,
            query_options=query_options,
            **kwargs,
        )
        results = [row._asdict() for row in results]

        k = k if k else self.k
        fetch_k = fetch_k if fetch_k else self.fetch_k
        lambda_mult = lambda_mult if lambda_mult else self.lambda_mult
        embedding_list = [json.loads(row[self.embedding_column]) for row in results]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        documents_with_scores = []
        for row in results:
            if row.get(self.metadata_json_column):
                row[self.metadata_json_column] = json.loads(
                    row[self.metadata_json_column]
                )
            document = _parse_doc_from_row(
                content_columns=[self.content_column],
                metadata_columns=self.metadata_columns,
                row=row,
                metadata_json_column=self.metadata_json_column,
            )

            documents_with_scores.append(
                (
                    document,
                    row["distance"],
                )
            )

        return [r for i, r in enumerate(documents_with_scores) if i in mmr_selected]

    def _query_collection(
        self,
        embedding: List[float],
        k: Optional[int] = None,
        filter: Optional[str] = None,
        query_options: Optional[QueryOptions] = None,
        map_results: Optional[bool] = True,
    ) -> List[Any]:
        column_names = self.__get_column_names()
        # Apply vector_to_string to the embedding_column
        for i, v in enumerate(column_names):
            if v == self.embedding_column:
                column_names[i] = f"vector_to_string({v}) as {self.embedding_column}"
        column_query = ", ".join(column_names)
        query_options = query_options if query_options else self.query_options
        if query_options.num_partitions and query_options.search_type == SearchType.KNN:
            raise ValueError("num_partitions is not supported for the search type KNN")

        k = k if k else query_options.num_neighbors
        distance_function = (
            f"{query_options.distance_measure.value}_distance"
            if query_options.distance_measure != DistanceMeasure.DOT_PRODUCT
            else query_options.distance_measure.value
        )
        if query_options.search_type == SearchType.KNN:
            filter = f"WHERE {filter}" if filter else ""
            stmt = f"SELECT {column_query}, {distance_function}({self.embedding_column}, string_to_vector('{embedding}')) AS distance FROM `{self.table_name}` {filter} ORDER BY distance LIMIT {k};"
        else:
            filter = f"AND {filter}" if filter else ""
            num_partitions = (
                f",num_partitions={query_options.num_partitions}"
                if query_options.num_partitions
                else ""
            )
            stmt = f"SELECT {column_query}, {distance_function}({self.embedding_column}, string_to_vector('{embedding}')) AS distance FROM `{self.table_name}` WHERE NEAREST({self.embedding_column}) TO (string_to_vector('{embedding}'), 'num_neighbors={k}{num_partitions}') {filter} ORDER BY distance;"

        # return self.engine._fetch(stmt)
        if map_results:
            return self.engine._fetch(stmt)
        else:
            return self.engine._fetch_rows(stmt)


### The following is copied from langchain-community until it's moved into core

Matrix = Union[List[List[float]], List[np.ndarray], np.ndarray]


def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: list,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """Calculate maximal marginal relevance."""
    if min(k, len(embedding_list)) <= 0:
        return []
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    similarity_to_query = cosine_similarity(query_embedding, embedding_list)[0]
    most_similar = int(np.argmax(similarity_to_query))
    idxs = [most_similar]
    selected = np.array([embedding_list[most_similar]])
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -np.inf
        idx_to_add = -1
        similarity_to_selected = cosine_similarity(embedding_list, selected)
        for i, query_score in enumerate(similarity_to_query):
            if i in idxs:
                continue
            redundant_score = max(similarity_to_selected[i])
            equation_score = (
                lambda_mult * query_score - (1 - lambda_mult) * redundant_score
            )
            if equation_score > best_score:
                best_score = equation_score
                idx_to_add = i
        idxs.append(idx_to_add)
        selected = np.append(selected, [embedding_list[idx_to_add]], axis=0)
    return idxs


def cosine_similarity(X: Matrix, Y: Matrix) -> np.ndarray:
    """Row-wise cosine similarity between two equal-width matrices."""
    if len(X) == 0 or len(Y) == 0:
        return np.array([])

    X = np.array(X)
    Y = np.array(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"Number of columns in X and Y must be the same. X has shape {X.shape} "
            f"and Y has shape {Y.shape}."
        )
    try:
        import simsimd as simd  # type: ignore

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        Z = 1 - simd.cdist(X, Y, metric="cosine")
        if isinstance(Z, float):
            return np.array([Z])
        return Z
    except ImportError:
        X_norm = np.linalg.norm(X, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1)
        # Ignore divide by zero errors run time warnings as those are handled below.
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = np.dot(X, Y.T) / np.outer(X_norm, Y_norm)
        similarity[np.isnan(similarity) | np.isinf(similarity)] = 0.0
        return similarity
