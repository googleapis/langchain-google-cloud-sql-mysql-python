# Copyright 2026 Google LLC
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

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import sqlalchemy
from langchain_community.embeddings import DeterministicFakeEmbedding

from langchain_google_cloud_sql_mysql import (
    DistanceMeasure,
    MySQLEngine,
    MySQLVectorStore,
    QueryOptions,
    SearchType,
)


class _FakeRow(NamedTuple):
    langchain_id: str
    content: str
    embedding: str
    langchain_metadata: Optional[str]
    distance: float


class _FakeMySQLEngine(MySQLEngine):
    """In-memory fake MySQLEngine recording executed SQL statements and bind params."""

    def __init__(self, index_name: Optional[str] = None) -> None:
        super().__init__(engine=sqlalchemy.create_engine("sqlite:///:memory:"))
        self.executed: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        self.executed_outside_tx: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        self.fetched: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        self.fetched_rows: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        self._index_name = index_name

    def _execute(self, query: str, params: Optional[dict] = None) -> None:
        self.executed.append((query, params))

    def _execute_outside_tx(self, query: str, params: Optional[dict] = None) -> None:
        self.executed_outside_tx.append((query, params))

    def _fetch(self, query: str, params: Optional[dict] = None) -> List[Dict[str, Any]]:
        self.fetched.append((query, params))
        if "SELECT DATABASE()" in query:
            return [{"DATABASE()": "test_db"}]
        if "information_schema.columns" in query:
            return [
                {"COLUMN_NAME": "langchain_id", "DATA_TYPE": "char"},
                {"COLUMN_NAME": "content", "DATA_TYPE": "text"},
                {"COLUMN_NAME": "embedding", "DATA_TYPE": "varbinary"},
                {"COLUMN_NAME": "langchain_metadata", "DATA_TYPE": "json"},
            ]
        if "`INFORMATION_SCHEMA`.`COLUMNS`" in query:
            return [
                {"COLUMN_NAME": "langchain_id"},
                {"COLUMN_NAME": "content"},
                {"COLUMN_NAME": "embedding"},
                {"COLUMN_NAME": "langchain_metadata"},
            ]
        if "innodb_vector_indexes" in query:
            if self._index_name:
                return [{"INDEX_NAME": self._index_name}]
            return []
        return []

    def _fetch_rows(self, query: str, params: Optional[dict] = None) -> List[_FakeRow]:
        self.fetched_rows.append((query, params))
        return [
            _FakeRow(
                langchain_id="doc-1",
                content="hello",
                embedding="[0.1, 0.2]",
                langchain_metadata='{"source": "unit"}',
                distance=0.05,
            )
        ]


def test_init_parameterizes_table_name_lookup() -> None:
    engine = _FakeMySQLEngine()

    MySQLVectorStore(
        engine=engine,
        embedding_service=DeterministicFakeEmbedding(size=2),
        table_name="vectors",
    )

    assert engine.fetched[0] == (
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = :table_name",
        {"table_name": "vectors"},
    )


def test_delete_binds_ids_instead_of_string_interpolation() -> None:
    engine = _FakeMySQLEngine()
    store = MySQLVectorStore(
        engine=engine,
        embedding_service=DeterministicFakeEmbedding(size=2),
        table_name="vectors",
    )
    injection_payload = "x') OR ('1'='1"

    deleted = store.delete(ids=[injection_payload, "doc-2"])

    assert deleted is True
    assert engine.executed == [
        (
            "DELETE FROM `vectors` WHERE `langchain_id` in (:id_0, :id_1)",
            {"id_0": injection_payload, "id_1": "doc-2"},
        )
    ]


def test_delete_empty_ids_returns_false_without_executing_sql() -> None:
    engine = _FakeMySQLEngine()
    store = MySQLVectorStore(
        engine=engine,
        embedding_service=DeterministicFakeEmbedding(size=2),
        table_name="vectors",
    )

    deleted = store.delete(ids=[])

    assert deleted is False
    assert engine.executed == []


def test_drop_vector_index_binds_table_name_in_index_lookup() -> None:
    engine = _FakeMySQLEngine(index_name="vectors_langchainvectorindex")
    store = MySQLVectorStore(
        engine=engine,
        embedding_service=DeterministicFakeEmbedding(size=2),
        table_name="vectors",
    )
    engine.fetched.clear()

    dropped = store.drop_vector_index()

    assert dropped == "vectors_langchainvectorindex"
    assert engine.fetched == [
        (
            "SELECT INDEX_NAME FROM information_schema.innodb_vector_indexes WHERE TABLE_NAME=:table_name;",
            {"table_name": "test_db.vectors"},
        )
    ]
    assert engine.executed_outside_tx == [
        ("DROP INDEX `vectors_langchainvectorindex` ON `test_db`.`vectors`;", None)
    ]


def test_similarity_search_by_vector_binds_embedding_for_knn_and_ann() -> None:
    engine = _FakeMySQLEngine()
    store = MySQLVectorStore(
        engine=engine,
        embedding_service=DeterministicFakeEmbedding(size=2),
        table_name="vectors",
    )
    engine.fetched.clear()

    docs = store.similarity_search_by_vector(
        embedding=[0.25, 0.75],
        k=3,
        query_options=QueryOptions(
            distance_measure=DistanceMeasure.COSINE,
            search_type=SearchType.ANN,
            num_leaves_to_search=4,
        ),
    )

    assert len(docs) == 1
    assert docs[0].page_content == "hello"
    assert engine.fetched == [
        (
            "SELECT COLUMN_NAME FROM `INFORMATION_SCHEMA`.`COLUMNS` WHERE `TABLE_SCHEMA` = :db_name AND `TABLE_NAME` = :table_name",
            {"db_name": "test_db", "table_name": "vectors"},
        )
    ]
    assert engine.fetched_rows == [
        (
            "SELECT langchain_id, content, vector_to_string(embedding) as embedding, langchain_metadata, "
            "APPROX_DISTANCE(embedding, string_to_vector(:embedding), 'distance_measure=cosine,num_leaves_to_search=4') AS distance "
            "FROM `vectors`  ORDER BY distance LIMIT 3;",
            {"embedding": "[0.25, 0.75]"},
        )
    ]
