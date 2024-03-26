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

import os
import uuid

import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.documents import Document

from langchain_google_cloud_sql_mysql import Column, MySQLEngine, MySQLVectorStore

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
CUSTOM_TABLE = "test-table-custom" + str(uuid.uuid4())
VECTOR_SIZE = 768

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)

texts = ["foo", "bar", "baz"]
metadatas = [{"page": str(i), "source": "google.com"} for i in range(len(texts))]
docs = [
    Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))
]

embeddings = [embeddings_service.embed_query(texts[i]) for i in range(len(texts))]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


class TestVectorStore:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for cloud sql instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for cloud sql")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DB_NAME", "database name on cloud sql instance")

    @pytest.fixture(scope="class")
    def engine(self, db_project, db_region, db_instance, db_name):
        engine = MySQLEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )

        yield engine

    @pytest.fixture(scope="function")
    def vs(self, engine):
        engine.init_vectorstore_table(
            DEFAULT_TABLE,
            VECTOR_SIZE,
            overwrite_existing=True,
        )

        vs = MySQLVectorStore(
            engine,
            embedding_service=embeddings_service,
            table_name=DEFAULT_TABLE,
        )
        yield vs
        engine._execute(f"DROP TABLE IF EXISTS `{DEFAULT_TABLE}`")

    @pytest.fixture(scope="function")
    def vs_custom(self, engine):
        engine.init_vectorstore_table(
            CUSTOM_TABLE,
            VECTOR_SIZE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=[Column("page", "TEXT"), Column("source", "TEXT")],
            metadata_json_column="mymeta",
            overwrite_existing=True,
        )

        vs = MySQLVectorStore(
            engine,
            embedding_service=embeddings_service,
            table_name=CUSTOM_TABLE,
            id_column="myid",
            content_column="mycontent",
            embedding_column="myembedding",
            metadata_columns=["page", "source"],
            metadata_json_column="mymeta",
        )
        yield vs
        engine._execute(f"DROP TABLE IF EXISTS `{CUSTOM_TABLE}`")

    def test_post_init(self, engine):
        with pytest.raises(ValueError):
            MySQLVectorStore(
                engine,
                embedding_service=embeddings_service,
                table_name=CUSTOM_TABLE,
                id_column="myid",
                content_column="noname",
                embedding_column="myembedding",
                metadata_columns=["page", "source"],
                metadata_json_column="mymeta",
            )

    def test_add_texts(self, engine, vs):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs.add_texts(texts, ids=ids)
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 3

        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs.add_texts(texts, metadatas, ids)
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 6
        engine._execute(f"TRUNCATE TABLE `{DEFAULT_TABLE}`")

    def test_add_texts_edge_cases(self, engine, vs):
        texts = ["Taylor's", '"Swift"', "best-friend"]
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs.add_texts(texts, ids=ids)
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 3
        engine._execute(f"TRUNCATE TABLE `{DEFAULT_TABLE}`")

    def test_add_docs(self, engine, vs):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        vs.add_documents(docs, ids=ids)
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 3
        engine._execute(f"TRUNCATE TABLE `{DEFAULT_TABLE}`")

    def test_add_embedding(self, engine, vs):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs._add_embeddings(texts, embeddings, metadatas, ids)
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 3
        engine._execute(f"TRUNCATE TABLE `{DEFAULT_TABLE}`")

    def test_delete(self, engine, vs):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs.add_texts(texts, ids=ids)
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 3
        # delete an ID
        vs.delete([ids[0]])
        results = engine._fetch(f"SELECT * FROM `{DEFAULT_TABLE}`")
        assert len(results) == 2

    def test_add_texts_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs_custom.add_texts(texts, ids=ids)
        results = engine._fetch(f"SELECT * FROM `{CUSTOM_TABLE}`")
        content = [result["mycontent"] for result in results]
        assert len(results) == 3
        assert "foo" in content
        assert "bar" in content
        assert "baz" in content
        assert results[0]["myembedding"]
        assert results[0]["page"] is None
        assert results[0]["source"] is None

        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs_custom.add_texts(texts, metadatas, ids)
        results = engine._fetch(f"SELECT * FROM `{CUSTOM_TABLE}`")
        assert len(results) == 6
        engine._execute(f"TRUNCATE TABLE `{CUSTOM_TABLE}`")

    def test_add_docs_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for i in range(len(texts))]
        docs = [
            Document(
                page_content=texts[i],
                metadata={"page": str(i), "source": "google.com"},
            )
            for i in range(len(texts))
        ]
        vs_custom.add_documents(docs, ids=ids)

        results = engine._fetch(f"SELECT * FROM `{CUSTOM_TABLE}`")
        content = [result["mycontent"] for result in results]
        assert len(results) == 3
        assert "foo" in content
        assert "bar" in content
        assert "baz" in content
        assert results[0]["myembedding"]
        pages = [result["page"] for result in results]
        assert "0" in pages
        assert "1" in pages
        assert "2" in pages
        assert results[0]["source"] == "google.com"
        engine._execute(f"TRUNCATE TABLE `{CUSTOM_TABLE}`")

    def test_add_embedding_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs_custom._add_embeddings(texts, embeddings, metadatas, ids)
        results = engine._fetch(f"SELECT * FROM `{CUSTOM_TABLE}`")
        assert len(results) == 3
        engine._execute(f"TRUNCATE TABLE `{CUSTOM_TABLE}`")

    def test_delete_custom(self, engine, vs_custom):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        vs_custom.add_texts(texts, ids=ids)
        results = engine._fetch(f"SELECT * FROM `{CUSTOM_TABLE}`")
        content = [result["mycontent"] for result in results]
        assert len(results) == 3
        assert "foo" in content
        # delete an ID
        vs_custom.delete([ids[0]])
        results = engine._fetch(f"SELECT * FROM `{CUSTOM_TABLE}`")
        content = [result["mycontent"] for result in results]
        assert len(results) == 2
        assert "foo" not in content

    # Need tests for store metadata=False
