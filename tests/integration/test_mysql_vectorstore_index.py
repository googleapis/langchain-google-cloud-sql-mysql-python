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

from langchain_google_cloud_sql_mysql import (
    DistanceMeasure,
    IndexType,
    MySQLEngine,
    MySQLVectorStore,
    SearchType,
    VectorIndex,
)

DEFAULT_TABLE = "test_table_" + str(uuid.uuid4()).split("-")[0]
TABLE_1000_ROWS = "test_table_1000_rows"
VECTOR_SIZE = 8

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


class TestVectorStoreFromMethods:
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

    @pytest.fixture(scope="class")
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
        vs.drop_vector_index()
        engine._execute(f"DROP TABLE IF EXISTS `{DEFAULT_TABLE}`")

    @pytest.fixture(scope="class")
    def vs_1000(self, engine):
        result = engine._fetch("SHOW TABLES")
        tables = [list(r.values())[0] for r in result]
        if TABLE_1000_ROWS not in tables:
            engine.init_vectorstore_table(
                TABLE_1000_ROWS,
                VECTOR_SIZE,
            )
        vs_1000 = MySQLVectorStore(
            engine,
            embedding_service=embeddings_service,
            table_name=TABLE_1000_ROWS,
        )
        row_count = vs_1000.engine._fetch(f"SELECT count(*) FROM `{TABLE_1000_ROWS}`")[
            0
        ]["count(*)"]
        # Add 1000 rows of data if the number of rows is less than 1000
        if row_count < 1000:
            texts_1000 = [
                f"{text}_{i}"
                for text in ["apple", "dog", "basketball", "coffee"]
                for i in range(1, 251)
            ]
            ids = [str(uuid.uuid4()) for _ in range(len(texts_1000))]
            vs_1000.add_texts(texts_1000, ids=ids)
        vs_1000.drop_vector_index()
        yield vs_1000
        vs_1000.drop_vector_index()

    def test_create_and_drop_index(self, vs):
        vs.apply_vector_index(VectorIndex())
        assert (
            vs._get_vector_index_name()
            == f"{vs.db_name}.{vs.table_name}_langchainvectorindex"
        )
        assert vs.query_options.search_type == SearchType.ANN
        vs.drop_vector_index()
        assert vs._get_vector_index_name() is None
        assert vs.query_options.search_type == SearchType.KNN

    def test_update_index(self, vs):
        vs.apply_vector_index(VectorIndex())
        assert (
            vs._get_vector_index_name()
            == f"{vs.db_name}.{vs.table_name}_langchainvectorindex"
        )
        assert vs.query_options.search_type == SearchType.ANN
        vs.alter_vector_index(
            VectorIndex(
                index_type=IndexType.BRUTE_FORCE_SCAN,
                distance_measure=DistanceMeasure.L2_SQUARED,
                num_neighbors=10,
            )
        )
        assert (
            vs._get_vector_index_name()
            == f"{vs.db_name}.{vs.table_name}_langchainvectorindex"
        )
        vs.drop_vector_index()
        assert vs.query_options.search_type == SearchType.KNN

    def test_create_and_drop_index_tree_sq(self, vs_1000):
        vs_1000.apply_vector_index(
            VectorIndex(
                name="tree_sq",
                index_type=IndexType.TREE_SQ,
                distance_measure=DistanceMeasure.L2_SQUARED,
                num_partitions=1,
                num_neighbors=5,
            )
        )
        assert vs_1000._get_vector_index_name() == f"{vs_1000.db_name}.tree_sq"
        assert vs_1000.query_options.search_type == SearchType.ANN
        vs_1000.drop_vector_index()
        assert vs_1000._get_vector_index_name() is None
        assert vs_1000.query_options.search_type == SearchType.KNN

    def test_create_and_drop_index_tree_ah(self, vs_1000):
        vs_1000.apply_vector_index(
            VectorIndex(
                name="tree_ah",
                index_type=IndexType.TREE_AH,
                distance_measure=DistanceMeasure.COSINE,
                num_partitions=2,
                num_neighbors=10,
            )
        )
        assert vs_1000._get_vector_index_name() == f"{vs_1000.db_name}.tree_ah"
        assert vs_1000.query_options.search_type == SearchType.ANN
        vs_1000.drop_vector_index()
        assert vs_1000._get_vector_index_name() is None
        assert vs_1000.query_options.search_type == SearchType.KNN
