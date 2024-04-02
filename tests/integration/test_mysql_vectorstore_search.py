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

from langchain_google_cloud_sql_mysql import (
    DistanceMeasure,
    IndexType,
    MySQLEngine,
    MySQLVectorStore,
    QueryOptions,
    SearchType,
    VectorIndex,
)

TABLE_1000_ROWS = "test_table_1000_rows_search"
VECTOR_SIZE = 8
DEFAULT_INDEX = VectorIndex(index_type=IndexType.TREE_SQ)

embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


class TestVectorStoreFromMethods:
    apple_100_text = "apple_100"
    embeddings_service = DeterministicFakeEmbedding(size=VECTOR_SIZE)
    apple_100_embedding = embeddings_service.embed_query(apple_100_text)

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
            embedding_service=self.embeddings_service,
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

    def test_search_query_collection_knn(self, vs_1000):
        result = vs_1000._query_collection(self.apple_100_embedding, k=10)
        assert len(result) == 10
        assert result[0]["content"] == self.apple_100_text

    def test_search_query_collection_knn_with_filter(self, vs_1000):
        vs_1000.drop_vector_index()
        result = vs_1000._query_collection(
            self.apple_100_embedding, k=5, filter=f"content != '{self.apple_100_text}'"
        )
        assert len(result) == 5
        assert result[0]["content"] == "apple_154"

    def test_search_query_collection_distance_measure(self, vs_1000):
        vs_1000.apply_vector_index(DEFAULT_INDEX)
        for measure in [
            DistanceMeasure.COSINE,
            DistanceMeasure.DOT_PRODUCT,
            DistanceMeasure.L2_SQUARED,
        ]:
            assert (
                vs_1000._query_collection(
                    self.apple_100_embedding,
                    query_options=QueryOptions(distance_measure=measure),
                )[0]["content"]
                == self.apple_100_text
            )
        vs_1000.drop_vector_index()

    def test_search_raise_when_num_partitions_set_for_knn(self, vs_1000):
        with pytest.raises(
            ValueError, match="num_partitions is not supported for the search type KNN"
        ):
            vs_1000._query_collection(
                self.apple_100_embedding,
                k=1,
                filter="content != 'apple_100'",
                query_options=QueryOptions(num_partitions=2),
            )

    def test_query_collection_ann_with_different_index_types(self, vs_1000):
        vs_1000.apply_vector_index(VectorIndex(index_type=IndexType.BRUTE_FORCE_SCAN))
        result = vs_1000._query_collection(self.apple_100_embedding)
        assert len(result) == 10
        assert result[0]["content"] == self.apple_100_text

        result = vs_1000._query_collection(self.apple_100_embedding, k=1)
        assert result[0]["content"] == self.apple_100_text

        vs_1000.alter_vector_index(VectorIndex(index_type=IndexType.TREE_SQ))
        result = vs_1000._query_collection(
            self.apple_100_embedding,
            k=5,
            query_options=QueryOptions(num_partitions=2, search_type=SearchType.ANN),
        )
        assert len(result) == 5
        assert result[0]["content"] == self.apple_100_text

        vs_1000.alter_vector_index(VectorIndex(index_type=IndexType.TREE_AH))
        result = vs_1000._query_collection(
            self.apple_100_embedding, k=5, filter=f"content != '{self.apple_100_text}'"
        )
        assert len(result) == 4
        assert result[0]["content"] == "apple_154"

    def test_similarity_search_with_score_by_vector(self, vs_1000):
        vs_1000.alter_vector_index(VectorIndex(index_type=IndexType.TREE_AH))
        docs_with_scores = vs_1000.similarity_search_with_score_by_vector(
            self.apple_100_embedding, k=5
        )
        assert len(docs_with_scores) == 5
        assert docs_with_scores[0][0].page_content == self.apple_100_text
        assert docs_with_scores[0][1] == 0

        docs_with_scores = vs_1000.similarity_search_with_score_by_vector(
            self.apple_100_embedding,
            k=1,
            query_options=QueryOptions(
                distance_measure=DistanceMeasure.DOT_PRODUCT, search_type=SearchType.KNN
            ),
        )
        assert len(docs_with_scores) == 1
        assert docs_with_scores[0][0].page_content == self.apple_100_text

    def test_similarity_search_by_vector(self, vs_1000):
        docs_with_scores = vs_1000.similarity_search_with_score_by_vector(
            self.apple_100_embedding, k=5
        )
        docs = vs_1000.similarity_search_by_vector(self.apple_100_embedding, k=5)
        assert [doc_with_score[0] for doc_with_score in docs_with_scores] == docs

    def test_similarity_search_with_score(self, vs_1000):
        docs_with_scores = vs_1000.similarity_search_with_score_by_vector(
            self.apple_100_embedding, k=5
        )
        docs_with_scores_from_text_search = vs_1000.similarity_search_with_score(
            self.apple_100_text, k=5
        )
        assert docs_with_scores == docs_with_scores_from_text_search

    def test_similarity_search(self, vs_1000):
        docs_with_scores = vs_1000.similarity_search_with_score_by_vector(
            self.apple_100_embedding,
            k=5,
            query_options=QueryOptions(num_partitions=2, search_type=SearchType.ANN),
        )
        docs = vs_1000.similarity_search(
            self.apple_100_text,
            k=5,
            query_options=QueryOptions(num_partitions=2, search_type=SearchType.ANN),
        )
        assert [doc_with_score[0] for doc_with_score in docs_with_scores] == docs

    def test_max_marginal_relevance_search_with_score_by_vector(self, vs_1000):
        docs_with_scores = vs_1000.max_marginal_relevance_search_with_score_by_vector(
            self.apple_100_embedding, k=5
        )
        assert len(docs_with_scores) == 5
        assert docs_with_scores[0][0].page_content == self.apple_100_text
        assert docs_with_scores[0][1] == 0

    def test_max_marginal_relevance_search_by_vector(self, vs_1000):
        docs_with_scores = vs_1000.max_marginal_relevance_search_with_score_by_vector(
            self.apple_100_embedding, k=5
        )
        docs = vs_1000.max_marginal_relevance_search_by_vector(
            self.apple_100_embedding, k=5
        )
        assert [doc_with_score[0] for doc_with_score in docs_with_scores] == docs

    def test_max_marginal_relevance_search(self, vs_1000):
        docs_with_scores = vs_1000.max_marginal_relevance_search_by_vector(
            self.apple_100_embedding, k=5
        )
        docs_with_scores_from_text_search = vs_1000.max_marginal_relevance_search(
            self.apple_100_text, k=5
        )
        assert docs_with_scores == docs_with_scores_from_text_search
