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

from unittest.mock import MagicMock
import pytest
from langchain_community.embeddings import DeterministicFakeEmbedding

from langchain_google_cloud_sql_mysql import (
    DistanceMeasure,
    IndexType,
    MySQLVectorStore,
    QueryOptions,
    SearchType,
    VectorIndex,
)


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine._fetch.side_effect = lambda stmt, *args, **kwargs: (
        [{"DATABASE()": "test_db"}]
        if "DATABASE()" in stmt
        else [
            {"COLUMN_NAME": "langchain_id", "DATA_TYPE": "varchar"},
            {"COLUMN_NAME": "content", "DATA_TYPE": "text"},
            {"COLUMN_NAME": "embedding", "DATA_TYPE": "varbinary"},
        ]
        if "information_schema" in stmt.lower()
        else [{"COLUMN_NAME": "langchain_id"}, {"COLUMN_NAME": "content"}, {"COLUMN_NAME": "embedding"}]
        if "COLUMNS" in stmt
        else []
    )
    return engine


@pytest.fixture
def vectorstore(mock_engine):
    embedding_service = DeterministicFakeEmbedding(size=8)
    vs = MySQLVectorStore(
        engine=mock_engine,
        embedding_service=embedding_service,
        table_name="my_table",
    )
    mock_engine._execute_outside_tx.reset_mock()
    mock_engine._fetch.reset_mock()
    return vs


def test_apply_vector_index_ga_syntax(vectorstore, mock_engine):
    index = VectorIndex(
        name="my_idx",
        index_type=IndexType.TREE_SQ,
        distance_measure=DistanceMeasure.L2_SQUARED,
        num_partitions=10,
    )
    vectorstore.apply_vector_index(index)
    
    expected_sql = (
        "CREATE VECTOR INDEX `my_idx` ON `my_table` (`embedding`) "
        "USING SCANN QUANTIZER=SQ8 DISTANCE_MEASURE=l2_squared NUM_LEAVES=10;"
    )
    mock_engine._execute_outside_tx.assert_called_once_with(expected_sql)
    assert vectorstore.query_options.search_type == SearchType.ANN


def test_drop_vector_index_ga_syntax(vectorstore, mock_engine):
    vectorstore._get_vector_index_name = MagicMock(return_value="test_db.my_idx")
    vectorstore.drop_vector_index()
    
    expected_sql = "DROP INDEX `my_idx` ON `my_table`;"
    mock_engine._execute_outside_tx.assert_called_once_with(expected_sql)
    assert vectorstore.query_options.search_type == SearchType.KNN


def test_alter_vector_index_ga_syntax(vectorstore, mock_engine):
    vectorstore._get_vector_index_name = MagicMock(return_value="test_db.my_idx")
    new_index = VectorIndex(
        name="my_idx",
        index_type=IndexType.TREE_SQ,
        distance_measure=DistanceMeasure.COSINE,
        num_partitions=20,
    )
    vectorstore.alter_vector_index(new_index)
    
    drop_sql = "DROP INDEX `my_idx` ON `my_table`;"
    create_sql = (
        "CREATE VECTOR INDEX `my_idx` ON `my_table` (`embedding`) "
        "USING SCANN QUANTIZER=SQ8 DISTANCE_MEASURE=cosine NUM_LEAVES=20;"
    )
    mock_engine._execute_outside_tx.assert_any_call(drop_sql)
    mock_engine._execute_outside_tx.assert_any_call(create_sql)


def test_query_collection_knn(vectorstore, mock_engine):
    vectorstore.__get_column_names = MagicMock(
        return_value=["langchain_id", "content", "vector_to_string(embedding) as embedding"]
    )
    query_options = QueryOptions(
        distance_measure=DistanceMeasure.L2_SQUARED,
        search_type=SearchType.KNN,
    )
    vectorstore._query_collection(
        embedding=[0.1, 0.2, 0.3],
        k=5,
        filter="content != 'test'",
        query_options=query_options,
    )
    
    called_stmt = mock_engine._fetch.call_args[0][0]
    expected_stmt = (
        "SELECT langchain_id, content, vector_to_string(embedding) as embedding, "
        "l2_squared_distance(embedding, string_to_vector(:embedding)) AS distance "
        "FROM `my_table` WHERE content != 'test' ORDER BY distance LIMIT 5;"
    )
    assert called_stmt == expected_stmt


def test_query_collection_ann_ga_syntax(vectorstore, mock_engine):
    vectorstore.__get_column_names = MagicMock(
        return_value=["langchain_id", "content", "vector_to_string(embedding) as embedding"]
    )
    query_options = QueryOptions(
        distance_measure=DistanceMeasure.COSINE,
        search_type=SearchType.ANN,
        num_partitions=10,
    )
    vectorstore._query_collection(
        embedding=[0.1, 0.2, 0.3],
        k=4,
        filter="content != 'test'",
        query_options=query_options,
    )
    
    called_stmt = mock_engine._fetch.call_args[0][0]
    expected_stmt = (
        "SELECT langchain_id, content, vector_to_string(embedding) as embedding, "
        "APPROX_DISTANCE(embedding, string_to_vector(:embedding), 'distance_measure=cosine,num_leaves_to_search=10') AS distance "
        "FROM `my_table` WHERE content != 'test' ORDER BY distance LIMIT 4;"
    )
    assert called_stmt == expected_stmt
