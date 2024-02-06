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
from langchain_core.documents import Document

from langchain_google_cloud_sql_mysql.mysql_loader import (
    DEFAULT_CONTENT_COL,
    DEFAULT_METADATA_COL,
    _parse_doc_from_row,
    _parse_row_from_doc,
)

test_doc = Document(
    page_content="Granny Smith 150 0.99",
    metadata={"fruit-id": 1, "fruit_name": "Apple", "organic": 1},
)
cols_default = [DEFAULT_CONTENT_COL, DEFAULT_METADATA_COL]
row_default = {
    "page_content": "Granny Smith 150 0.99",
    "langchain_metadata": {"fruit-id": 1, "fruit_name": "Apple", "organic": 1},
}
row_customized_flat = {
    "fruit-id": 1,
    "fruit_name": "Apple",
    "variety": "Granny Smith",
    "quantity_in_stock": 150,
    "price_per_unit": 0.99,
    "organic": 1,
}
row_customized_nested = {
    "variety": "Granny Smith",
    "quantity_in_stock": 150,
    "price_per_unit": 0.99,
    "langchain_metadata": {"fruit-id": 1, "fruit_name": "Apple", "organic": 1},
}


def test_row2doc_default():
    assert (
        _parse_doc_from_row([DEFAULT_CONTENT_COL], [DEFAULT_METADATA_COL], row_default)
        == test_doc
    )


def test_row2doc_customized():
    assert (
        _parse_doc_from_row(
            ["variety", "quantity_in_stock", "price_per_unit"],
            ["fruit-id", "fruit_name", "organic"],
            row_customized_flat,
        )
        == test_doc
    )


def test_row2doc_unnest_default_metadata():
    assert (
        _parse_doc_from_row(
            ["variety", "quantity_in_stock", "price_per_unit"],
            ["langchain_metadata"],
            row_customized_nested,
        )
        == test_doc
    )


def test_row2doc_ovrride_default_metadata():
    row_override = row_customized_nested.copy()
    row_override["fruit_name"] = "Banana"
    assert _parse_doc_from_row(
        ["quantity_in_stock", "price_per_unit"],
        ["fruit_name", "variety", "langchain_metadata"],
        row_override,
    ) == Document(
        page_content="150 0.99",
        metadata={
            "fruit-id": 1,
            "variety": "Granny Smith",
            "fruit_name": "Banana",
            "organic": 1,
        },
    )


def test_row2doc_metadata_col_nonexist():
    assert _parse_doc_from_row(
        ["variety", "quantity_in_stock", "price_per_unit"],
        ["fruit-id"],
        row_customized_nested,
    ) == Document(page_content="Granny Smith 150 0.99")


def test_doc2row_default():
    assert _parse_row_from_doc(cols_default, test_doc) == row_default


def test_doc2row_customized():
    assert _parse_row_from_doc([DEFAULT_CONTENT_COL, "fruit-id"], test_doc) == {
        "fruit-id": 1,
        "page_content": "Granny Smith 150 0.99",
    }


def test_doc2row_no_metadata():
    assert _parse_row_from_doc([DEFAULT_CONTENT_COL], test_doc) == {
        "page_content": "Granny Smith 150 0.99",
    }


def test_doc2row_store_extra_metadata():
    assert _parse_row_from_doc(
        [DEFAULT_CONTENT_COL, "fruit-id", DEFAULT_METADATA_COL], test_doc
    ) == {
        "fruit-id": 1,
        "page_content": "Granny Smith 150 0.99",
        "langchain_metadata": {
            "fruit_name": "Apple",
            "organic": 1,
        },
    }


def test_doc2row2doc_customized_metadata():
    customized_metadatas = [
        [],
        ["fruit-id"],
        ["fruit-id", "fruit_name", "organic"],
        ["fruit-id", "fruit_name", "organic", "other"],
    ]
    for metadata in customized_metadatas:
        assert test_doc == _parse_doc_from_row(
            [DEFAULT_CONTENT_COL],
            metadata + [DEFAULT_METADATA_COL],
            _parse_row_from_doc(metadata + [DEFAULT_METADATA_COL], test_doc),
        )
