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
import json
import os
from typing import Generator

import pytest
import sqlalchemy
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document

from langchain_google_cloud_sql_mysql import (
    MySQLDocumentSaver,
    MySQLEngine,
    MySQLLoader,
)

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
instance_id = os.environ["INSTANCE_ID"]
table_name = os.environ["TABLE_NAME"]
db_name = os.environ["DB_NAME"]


@pytest.fixture(name="engine")
def setup() -> Generator:
    engine = MySQLEngine.from_instance(
        project_id=project_id, region=region, instance=instance_id, database=db_name
    )
    yield engine

    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS `{table_name}`"))
        conn.commit()


@pytest.fixture
def default_setup(engine):
    with engine.connect() as conn:
        conn.execute(
            sqlalchemy.text(
                f"""
                CREATE TABLE IF NOT EXISTS `{table_name}`(
                    fruit_id INT AUTO_INCREMENT PRIMARY KEY,
                    fruit_name VARCHAR(100) NOT NULL,
                    variety VARCHAR(50),  
                    quantity_in_stock INT NOT NULL,
                    price_per_unit DECIMAL(6,2) NOT NULL,
                    organic TINYINT(1) NOT NULL
                )
                """
            )
        )
        conn.commit()
    yield engine


def test_load_from_query_default(default_setup):
    with default_setup.connect() as conn:
        conn.execute(
            sqlalchemy.text(
                f"""
                INSERT INTO `{table_name}` (fruit_name, variety, quantity_in_stock, price_per_unit, organic)
                VALUES
                    ('Apple', 'Granny Smith', 150, 1, 1);
                """
            )
        )
        conn.commit()
    query = f"SELECT * FROM `{table_name}`;"
    loader = MySQLLoader(
        engine=default_setup,
        query=query,
    )

    documents = loader.load()
    assert documents == [
        Document(
            page_content="1",
            metadata={
                "fruit_name": "Apple",
                "variety": "Granny Smith",
                "quantity_in_stock": 150,
                "price_per_unit": 1,
                "organic": 1,
            },
        )
    ]


def test_load_from_query_customized_content_customized_metadata(default_setup):
    with default_setup.connect() as conn:
        conn.execute(
            sqlalchemy.text(
                f"""
                INSERT INTO `{table_name}` (fruit_name, variety, quantity_in_stock, price_per_unit, organic)
                VALUES
                    ('Apple', 'Granny Smith', 150, 0.99, 1),
                    ('Banana', 'Cavendish', 200, 0.59, 0),
                    ('Orange', 'Navel', 80, 1.29, 1);
                """
            )
        )
        conn.commit()
    query = f"SELECT * FROM `{table_name}`;"
    loader = MySQLLoader(
        engine=default_setup,
        query=query,
        content_columns=[
            "fruit_name",
            "variety",
            "quantity_in_stock",
            "price_per_unit",
            "organic",
        ],
        metadata_columns=["fruit_id"],
    )

    documents = loader.load()

    assert documents == [
        Document(
            page_content="Apple Granny Smith 150 0.99 1",
            metadata={"fruit_id": 1},
        ),
        Document(
            page_content="Banana Cavendish 200 0.59 0",
            metadata={"fruit_id": 2},
        ),
        Document(
            page_content="Orange Navel 80 1.29 1",
            metadata={"fruit_id": 3},
        ),
    ]


def test_load_from_query_customized_content_default_metadata(default_setup):
    with default_setup.connect() as conn:
        conn.execute(
            sqlalchemy.text(
                f"""
                INSERT INTO `{table_name}` (fruit_name, variety, quantity_in_stock, price_per_unit, organic)
                VALUES
                    ('Apple', 'Granny Smith', 150, 0.99, 1);
                """
            )
        )
        conn.commit()
    query = f"SELECT * FROM `{table_name}`;"
    loader = MySQLLoader(
        engine=default_setup,
        query=query,
        content_columns=[
            "variety",
            "quantity_in_stock",
            "price_per_unit",
        ],
    )

    documents = loader.load()
    assert documents == [
        Document(
            page_content="Granny Smith 150 0.99",
            metadata={
                "fruit_id": 1,
                "fruit_name": "Apple",
                "organic": 1,
            },
        )
    ]


def test_load_from_query_default_content_customized_metadata(default_setup):
    with default_setup.connect() as conn:
        conn.execute(
            sqlalchemy.text(
                f"""
                INSERT INTO `{table_name}` (fruit_name, variety, quantity_in_stock, price_per_unit, organic)
                VALUES
                    ('Apple', 'Granny Smith', 150, 1, 1);
                """
            )
        )
        conn.commit()

    query = f"SELECT * FROM `{table_name}`;"
    loader = MySQLLoader(
        engine=default_setup,
        query=query,
        metadata_columns=[
            "fruit_name",
            "organic",
        ],
    )

    documents = loader.load()
    assert documents == [
        Document(
            page_content="1",
            metadata={
                "fruit_name": "Apple",
                "organic": 1,
            },
        )
    ]


def test_load_from_query_with_langchain_metadata(engine):
    with engine.connect() as conn:
        conn.execute(
            sqlalchemy.text(
                f"""
                CREATE TABLE IF NOT EXISTS `{table_name}`(
                    fruit_id INT AUTO_INCREMENT PRIMARY KEY,
                    fruit_name VARCHAR(100) NOT NULL,
                    variety VARCHAR(50),
                    quantity_in_stock INT NOT NULL,
                    price_per_unit DECIMAL(6,2) NOT NULL,
                    langchain_metadata JSON NOT NULL
                )
                """
            )
        )
        metadata = json.dumps({"organic": 1})
        conn.execute(
            sqlalchemy.text(
                f"""
                INSERT INTO `{table_name}` (fruit_name, variety, quantity_in_stock, price_per_unit, langchain_metadata)
                VALUES
                    ('Apple', 'Granny Smith', 150, 1, '{metadata}');
                """
            )
        )
        conn.commit()
    query = f"SELECT * FROM `{table_name}`;"
    loader = MySQLLoader(
        engine=engine,
        query=query,
        metadata_columns=[
            "fruit_name",
            "langchain_metadata",
        ],
    )

    documents = loader.load()
    assert documents == [
        Document(
            page_content="1",
            metadata={
                "fruit_name": "Apple",
                "organic": 1,
            },
        )
    ]


def test_save_doc_with_default_metadata(engine):
    engine.init_document_table(table_name)
    test_docs = [
        Document(
            page_content="Apple Granny Smith 150 0.99 1",
            metadata={"fruit_id": 1},
        ),
        Document(
            page_content="Banana Cavendish 200 0.59 0",
            metadata={"fruit_id": 2},
        ),
        Document(
            page_content="Orange Navel 80 1.29 1",
            metadata={"fruit_id": 3},
        ),
    ]
    saver = MySQLDocumentSaver(engine=engine, table_name=table_name)
    loader = MySQLLoader(engine=engine, table_name=table_name)

    saver.add_documents(test_docs)
    docs = loader.load()

    assert docs == test_docs
    assert engine._load_document_table(table_name).columns.keys() == [
        "page_content",
        "langchain_metadata",
    ]


@pytest.mark.parametrize("store_metadata", [True, False])
def test_save_doc_with_customized_metadata(engine, store_metadata):
    engine.init_document_table(
        table_name,
        metadata_columns=[
            sqlalchemy.Column(
                "fruit_name",
                sqlalchemy.UnicodeText,
                primary_key=False,
                nullable=True,
            ),
            sqlalchemy.Column(
                "organic",
                sqlalchemy.Boolean,
                primary_key=False,
                nullable=True,
            ),
        ],
        store_metadata=store_metadata,
    )
    test_docs = [
        Document(
            page_content="Granny Smith 150 0.99",
            metadata={"fruit_id": 1, "fruit_name": "Apple", "organic": 1},
        ),
    ]
    saver = MySQLDocumentSaver(engine=engine, table_name=table_name)
    loader = MySQLLoader(
        engine=engine,
        table_name=table_name,
        metadata_columns=[
            "fruit_id",
            "fruit_name",
            "organic",
        ],
    )

    saver.add_documents(test_docs)
    docs = loader.load()

    if store_metadata:
        docs == test_docs
        assert engine._load_document_table(table_name).columns.keys() == [
            "page_content",
            "fruit_name",
            "organic",
            "langchain_metadata",
        ]
    else:
        assert docs == [
            Document(
                page_content="Granny Smith 150 0.99",
                metadata={"fruit_name": "Apple", "organic": 1},
            ),
        ]
        assert engine._load_document_table(table_name).columns.keys() == [
            "page_content",
            "fruit_name",
            "organic",
        ]


def test_save_doc_without_metadata(engine):
    engine.init_document_table(
        table_name,
        store_metadata=False,
    )
    test_docs = [
        Document(
            page_content="Granny Smith 150 0.99",
            metadata={"fruit_id": 1, "fruit_name": "Apple", "organic": 1},
        ),
    ]
    saver = MySQLDocumentSaver(engine=engine, table_name=table_name)
    loader = MySQLLoader(
        engine=engine,
        table_name=table_name,
    )

    saver.add_documents(test_docs)
    docs = loader.load()

    assert docs == [
        Document(
            page_content="Granny Smith 150 0.99",
            metadata={},
        ),
    ]
    assert engine._load_document_table(table_name).columns.keys() == [
        "page_content",
    ]


def test_delete_doc_with_default_metadata(engine):
    engine.init_document_table(table_name)
    test_docs = [
        Document(
            page_content="Apple Granny Smith 150 0.99 1",
            metadata={"fruit_id": 1},
        ),
        Document(
            page_content="Banana Cavendish 200 0.59 0 1",
            metadata={"fruit_id": 2},
        ),
    ]
    saver = MySQLDocumentSaver(engine=engine, table_name=table_name)
    loader = MySQLLoader(engine=engine, table_name=table_name)

    saver.add_documents(test_docs)
    docs = loader.load()
    assert docs == test_docs

    saver.delete(docs[:1])
    assert len(loader.load()) == 1

    saver.delete(docs)
    assert len(loader.load()) == 0


@pytest.mark.parametrize("store_metadata", [True, False])
def test_delete_doc_with_customized_metadata(engine, store_metadata):
    engine.init_document_table(
        table_name,
        metadata_columns=[
            sqlalchemy.Column(
                "fruit_name",
                sqlalchemy.UnicodeText,
                primary_key=False,
                nullable=True,
            ),
            sqlalchemy.Column(
                "organic",
                sqlalchemy.Boolean,
                primary_key=False,
                nullable=True,
            ),
        ],
        store_metadata=store_metadata,
    )
    test_docs = [
        Document(
            page_content="Granny Smith 150 0.99",
            metadata={"fruit-id": 1, "fruit_name": "Apple", "organic": 1},
        ),
        Document(
            page_content="Cavendish 200 0.59 0",
            metadata={"fruit_id": 2, "fruit_name": "Banana", "organic": 1},
        ),
    ]
    saver = MySQLDocumentSaver(engine=engine, table_name=table_name)
    loader = MySQLLoader(engine=engine, table_name=table_name)

    saver.add_documents(test_docs)
    docs = loader.load()
    assert len(docs) == 2

    saver.delete(docs[:1])
    assert len(loader.load()) == 1

    saver.delete(docs)
    assert len(loader.load()) == 0


def test_delete_doc_with_query(engine):
    engine.init_document_table(
        table_name,
        metadata_columns=[
            sqlalchemy.Column(
                "fruit_name",
                sqlalchemy.UnicodeText,
                primary_key=False,
                nullable=True,
            ),
            sqlalchemy.Column(
                "organic",
                sqlalchemy.Boolean,
                primary_key=False,
                nullable=True,
            ),
        ],
        store_metadata=True,
    )
    test_docs = [
        Document(
            page_content="Granny Smith 150 0.99",
            metadata={"fruit-id": 1, "fruit_name": "Apple", "organic": 1},
        ),
        Document(
            page_content="Cavendish 200 0.59 0",
            metadata={"fruit_id": 2, "fruit_name": "Banana", "organic": 0},
        ),
        Document(
            page_content="Navel 80 1.29 1",
            metadata={"fruit_id": 3, "fruit_name": "Orange", "organic": 1},
        ),
    ]
    saver = MySQLDocumentSaver(engine=engine, table_name=table_name)
    loader = MySQLLoader(engine=engine, table_name=table_name)
    query = f"select * from `{table_name}` where fruit_name='Apple';"
    query_loader = MySQLLoader(engine=engine, query=query)

    saver.add_documents(test_docs)
    docs = query_loader.load()
    assert len(docs) == 1

    saver.delete(docs)
    assert len(loader.load()) == 2


def test_load_and_spilt(engine):
    engine.init_document_table(table_name)
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=10,
        chunk_overlap=2,
        length_function=len,
        is_separator_regex=False,
    )
    test_docs = [
        Document(
            page_content="Apple Granny Smith 150 0.99 1",
            metadata={"fruit_id": 1},
        ),
    ]
    saver = MySQLDocumentSaver(engine=engine, table_name=table_name)
    loader = MySQLLoader(engine=engine, table_name=table_name)

    saver.add_documents(test_docs)
    docs = loader.load_and_split(text_splitter=text_splitter)

    assert len(docs) == 4
