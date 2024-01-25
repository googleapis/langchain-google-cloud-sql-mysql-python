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
from typing import Generator

import pymysql
import pytest
import sqlalchemy
from google.cloud.sql.connector import Connector
from langchain_core.documents import Document

from langchain_google_cloud_sql_mysql import MySQLDocumentLoader, MySQLEngine

project_id = os.environ.get("PROJECT_ID", None)
region = os.environ.get("REGION")
instance_id = os.environ.get("INSTANCE_ID")
table_name = os.environ.get("TABLE_NAME")
db_name = os.environ.get("DB_NAME")

test_docs = [
    Document(
        page_content="fruit_name: Apple\nvariety: Granny Smith\nquantity_in_stock: 150\nprice_per_unit: 0.99\norganic: 1",
        metadata={"fruit_id": 1},
    ),
    Document(
        page_content="fruit_name: Banana\nvariety: Cavendish\nquantity_in_stock: 200\nprice_per_unit: 0.59\norganic: 0",
        metadata={"fruit_id": 2},
    ),
    Document(
        page_content="fruit_name: Orange\nvariety: Navel\nquantity_in_stock: 80\nprice_per_unit: 1.29\norganic: 1",
        metadata={"fruit_id": 3},
    ),
    Document(
        page_content="fruit_name: Strawberry\nvariety: Camarosa\nquantity_in_stock: 35\nprice_per_unit: 2.49\norganic: 1",
        metadata={"fruit_id": 4},
    ),
    Document(
        page_content="fruit_name: Grape\nvariety: Thompson Seedless\nquantity_in_stock: 120\nprice_per_unit: 1.99\norganic: 0",
        metadata={"fruit_id": 5},
    ),
]


def init_connection_engine() -> sqlalchemy.engine.Engine:
    engine = MySQLEngine.from_instance(
        project_id=project_id, region=region, instance=instance_id, database=db_name
    )
    return engine


@pytest.fixture(name="mysql")
def setup() -> Generator:
    engine = init_connection_engine()

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
        conn.execute(
            sqlalchemy.text(
                f"""
                INSERT INTO `{table_name}` (fruit_name, variety, quantity_in_stock, price_per_unit, organic)
                VALUES
                    ('Apple', 'Granny Smith', 150, 0.99, 1),
                    ('Banana', 'Cavendish', 200, 0.59, 0),
                    ('Orange', 'Navel', 80, 1.29, 1),
                    ('Strawberry', 'Camarosa', 35, 2.49, 1),
                    ('Grape', 'Thompson Seedless', 120, 1.99, 0);
                """
            )
        )
        conn.commit()

    yield engine

    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS `{table_name}`"))
        conn.commit()


def test_load_from_query(mysql):
    query = f"SELECT * FROM `{table_name}`;"

    loader = MySQLDocumentLoader(
        engine=mysql,
        query=query,
        page_content_columns=[
            "fruit_name",
            "variety",
            "quantity_in_stock",
            "price_per_unit",
            "organic",
        ],
        metadata_columns=["fruit_id"],
    )

    documents = loader.load()
    assert documents == test_docs
