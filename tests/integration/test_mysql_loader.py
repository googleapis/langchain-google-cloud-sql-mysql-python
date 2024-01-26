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

import pytest
import sqlalchemy
from langchain_core.documents import Document

from langchain_google_cloud_sql_mysql import MySQLEngine, MySQLLoader

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

    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS `{table_name}`"))
        conn.commit()


def test_load_from_query(engine):
    with engine.connect() as conn:
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
        engine=engine,
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


def test_load_from_query_default(engine):
    with engine.connect() as conn:
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
        engine=engine,
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
