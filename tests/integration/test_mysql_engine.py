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

import sqlalchemy

from langchain_google_cloud_sql_mysql import MySQLEngine

project_id = os.environ["PROJECT_ID"]
region = os.environ["REGION"]
instance_id = os.environ["INSTANCE_ID"]
db_name = os.environ["DB_NAME"]
db_user = os.environ["DB_USER"]
db_password = os.environ["DB_PASSWORD"]


def test_mysql_engine_with_basic_auth() -> None:
    """Test MySQLEngine works with basic user/password auth."""
    # override MySQLEngine._connector to allow a new Connector to be initiated
    MySQLEngine._connector = None
    engine = MySQLEngine.from_instance(
        project_id=project_id,
        region=region,
        instance=instance_id,
        database=db_name,
        user=db_user,
        password=db_password,
    )
    # test connection with query
    with engine.connect() as conn:
        res = conn.execute(sqlalchemy.text("SELECT 1")).fetchone()
        conn.commit()
        assert res[0] == 1
    # reset MySQLEngine._connector to allow a new Connector to be initiated
    MySQLEngine._connector = None
