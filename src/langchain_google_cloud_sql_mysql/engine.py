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

# TODO: Remove below import when minimum supported Python version is 3.10
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import google.auth
import google.auth.transport.requests
import requests
import sqlalchemy
from google.cloud.sql.connector import Connector, RefreshStrategy

from .version import __version__

if TYPE_CHECKING:
    import google.auth.credentials
    import pymysql

USER_AGENT = "langchain-google-cloud-sql-mysql-python/" + __version__

from dataclasses import dataclass


@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool = True

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise ValueError("Column name must be type string")
        if not isinstance(self.data_type, str):
            raise ValueError("Column data_type must be type string")


def _get_iam_principal_email(
    credentials: google.auth.credentials.Credentials,
) -> str:
    """Get email address associated with current authenticated IAM principal.

    Email will be used for automatic IAM database authentication to Cloud SQL.

    Args:
        credentials (google.auth.credentials.Credentials):
            The credentials object to use in finding the associated IAM
            principal email address.

    Returns:
        email (str):
            The email address associated with the current authenticated IAM
            principal.
    """
    # refresh credentials if they are not valid
    if not credentials.valid:
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
    # if credentials are associated with a service account email, return early
    if hasattr(credentials, "_service_account_email"):
        return credentials._service_account_email
    # call OAuth2 api to get IAM principal email associated with OAuth2 token
    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={credentials.token}"
    response = requests.get(url)
    response.raise_for_status()
    response_json: Dict = response.json()
    email = response_json.get("email")
    if email is None:
        raise ValueError(
            "Failed to automatically obtain authenticated IAM princpal's "
            "email address using environment's ADC credentials!"
        )
    return email


class MySQLEngine:
    """A class for managing connections to a Cloud SQL for MySQL database."""

    _connector: Optional[Connector] = None

    def __init__(
        self,
        engine: sqlalchemy.engine.Engine,
    ) -> None:
        self.engine = engine

    @classmethod
    def from_instance(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> MySQLEngine:
        """Create an instance of MySQLEngine from Cloud SQL instance
        details.

        This method uses the Cloud SQL Python Connector to connect to Cloud SQL
        using automatic IAM database authentication with the Google ADC
        credentials sourced from the environment by default. If user and
        password arguments are given, basic database authentication will be
        used for database login.

        More details can be found at https://github.com/GoogleCloudPlatform/cloud-sql-python-connector#credentials

        Args:
            project_id (str): Project ID of the Google Cloud Project where
                the Cloud SQL instance is located.
            region (str): Region where the Cloud SQL instance is located.
            instance (str): The name of the Cloud SQL instance.
            database (str): The name of the database to connect to on the
                Cloud SQL instance.
            user (str, optional): Database user to use for basic database
                authentication and login. Defaults to None.
            password (str, optional): Database password for 'user' to use for
                basic database authentication and login. Defaults to None.

        Returns:
            (MySQLEngine): The engine configured to connect to a
                Cloud SQL instance database.
        """
        # error if only one of user or password is set, must be both or neither
        if bool(user) ^ bool(password):
            raise ValueError(
                "Only one of 'user' or 'password' were specified. Either "
                "both should be specified to use basic user/password "
                "authentication or neither for IAM DB authentication."
            )
        engine = cls._create_connector_engine(
            instance_connection_name=f"{project_id}:{region}:{instance}",
            database=database,
            user=user,
            password=password,
        )
        return cls(engine=engine)

    @classmethod
    def _create_connector_engine(
        cls,
        instance_connection_name: str,
        database: str,
        user: Optional[str],
        password: Optional[str],
    ) -> sqlalchemy.engine.Engine:
        """Create a SQLAlchemy engine using the Cloud SQL Python Connector.

        Defaults to use "pymysql" driver and to connect using automatic IAM
        database authentication with the IAM principal associated with the
        environment's Google Application Default Credentials. If user and
        password arguments are given, basic database authentication will be
        used for database login.

        Args:
            instance_connection_name (str): The instance connection
                name of the Cloud SQL instance to establish a connection to.
                (ex. "project-id:instance-region:instance-name")
            database (str): The name of the database to connect to on the
                Cloud SQL instance.
            user (str, optional): Database user to use for basic database
                authentication and login. Defaults to None.
            password (str, optional): Database password for 'user' to use for
                basic database authentication and login. Defaults to None.

        Returns:
            (sqlalchemy.engine.Engine): Engine configured using the Cloud SQL
                Python Connector.
        """
        # if user and password are given, use basic auth
        if user and password:
            enable_iam_auth = False
            db_user = user
        # otherwise use automatic IAM database authentication
        else:
            # get application default credentials
            credentials, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/userinfo.email"]
            )
            db_user = _get_iam_principal_email(credentials)
            enable_iam_auth = True

        if cls._connector is None:
            cls._connector = Connector(
                user_agent=USER_AGENT, refresh_strategy=RefreshStrategy.LAZY
            )

        # anonymous function to be used for SQLAlchemy 'creator' argument
        def getconn() -> pymysql.Connection:
            conn = cls._connector.connect(  # type: ignore
                instance_connection_name,
                "pymysql",
                user=db_user,
                password=password,
                db=database,
                enable_iam_auth=enable_iam_auth,
            )
            return conn

        return sqlalchemy.create_engine(
            "mysql+pymysql://",
            creator=getconn,
        )

    def connect(self) -> sqlalchemy.engine.Connection:
        """Create a connection from SQLAlchemy connection pool.

        Returns:
            (sqlalchemy.engine.Connection): a single DBAPI connection checked
                out from the connection pool.
        """
        return self.engine.connect()

    def _execute(self, query: str, params: Optional[dict] = None) -> None:
        """Executes a SQL query within a transaction."""
        with self.engine.connect() as conn:
            conn.execute(sqlalchemy.text(query), params)
            conn.commit()

    def _execute_outside_tx(self, query: str, params: Optional[dict] = None) -> None:
        """Executes a SQL query with autocommit (outside of transaction)."""
        with self.engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(sqlalchemy.text(query), params)

    def _fetch(self, query: str, params: Optional[dict] = None):
        """Fetch results from a SQL query."""
        with self.engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(query), params)
            result_map = result.mappings()
            result_fetch = result_map.fetchall()
            return result_fetch

    def _fetch_rows(self, query: str, params: Optional[dict] = None):
        """Fetch results from a SQL query as rows."""
        with self.engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(query), params)
            result_fetch = result.fetchall()  # Directly fetch rows
            return result_fetch

    def init_chat_history_table(self, table_name: str) -> None:
        """Create table with schema required for MySQLChatMessageHistory class.

        Required schema is as follows:

        ::

            CREATE TABLE {table_name} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id TEXT NOT NULL,
                data JSON NOT NULL,
                type TEXT NOT NULL
            )

        Args:
            table_name (str): The name of the table to create.

        """

        create_table_query = f"""CREATE TABLE IF NOT EXISTS `{table_name}` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id TEXT NOT NULL,
            data JSON NOT NULL,
            type TEXT NOT NULL
        );"""

        with self.engine.connect() as conn:
            conn.execute(sqlalchemy.text(create_table_query))
            conn.commit()

    def init_document_table(
        self,
        table_name: str,
        metadata_columns: List[sqlalchemy.Column] = [],
        content_column: str = "page_content",
        metadata_json_column: Optional[str] = "langchain_metadata",
        overwrite_existing: bool = False,
    ) -> None:
        """
        Create a table for saving of langchain documents.

        Args:
            table_name (str): The MySQL database table name.
            metadata_columns (List[sqlalchemy.Column]): A list of SQLAlchemy Columns
                to create for custom metadata. Optional.
            content_column (str): The column to store document content.
                Deafult: `page_content`.
            metadata_json_column (Optional[str]): The column to store extra metadata in JSON format.
                Default: `langchain_metadata`. Optional.
            overwrite_existing (bool): Whether to drop existing table. Default: False.
        """
        if overwrite_existing:
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS `{table_name}`;"))

        columns = [
            sqlalchemy.Column(
                content_column,
                sqlalchemy.UnicodeText,
                primary_key=False,
                nullable=False,
            )
        ]
        columns += metadata_columns
        if metadata_json_column:
            columns.append(
                sqlalchemy.Column(
                    metadata_json_column,
                    sqlalchemy.JSON,
                    primary_key=False,
                    nullable=True,
                )
            )
        sqlalchemy.Table(table_name, sqlalchemy.MetaData(), *columns).create(
            self.engine
        )

    def _load_document_table(self, table_name: str) -> sqlalchemy.Table:
        """
        Load table schema from existing table in MySQL database.

        Args:
            table_name (str): The MySQL database table name.

        Returns:
            (sqlalchemy.Table): The loaded table.
        """
        metadata = sqlalchemy.MetaData()
        sqlalchemy.MetaData.reflect(metadata, bind=self.engine, only=[table_name])
        return metadata.tables[table_name]

    def init_vectorstore_table(
        self,
        table_name: str,
        vector_size: int,
        content_column: str = "content",
        embedding_column: str = "embedding",
        metadata_columns: List[Column] = [],
        metadata_json_column: str = "langchain_metadata",
        id_column: str = "langchain_id",
        overwrite_existing: bool = False,
        store_metadata: bool = True,
    ) -> None:
        """
        Create a table for saving of vectors to be used with MySQLVectorStore.

        Args:
            table_name (str): The MySQL database table name.
            vector_size (int): Vector size for the embedding model to be used.
            content_column (str): Name of the column to store document content.
                Default: `page_content`.
            embedding_column (str) : Name of the column to store vector embeddings.
                Default: `embedding`.
            metadata_columns (List[Column]): A list of Columns to create for custom
                metadata. Default: []. Optional.
            metadata_json_column (str): The column to store extra metadata in JSON format.
                Default: `langchain_metadata`. Optional.
            id_column (str):  Name of the column to store ids.
                Default: `langchain_id`. Optional,
            overwrite_existing (bool): Whether to drop existing table. Default: False.
            store_metadata (bool): Whether to store metadata in the table.
                Default: True.
        """
        query = f"""CREATE TABLE `{table_name}`(
            `{id_column}` CHAR(36) PRIMARY KEY,
            `{content_column}` TEXT NOT NULL,
            `{embedding_column}` vector({vector_size}) USING VARBINARY NOT NULL"""
        for column in metadata_columns:
            nullable = "NOT NULL" if not column.nullable else ""
            query += f",\n`{column.name}` {column.data_type} {nullable}"
        if store_metadata:
            query += f""",\n`{metadata_json_column}` JSON"""
        query += "\n);"

        with self.engine.connect() as conn:
            if overwrite_existing:
                conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS `{table_name}`"))
            conn.execute(sqlalchemy.text(query))
