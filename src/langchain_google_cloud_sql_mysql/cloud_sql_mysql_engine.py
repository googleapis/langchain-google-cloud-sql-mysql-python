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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

import google.auth
import google.auth.transport.requests
import requests
import sqlalchemy
from google.cloud.sql.connector import Connector

if TYPE_CHECKING:
    import google.auth.credentials
    import pymysql

logger = logging.getLogger(__name__)


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
    # if credentials are associated with a service account email, return early
    if hasattr(credentials, "_service_account_email"):
        return credentials._service_account_email
    # refresh credentials if they are not valid
    if not credentials.valid:
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
    # call OAuth2 api to get IAM principal email associated with OAuth2 token
    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={credentials.token}"
    response = requests.get(url)
    response_json: Dict = response.json()
    email = response_json.get("email")
    if email is None:
        raise ValueError(
            "Failed to automatically obtain authenticated IAM princpal's "
            "email address using environment's ADC credentials!"
        )
    return email


class CloudSQLMySQLEngine:
    """A class for managing connections to a Cloud SQL for MySQL database."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        instance: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        self._project_id = project_id
        self._region = region
        self._instance = instance
        self._database = database
        self.engine = self._create_connector_engine()

    def close(self) -> None:
        """Utility method for closing the Cloud SQL Python Connector
        background tasks.
        """
        if hasattr(self, "_connector"):
            self._connector.close()

    @classmethod
    def from_instance(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
    ) -> CloudSQLMySQLEngine:
        """Create an instance of CloudSQLMySQLEngine from Cloud SQL instance
        details.

        This method uses the Cloud SQL Python Connector to connect to Cloud SQL
        using automatic IAM database authentication with the Google ADC
        credentials sourced from the environment.

        More details can be found at https://github.com/GoogleCloudPlatform/cloud-sql-python-connector#credentials

        Args:
            project_id (str): Project ID of the Google Cloud Project where
                the Cloud SQL instance is located.
            region (str): Region where the Cloud SQL instance is located.
            instance (str): The name of the Cloud SQL instance.
            database (str): The name of the database to connect to on the
                Cloud SQL instance.

        Returns:
            (CloudSQLMySQLEngine): The engine configured to connect to a
                Cloud SQL instance database.
        """
        return cls(
            project_id=project_id,
            region=region,
            instance=instance,
            database=database,
        )

    def _create_connector_engine(self) -> sqlalchemy.engine.Engine:
        """Create a SQLAlchemy engine using the Cloud SQL Python Connector.

        Defaults to use "pymysql" driver and to connect using automatic IAM
        database authentication with the IAM principal associated with the
        environment's Google Application Default Credentials.

        Returns:
            (sqlalchemy.engine.Engine): Engine configured using the Cloud SQL
                Python Connector.
        """
        # get application default credentials
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/userinfo.email"]
        )
        logger.log(
            logging.ERROR,
            "%s",
            [getattr(credentials, attr) for attr in dir(credentials)],
        )
        iam_database_user = _get_iam_principal_email(credentials)
        logger.log(logging.ERROR, "%s", iam_database_user)
        self._connector = Connector()

        # anonymous function to be used for SQLAlchemy 'creator' argument
        def getconn() -> pymysql.Connection:
            conn = self._connector.connect(
                f"{self._project_id}:{self._region}:{self._instance}",
                "pymysql",
                user=iam_database_user,
                db=self._database,
                enable_iam_auth=True,
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
