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
from typing import List

import sqlalchemy
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from langchain_google_cloud_sql_mysql.mysql_engine import MySQLEngine


class MySQLChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Cloud SQL for MySQL database."""

    def __init__(
        self,
        engine: MySQLEngine,
        session_id: str,
        table_name: str = "message_store",
    ) -> None:
        self.engine = engine
        self.session_id = session_id
        self.table_name = table_name
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        create_table_query = f"""CREATE TABLE IF NOT EXISTS {self.table_name} (
          id INT AUTO_INCREMENT PRIMARY KEY,
          session_id TEXT NOT NULL,
          message JSON NOT NULL
        );"""

        with self.engine.connect() as conn:
            conn.execute(sqlalchemy.text(create_table_query))
            conn.commit()

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Cloud SQL"""
        query = f"SELECT message FROM {self.table_name} WHERE session_id = '{self.session_id}' ORDER BY id;"
        with self.engine.connect() as conn:
            results = conn.execute(sqlalchemy.text(query)).fetchall()
        # load SQLAlchemy row objects into dicts
        items = [json.loads(result[0]) for result in results]
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Cloud SQL"""
        query = f"INSERT INTO {self.table_name} (session_id, message) VALUES (:session_id, :message);"
        with self.engine.connect() as conn:
            conn.execute(
                sqlalchemy.text(query),
                {
                    "session_id": self.session_id,
                    "message": json.dumps(message_to_dict(message)),
                },
            )
            conn.commit()

    def clear(self) -> None:
        """Clear session memory from Cloud SQL"""
        query = f"DELETE FROM {self.table_name} WHERE session_id = :session_id;"
        with self.engine.connect() as connection:
            connection.execute(sqlalchemy.text(query), {"session_id": self.session_id})
            connection.commit()
