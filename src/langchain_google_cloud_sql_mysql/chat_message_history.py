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
from langchain_core.messages import BaseMessage, messages_from_dict

from .engine import MySQLEngine


class MySQLChatMessageHistory(BaseChatMessageHistory):
    """Chat message history stored in a Cloud SQL MySQL database.

    Args:
        engine (MySQLEngine): Connection pool engine for managing
            connections to Cloud SQL for MySQL.
        session_id (str): Arbitrary key that is used to store the messages
            of a single chat session.
        table_name (str): The name of the table to use for storing/retrieving
            the chat message history.
    """

    def __init__(
        self,
        engine: MySQLEngine,
        session_id: str,
        table_name: str,
    ) -> None:
        self.engine = engine
        self.session_id = session_id
        self.table_name = table_name
        self._verify_schema()

    def _verify_schema(self) -> None:
        """Verify table exists with required schema for MySQLChatMessageHistory class.

        Use helper method MySQLEngine.init_chat_history_table(...) to create
        table with valid schema.
        """
        insp = sqlalchemy.inspect(self.engine.engine)
        # check table exists
        if insp.has_table(self.table_name):
            # check that all required columns are present
            required_columns = ["id", "session_id", "data", "type"]
            column_names = [
                c["name"] for c in insp.get_columns(table_name=self.table_name)
            ]
            if not (all(x in column_names for x in required_columns)):
                raise IndexError(
                    f"Table '{self.table_name}' has incorrect schema. Got "
                    f"column names '{column_names}' but required column names "
                    f"'{required_columns}'.\nPlease create table with following schema:"
                    f"\nCREATE TABLE {self.table_name} ("
                    "\n    id INT AUTO_INCREMENT PRIMARY KEY,"
                    "\n    session_id TEXT NOT NULL,"
                    "\n    data JSON NOT NULL,"
                    "\n    type TEXT NOT NULL"
                    "\n);"
                )
        else:
            raise AttributeError(
                f"Table '{self.table_name}' does not exist. Please create "
                "it before initializing MySQLChatMessageHistory. See "
                "MySQLEngine.init_chat_history_table() for a helper method."
            )

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Cloud SQL"""
        query = f"SELECT data, type FROM `{self.table_name}` WHERE session_id = :session_id ORDER BY id;"
        with self.engine.connect() as conn:
            results = conn.execute(
                sqlalchemy.text(query), {"session_id": self.session_id}
            ).fetchall()
        # load SQLAlchemy row objects into dicts
        items = [
            {"data": json.loads(result[0]), "type": result[1]} for result in results
        ]
        messages = messages_from_dict(items)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in Cloud SQL"""
        query = f"INSERT INTO `{self.table_name}` (session_id, data, type) VALUES (:session_id, :data, :type);"
        with self.engine.connect() as conn:
            conn.execute(
                sqlalchemy.text(query),
                {
                    "session_id": self.session_id,
                    "data": json.dumps(message.dict()),
                    "type": message.type,
                },
            )
            conn.commit()

    def clear(self) -> None:
        """Clear session memory from Cloud SQL"""
        query = f"DELETE FROM `{self.table_name}` WHERE session_id = :session_id;"
        with self.engine.connect() as conn:
            conn.execute(sqlalchemy.text(query), {"session_id": self.session_id})
            conn.commit()
