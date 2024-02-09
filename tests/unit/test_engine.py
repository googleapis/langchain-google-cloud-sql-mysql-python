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

import pytest

from langchain_google_cloud_sql_mysql import MySQLEngine


def test_mysql_engine_with_invalid_arg_pattern() -> None:
    """Test MySQLEngine errors when only one of user or password is given.

    Both user and password must be specified (basic authentication)
    or neither (IAM authentication).
    """
    expected_error_msg = "Only one of 'user' or 'password' were specified. Either both should be specified to use basic user/password authentication or neither for IAM DB authentication."
    # test password not set
    with pytest.raises(ValueError) as exc_info:
        MySQLEngine.from_instance(
            project_id="my-project",
            region="my-region",
            instance="my-instance",
            database="my-db",
            user="my-user",
        )
        # assert custom error is present
        assert exc_info.value.args[0] == expected_error_msg

    # test user not set
    with pytest.raises(ValueError) as exc_info:
        MySQLEngine.from_instance(
            project_id="my-project",
            region="my-region",
            instance="my-instance",
            database="my-db",
            password="my-pass",
        )
        # assert custom error is present
        assert exc_info.value.args[0] == expected_error_msg
