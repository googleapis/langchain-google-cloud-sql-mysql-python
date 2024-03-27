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
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SearchType(Enum):
    """Enum for search types."""

    KNN = "KNN"
    ANN = "ANN"


@dataclass
class QueryOptions:
    num_partitions: Optional[int] = None
    num_neighbors: Optional[int] = None
    search_type: SearchType = SearchType.KNN

    def to_string(self) -> str:
        """
        Custom string representation of the QueryOptions instance.
        """
        return str(
            {
                "num_partitions": self.num_partitions,
                "num_neighbors": self.num_neighbors,
                "search_type": self.search_type.name,
            }
        )


DEFAULT_QUERY_OPTIONS = QueryOptions()


from enum import Enum


class IndexType(Enum):
    """Enum for index types."""

    BRUTE_FORCE_SCAN = "BRUTE_FORCE"
    TREE_AH = "TREE_AH"
    TREE_SQ = "TREE_SQ"


class DistanceMeasure(Enum):
    """Enum for distance measures."""

    COSINE = "cosine"
    SQUARED_L2 = "squared_l2"
    DOT_PRODUCT = "dot_product"


class VectorIndex:
    """VectorIndex class for index types."""

    def __init__(
        self,
        name: str = None,
        index_type: IndexType = None,
        distance_measure: DistanceMeasure = None,
        num_partitions: int = None,
        num_neighbors: int = None,
    ):
        self.name = name
        self.index_type = index_type
        self.distance_measure = distance_measure
        self.num_partitions = num_partitions
        self.num_neighbors = num_neighbors
