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
    """Defines the types of search algorithms that can be used.

    Attributes:
        KNN: K-Nearest Neighbors search.
        ANN: Approximate Nearest Neighbors search.
    """

    KNN = "KNN"
    ANN = "ANN"


class DistanceMeasure(Enum):
    """Enumerates the types of distance measures that can be used in searches.

    Attributes:
        COSINE: Cosine similarity measure.
        L2_SQUARED: Squared L2 norm (Euclidean) distance.
        DOT_PRODUCT: Dot product similarity.
    """

    COSINE = "cosine"
    L2_SQUARED = "l2_squared"
    DOT_PRODUCT = "dot_product"


@dataclass
class QueryOptions:
    """Holds configuration options for executing a search query.

    Attributes:
        num_partitions (Optional[int]): The number of partitions to divide the search space into. None means default partitioning.
        num_neighbors (int): The number of nearest neighbors to retrieve. Default to 10.
        search_type (SearchType): The type of search algorithm to use. Defaults to KNN.
    """

    num_partitions: Optional[int] = None
    num_neighbors: int = 10
    distance_measure: DistanceMeasure = DistanceMeasure.L2_SQUARED
    search_type: SearchType = SearchType.KNN


DEFAULT_QUERY_OPTIONS = QueryOptions()


class IndexType(Enum):
    """Defines the types of indexes that can be used for vector storage.

    Attributes:
        BRUTE_FORCE_SCAN: A simple brute force scan approach.
        TREE_AH: A tree-based index, specifically Annoy (Approximate Nearest Neighbors Oh Yeah).
        TREE_SQ: A tree-based index, specifically ScaNN (Scalable Nearest Neighbors).
    """

    BRUTE_FORCE_SCAN = "BRUTE_FORCE"
    TREE_AH = "TREE_AH"
    TREE_SQ = "TREE_SQ"


class VectorIndex:
    """Represents a vector index for storing and querying vectors.

    Attributes:
        name (Optional[str]): The name of the index.
        index_type (Optional[IndexType]): The type of index.
        distance_measure (Optional[DistanceMeasure]): The distance measure to use for the index.
        num_partitions (Optional[int]): The number of partitions for the index. None for default.
        num_neighbors (Optional[int]): The default number of neighbors to return for queries.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        index_type: Optional[IndexType] = None,
        distance_measure: Optional[DistanceMeasure] = None,
        num_partitions: Optional[int] = None,
        num_neighbors: Optional[int] = None,
    ):
        """Initializes a new instance of the VectorIndex class."""
        self.name = name
        self.index_type = index_type
        self.distance_measure = distance_measure
        self.num_partitions = num_partitions
        self.num_neighbors = num_neighbors
