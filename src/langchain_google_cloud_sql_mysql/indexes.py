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

@dataclass
class QueryOptions:
    """Holds configuration options for executing a search query.

    Attributes:
        num_partitions (Optional[int]): The number of partitions to divide the search space into. None means default partitioning.
        num_neighbors (Optional[int]): The number of nearest neighbors to retrieve. None means use the default.
        search_type (SearchType): The type of search algorithm to use. Defaults to KNN.
    """

    num_partitions: Optional[int] = None
    num_neighbors: Optional[int] = None
    search_type: SearchType = SearchType.KNN

    def to_string(self) -> str:
        """Generates a custom string representation of the QueryOptions instance."""
        return str(
            {
                "num_partitions": self.num_partitions,
                "num_neighbors": self.num_neighbors,
                "search_type": self.search_type.name,
            }
        )

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

class DistanceMeasure(Enum):
    """Enumerates the types of distance measures that can be used in searches.

    Attributes:
        COSINE: Cosine similarity measure.
        SQUARED_L2: Squared L2 norm (Euclidean) distance.
        DOT_PRODUCT: Dot product similarity.
    """
    COSINE = "cosine"
    SQUARED_L2 = "squared_l2"
    DOT_PRODUCT = "dot_product"

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
