from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    HnswConfigDiff, 
    OptimizersConfigDiff,
    WalConfigDiff,
    QuantizationConfig,
    ScalarQuantization,
    ProductQuantization,
    ScalarType,
    CompressionRatio,
)
import uuid
from Whole_chain.vector_stores import VectorStore

def get_default_hnsw_config() -> HnswConfigDiff:
    """Get default HNSW index configuration."""
    return HnswConfigDiff(
        m=16,  # Number of edges per node in the index graph
        ef_construct=100,  # Size of the dynamic candidate list for constructing the index
        full_scan_threshold=10000,  # Number of points after which to enable full scan
        max_indexing_threads=0,  # Auto-detect number of threads
        on_disk=False,  # Store index in RAM
    )

def get_default_optimizer_config() -> OptimizersConfigDiff:
    """Get default optimizer configuration."""
    return OptimizersConfigDiff(
        deleted_threshold=0.2,  # Minimal fraction of deleted vectors for optimization
        vacuum_min_vector_number=1000,  # Minimal number of vectors for optimization
        default_segment_number=0,  # Auto-detect optimal segment number
        max_segment_size=None,  # Default segment size
        memmap_threshold=None,  # Default memmap threshold
        indexing_threshold=20000,  # Minimal number of vectors for indexing
        flush_interval_sec=5,  # Interval between force flushes
        max_optimization_threads=0,  # Auto-detect number of threads
    )

def get_default_wal_config() -> WalConfigDiff:
    return WalConfigDiff(
        wal_capacity_mb=32,  # Size of WAL segment
        wal_segments_ahead=0,  # Number of WAL segments to create ahead
    )

def get_default_quantization_config() -> QuantizationConfig:
    return QuantizationConfig(
        scalar=ScalarQuantization(
            type=ScalarType.INT8,  # 8-bit quantization
            always_ram=True,  # Keep quantized vectors in RAM
            quantile=0.99,  # Quantile for quantization
        ),
        product=ProductQuantization(
            compression=CompressionRatio.X4,  # 4x compression
            always_ram=True,  # Keep quantized vectors in RAM
        ),
    )

class QdrantWrapper:
    
    def __init__(
        self,
        url: str = None,
        prefer_grpc: bool = True,
        timeout: Optional[float] = None,
    ):
        if url == ":memory:" or url is None:
            self.client = QdrantClient(":memory:", prefer_grpc=prefer_grpc, timeout=timeout)
        else:
            self.client = QdrantClient(url=url, prefer_grpc=prefer_grpc, timeout=timeout)
    
    def create_collection(
        self,
        name: str,
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE,
        hnsw_config: Optional[HnswConfigDiff] = None,
        wal_config: Optional[WalConfigDiff] = None,
        optimizers_config: Optional[OptimizersConfigDiff] = None,
        shard_number: int = 1,
        replication_factor: int = 1,
        write_consistency_factor: int = 1,
        on_disk_payload: bool = False,
        quantization_config: Optional[QuantizationConfig] = None,
        on_disk: bool = False,
        init_from: Optional[str] = None,
    ) -> None:
        try:
            # Use default configurations if not provided
            hnsw_config = hnsw_config or get_default_hnsw_config()
            wal_config = wal_config or get_default_wal_config()
            optimizers_config = optimizers_config or get_default_optimizer_config()
            
            vectors_config = VectorParams(
                size=vector_size, 
                distance=distance,
                on_disk=on_disk
            )
            
            self.client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                hnsw_config=hnsw_config,
                wal_config=wal_config,
                optimizers_config=optimizers_config,
                shard_number=shard_number,
                replication_factor=replication_factor,
                write_consistency_factor=write_consistency_factor,
                on_disk_payload=on_disk_payload,
                quantization_config=quantization_config,
                init_from=init_from,
            )
        except Exception as e:
            raise RuntimeError(f"Error creating collection: {e}")
            
    def upsert(self, collection_name: str, points: List[PointStruct], wait: bool = True) -> None:
        
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=wait
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upsert points: {e}")
        
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        offset: int = 0,
        score_threshold: Optional[float] = None,
        query_filter: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ):
        try:
            return self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                offset=offset,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to perform search: {e}")

class QdrantVectorStore(VectorStore):
    
    def __init__(
        self,
        url: str = None,
        collection_name: str = "default",
        vector_size: int = 1536,
        distance: Distance = Distance.COSINE,
        on_disk: bool = False,
        hnsw_config: Optional[HnswConfigDiff] = None,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        self.client = QdrantWrapper(url)
        self.collection_name = collection_name
        self.client.create_collection(
            name=collection_name,
            vector_size=vector_size,
            distance=distance,
            on_disk=on_disk,
            hnsw_config=hnsw_config,
            quantization_config=quantization_config,
        )

    def add(self, text: str, embedding: List[float]) -> None:
        point_id = str(uuid.uuid4())
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={"text": text}
        )
        self.client.upsert(self.collection_name, [point])

    def query(
        self,
        query_embedding: List[float],
        k: int = 10,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        
        response = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            score_threshold=score_threshold,
            query_filter=filter,
        )
        return [hit.payload["text"] for hit in response] 