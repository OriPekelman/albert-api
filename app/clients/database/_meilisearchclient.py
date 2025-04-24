from typing import List, Optional, Dict, Any
import meilisearch
from uuid import uuid4
import traceback
from fastapi import HTTPException

from app.schemas.chunks import Chunk
from app.schemas.search import Search, SearchMethod
from app.utils.exceptions import NotImplementedException
from app.utils.logging import logger
from app.utils.settings import settings


class MeilisearchClient:
    def __init__(self, *args, **kwargs):
        self.client = meilisearch.Client(kwargs.get("url", "http://localhost:7700"), kwargs.get("api_key"))

    async def check(self) -> bool:
        try:
            self.client.health()
            return True
        except Exception:
            return False

    async def create_collection(self, collection_id: int, vector_size: int) -> None:
        """Create a new index in Meilisearch with vector search capabilities"""
        index_name = str(collection_id)
        task_info = None # Initialize task_info
        try:
            # Create the index
            logger.debug(f"Attempting to create Meilisearch index: {index_name}")
            task_info = self.client.create_index(index_name, {"primaryKey": "id"})
            logger.info(f"Index creation task submitted for '{index_name}'. Task ID: {getattr(task_info, 'task_uid', 'N/A')}")
            # It might be good practice to wait for index creation before updating settings
            # self.client.wait_for_task(task_info.task_uid) 
            
            # Get embedder config from settings args
            embedder_config = settings.databases.meilisearch.args.get("embedders", {}).get("default-embedder")
            if not embedder_config:
                raise ValueError("Missing 'default-embedder' configuration under databases.meilisearch.args.embedders in config.")
            
            # Get document template, default to basic content if not specified
            doc_template = embedder_config.get("documentTemplate", "{{ doc.content }}")

            # Configure vector search settings
            index_settings = {
                "searchableAttributes": ["content"],
                "filterableAttributes": ["metadata.document_id", "metadata.collection_id"],
                "embedders": {
                    "default-embedder": {
                        "source": embedder_config.get("source", "openAi"),
                        "model": embedder_config.get("model"),
                        "apiKey": embedder_config.get("apiKey"),
                        "dimensions": vector_size, # Keep dimensions here
                        "documentTemplate": doc_template
                    }
                }
            }
            
            # Filter out None values from the embedder config before sending
            embedder_settings = index_settings["embedders"]["default-embedder"]
            index_settings["embedders"]["default-embedder"] = {k: v for k, v in embedder_settings.items() if v is not None}

            # Update the settings for the index
            logger.debug(f"Attempting to update settings for index: {index_name}")
            task_info = self.client.index(index_name).update_settings(index_settings)
            logger.info(f"Settings update task submitted for '{index_name}'. Task ID: {getattr(task_info, 'task_uid', 'N/A')}")
            # Optionally wait for the task to complete
            # self.client.wait_for_task(task.task_uid)
            logger.info(f"Successfully submitted creation/configuration for index '{index_name}'.")
            
        except meilisearch.errors.MeilisearchApiError as e:
            logger.error(f"Meilisearch API error during collection creation/configuration for index '{index_name}': {e}. Last task info: {task_info}")
            # Re-raise as HTTPException for FastAPI to handle cleanly?
            # Depending on desired behavior, might need cleanup (e.g., delete partially created index)
            raise HTTPException(status_code=400, detail=f"Meilisearch error: {e}") 
        except Exception as e:
            logger.error(f"Unexpected error during collection creation/configuration for index '{index_name}': {e}. Last task info: {task_info}")
            logger.error(traceback.format_exc()) # Use logger.error for traceback too
            # Re-raise as HTTPException
            raise HTTPException(status_code=500, detail=f"Internal error configuring search index: {type(e).__name__}")

    async def delete_collection(self, collection_id: int) -> None:
        """Delete an index from Meilisearch"""
        try:
            self.client.delete_index(str(collection_id))
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            raise

    async def get_chunk_count(self, collection_id: int, document_id: int) -> Optional[int]:
        """Get the count of chunks for a document"""
        try:
            index = self.client.index(str(collection_id))
            result = index.search("", {
                "filter": f"metadata.document_id = {document_id}",
                "limit": 0
            })
            return result.get("estimatedTotalHits", 0)
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return None

    async def delete_document(self, collection_id: int, document_id: int) -> None:
        """Delete all chunks associated with a document"""
        try:
            index = self.client.index(str(collection_id))
            index.delete_documents({
                "filter": f"metadata.document_id = {document_id}"
            })
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise

    async def get_chunks(self, collection_id: int, document_id: int, offset: int = 0, limit: int = 10, chunk_id: Optional[int] = None) -> List[Chunk]:
        """Get chunks for a document with pagination"""
        try:
            index = self.client.index(str(collection_id))
            filter_expr = f"metadata.document_id = {document_id}"
            if chunk_id:
                filter_expr += f" AND metadata.id = {chunk_id}"
                
            result = index.search("", {
                "filter": filter_expr,
                "offset": offset,
                "limit": limit,
                "sort": ["id:asc"]
            })
            
            chunks = []
            for hit in result.get("hits", []):
                chunks.append(Chunk(
                    id=hit["id"],
                    content=hit["content"],
                    metadata=hit["metadata"]
                ))
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []

    async def upsert(self, collection_id: int, chunks: List[Chunk]) -> None:
        """Add or update chunks in Meilisearch. Meilisearch will automatically generate embeddings."""
        try:
            index = self.client.index(str(collection_id))
            documents = []
            
            for chunk in chunks:
                doc = {
                    "id": str(uuid4()),
                    "content": chunk.content,
                    "metadata": chunk.metadata
                }
                documents.append(doc)
                
            index.add_documents(documents)
            
        except Exception as e:
            logger.error(f"Error upserting chunks: {e}")
            raise

    async def search(
        self,
        method: SearchMethod,
        collection_ids: List[int],
        query_prompt: str,
        query_vector: list[float] = None,  # Kept for compatibility but not used
        k: int = 4,
        score_threshold: float = 0.0,
    ) -> List[Search]:
        """Search across collections using the specified method"""
        searches = []
        
        for collection_id in collection_ids:
            index = self.client.index(str(collection_id))
            search_params = {
                "limit": k,
                "rankingScoreThreshold": score_threshold if score_threshold and score_threshold > 0 else None
            }
            
            if method == SearchMethod.SEMANTIC:
                # Pure semantic search using Meilisearch's built-in capabilities
                search_params.update({
                    "hybrid": {
                        "semanticRatio": 1.0,  # Pure semantic search
                        "embedder": "default-embedder"  # Using the default embedder configured in settings
                    }
                })
                
            elif method == SearchMethod.LEXICAL:
                # Pure keyword search
                search_params.update({
                    "hybrid": {
                        "semanticRatio": 0.0,  # Pure lexical search
                        "embedder": "default-embedder"
                    }
                })
                
            else:  # HYBRID
                # Balanced hybrid search
                search_params.update({
                    "hybrid": {
                        "semanticRatio": 0.5,  # Equal mix of semantic and lexical search
                        "embedder": "default-embedder"
                    }
                })
            
            result = index.search(query_prompt, search_params)
            
            # Process results
            for hit in result.get("hits", []):
                try:
                    searches.append(Search(
                        method=method.value,
                        score=hit.get("_score", 0.0),
                        chunk=Chunk(
                            id=hit["id"],
                            content=hit["content"],
                            metadata=hit["metadata"]
                        )
                    ))
                except KeyError as e:
                    logger.error(f"KeyError processing search hit: {e}. Hit data: {hit}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error processing search hit: {e}. Hit data: {hit}")
                    logger.debug(traceback.format_exc())
                    continue
        
        # Sort by score and limit to k results
        searches = sorted(searches, key=lambda x: x.score, reverse=True)[:k]
        return searches 