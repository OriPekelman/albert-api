from datetime import datetime
import json
import traceback
from typing import Callable, Optional, AsyncGenerator

from fastapi import Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware

from app.sql.models import Usage
from app.sql.session import get_db
from app.utils import variables
from app.utils.logging import logger


class UsagesMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, db_func: Callable[[], AsyncGenerator[AsyncSession, None]] = get_db):
        super().__init__(app)
        self.db_func = db_func
        # Get all model endpoints from variables.py
        self.MODELS_ENDPOINTS = [getattr(variables, var_name) for var_name in dir(variables) if var_name.startswith("ENDPOINT__")]

    async def _extract_model_from_multipart(self, body: bytes, content_type: str) -> Optional[str]:
        try:
            # Find the model field in the multipart form data
            parts = body.split(b"\r\n")
            for i, part in enumerate(parts):
                if b'Content-Disposition: form-data; name="model"' in part and i + 2 < len(parts):
                    # The value is 2 lines after the Content-Disposition header
                    return parts[i + 2].decode("utf-8")
            return None
        except Exception as e:
            logger.warning(f"Error extracting model from multipart data: {str(e)}")
            return None

    async def _extract_model_from_json(self, body: bytes) -> Optional[str]:
        try:
            logger.debug(f"Attempting to parse JSON body: {body.decode('utf-8')}")
            json_body = json.loads(body.decode("utf-8"))
            return json_body.get("model")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse JSON request body: {str(e)}\nBody content: {body}")
            return None

    async def _handle_streaming_response(self, response: Response) -> tuple[dict, Response]:
        usage_data = {}
        if hasattr(response, "body_iterator"):
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                try:
                    # Handle SSE format by stripping 'data: ' prefix
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode('utf-8')
                    if not chunk.strip():  # Skip empty chunks
                        continue
                    if chunk.startswith('data: '):
                        chunk = chunk[6:]  # Remove 'data: ' prefix
                    if chunk.strip() == '[DONE]':
                        continue
                    try:
                        # Only try to parse if it looks like JSON
                        if chunk.strip().startswith('{'):
                            chunk_data = json.loads(chunk)
                            if "usage" in chunk_data:
                                usage_data = chunk_data["usage"]
                    except json.JSONDecodeError:
                        logger.debug(f"Non-JSON chunk received: {chunk}")
                        continue
                except (UnicodeDecodeError, AttributeError) as e:
                    logger.debug(f"Error processing chunk: {str(e)}")
                    continue

            async def new_body_iterator():
                for chunk in chunks:
                    yield chunk

            response.body_iterator = new_body_iterator()
        return usage_data, response

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip if middleware is disabled via environment variable
        endpoint = request.url.path
        if not any(endpoint.endswith(model_endpoint) for model_endpoint in self.MODELS_ENDPOINTS):
            return await call_next(request)

        method = request.method
        content_type = request.headers.get("Content-Type", "")
        logger.debug(f"Request endpoint: {endpoint}")
        logger.debug(f"Request method: {method}")
        logger.debug(f"Content-Type: {content_type}")

        # Extract model from request
        model = None
        try:
            # Try to get the body
            body = await request.body()
            logger.debug(f"Raw request body: {body}")
            body_content = body.decode("utf-8") if body else ""
            logger.debug(f"Decoded request body: {body_content}")

            if content_type.startswith("multipart/form-data"):
                model = await self._extract_model_from_multipart(body, content_type)
            else:
                try:
                    # Try to parse as JSON first
                    json_body = json.loads(body_content)
                    logger.debug(f"Parsed JSON body: {json_body}")
                    model = json_body.get("model")
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    # If JSON parsing fails, try to extract model from query params
                    logger.warning(f"Failed to parse JSON request body: {str(e)}\nBody content: {body_content}")
                    model = request.query_params.get("model")

            # Preserve original request body
            original_receive = request._receive

            async def receive():
                original = await original_receive()
                return {**original, "body": body}

            request._receive = receive
        except Exception as e:
            logger.debug(f"Error handling request body: {str(e)}")
            return await call_next(request)

        start_time = datetime.now()
        # Get response
        response = await call_next(request)
        duration = int((datetime.now() - start_time).total_seconds() * 1000)
        if not model:
            return response

        if not hasattr(request.app.state, "user") or request.app.state.user.id == 0:  # master key
            return response

        try:
            usage_data, response = await self._handle_streaming_response(response)
            # Log usage
            async for session in self.db_func():
                log = Usage(
                    datetime=start_time,
                    duration=duration,
                    user_id=request.app.state.user.id,
                    token_id=request.app.state.token_id,
                    endpoint=endpoint,
                    model=model,
                    prompt_tokens=usage_data.get("prompt_tokens"),
                    completion_tokens=usage_data.get("completion_tokens"),
                    total_tokens=usage_data.get("total_tokens"),
                    status=response.status_code,
                    method=method,
                )
                session.add(log)
                await session.commit()
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.error(f"Failed to log usage: {str(e)}")
            await session.rollback()
        finally:
            await session.close()

        return response
