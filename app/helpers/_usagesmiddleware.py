import asyncio
from datetime import datetime
import logging
from typing import AsyncGenerator, Callable

from fastapi import Request, Response
from fastapi.routing import APIRoute
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

from app.sql.models import Usage
from app.sql.session import get_db
from app.utils.usage_decorator import NoUserIdException, StreamingRequestException, extract_usage_from_request, extract_usage_from_response

logger = logging.getLogger(__name__)


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
        route = self.get_route(request)

        if route and getattr(route.endpoint, "is_log_usage_decorated", False):
            logger.debug("Endpoint is decorated with log_usage, skipping middleware logging.")
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
        usage = Usage(datetime=start_time, endpoint="N/A")
        try:
            await extract_usage_from_request(usage, request)
        except NoUserIdException:
            logger.info("No user ID found in request, skipping usage logging.")
            return await call_next(request)
        except StreamingRequestException:
            logger.debug("Streaming request, should be handled by decorator.")
            return await call_next(request)

        response = await call_next(request)
        asyncio.create_task(extract_usage_from_response(response, start_time, usage))
        return response

    def get_route(self, request):
        route = None
        for r in request.app.router.routes:
            match, _ = r.matches(request.scope)
            if match == Match.FULL and isinstance(r, APIRoute):
                route = r
                break
        return route
