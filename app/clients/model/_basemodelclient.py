from abc import ABC
import ast
import importlib
from json import dumps, loads
from typing import Literal, Optional, Type
from urllib.parse import urljoin

from fastapi import HTTPException
import httpx

from app.utils.variables import (
    ENDPOINT__AUDIO_TRANSCRIPTIONS,
    ENDPOINT__CHAT_COMPLETIONS,
    ENDPOINT__COMPLETIONS,
    ENDPOINT__EMBEDDINGS,
    ENDPOINT__MODELS,
    ENDPOINT__OCR,
    ENDPOINT__RERANK,
)

from app.schemas.core.settings import ModelClientType


class BaseModelClient(ABC):
    ENDPOINT_TABLE = {
        ENDPOINT__AUDIO_TRANSCRIPTIONS: None,
        ENDPOINT__CHAT_COMPLETIONS: None,
        ENDPOINT__COMPLETIONS: None,
        ENDPOINT__EMBEDDINGS: None,
        ENDPOINT__MODELS: None,
        ENDPOINT__OCR: None,
        ENDPOINT__RERANK: None,
    }

    @staticmethod
    def import_module(type: Literal[ModelClientType.OPENAI, ModelClientType.VLLM, ModelClientType.TEI]) -> "Type[BaseModelClient]":
        """
        Static method to import a subclass of BaseModelClient.

        Args:
            type(str): The type of model client to import.

        Returns:
            Type[BaseModelClient]: The subclass of BaseModelClient.
        """
        module = importlib.import_module(f"app.clients.model._{type.value}modelclient")
        return getattr(module, f"{type.capitalize()}ModelClient")

    def _format_request(self, endpoint: str, json: Optional[dict] = None, files: Optional[dict] = None, data: Optional[dict] = None) -> dict:
        """
        Format a request to a client model. This method can be overridden by a subclass to add additional headers or parameters. This method format the requested endpoint thanks the ENDPOINT_TABLE attribute.

        Args:
            json(dict): The JSON body to use for the request.
            files(dict): The files to use for the request.
            data(dict): The data to use for the request.

        Returns:
            tuple: The formatted request composed of the url, headers, json, files and data.
        """
        # self.endpoint is set by the ModelRouter
        url = urljoin(base=self.api_url, url=self.ENDPOINT_TABLE[self.endpoint])
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if json and "model" in json:
            json["model"] = self.model

        return url, headers, json, files, data

    def _format_response(self, response: httpx.Response) -> httpx.Response:
        """
        Format a response from a client model. This method can be overridden by a subclass to add additional headers or parameters.

        Args:
            response(httpx.Response): The response from the API.

        Returns:
            httpx.Response: The formatted response.
        """
        return response

    async def forward_request(
        self,
        endpoint: str,
        method: str,
        json: Optional[dict] = None,
        files: Optional[dict] = None,
        data: Optional[dict] = None,
        additional_data_value: Optional[list] = None,
        additional_data_key: Optional[str] = None,
    ) -> httpx.Response:
        """
        Forward a request to a client model and add additional data to the response if provided.

        Args:
            method(str): The method to use for the request.
            json(dict): The JSON body to use for the request.
            files(dict): The files to use for the request.
            data(dict): The data to use for the request.
            additional_data_value(list): The value to add to the response.
            additional_data_key(str): The key to add the value to.

        Returns:
            httpx.Response: The response from the API.
        """

        url, headers, json, files, data = self._format_request(endpoint=endpoint, json=json, files=files, data=data)

        async with httpx.AsyncClient(timeout=self.timeout) as async_client:
            try:
                response = await async_client.request(method=method, url=url, headers=headers, json=json, files=files, data=data)
            except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
                raise HTTPException(status_code=504, detail="Request timed out, model is too busy.")
            except Exception as e:
                raise HTTPException(status_code=500, detail=type(e).__name__)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                # Try to parse error as JSON, default to raw text if fails
                try:
                    message = loads(response.text)
                    if "message" in message:
                        try:
                            message_detail = ast.literal_eval(message["message"])
                        except Exception:
                            message_detail = message["message"]
                    elif "detail" in message:
                         message_detail = message["detail"]
                    else:
                        message_detail = response.text
                except Exception:
                    message_detail = response.text
                raise HTTPException(status_code=response.status_code, detail=message_detail)

        # Check content type before assuming JSON
        content_type = response.headers.get("content-type", "").lower()
        is_json_response = "application/json" in content_type

        # Add additional data only if it's a JSON response
        if is_json_response and additional_data_value and additional_data_key:
            try:
                data = response.json()
                data[additional_data_key] = additional_data_value
                # Re-create response with updated JSON content
                response = httpx.Response(status_code=response.status_code, headers=response.headers, content=dumps(data).encode('utf-8'))
            except Exception as e:
                 logger.error(f"Error adding additional data to JSON response: {e}")
                 # Proceed with original response if modification fails

        response = self._format_response(response=response) # Subclass formatting hook

        return response

    async def forward_stream(
        self,
        endpoint: str,
        method: str,
        json: Optional[dict] = None,
        files: Optional[dict] = None,
        data: Optional[dict] = None,
        additional_data_value: Optional[list] = None,
        additional_data_key: Optional[str] = None,
    ):
        """
        Forward a stream request to a client model and add additional data to the response if provided.

        Args:
            method(str): The method to use for the request.
            json(dict): The JSON body to use for the request.
            files(dict): The files to use for the request.
            data(dict): The data to use for the request.
            additional_data_value(list): The value to add to the response (only on the first chunk).
            additional_data_key(str): The key to add the value to (only on the first chunk).
        """

        url, headers, json, files, data = self._format_request(endpoint=endpoint, json=json, files=files, data=data)

        async with httpx.AsyncClient(timeout=self.timeout) as async_client:
            try:
                async with async_client.stream(method=method, url=url, headers=headers, json=json, files=files, data=data) as response:
                    # Check status code immediately after stream starts
                    if response.status_code // 100 != 2:
                        # Attempt to read the error body
                        error_body_bytes = await response.aread()
                        try:
                            decoded_chunk = error_body_bytes.decode(encoding="utf-8")
                            chunks = loads(decoded_chunk)
                            if "message" in chunks:
                                try:
                                    chunks["message"] = ast.literal_eval(chunks["message"])
                                except Exception:
                                    pass
                            formatted_error_chunk = dumps(chunks).encode(encoding="utf-8")
                        except json.JSONDecodeError:
                            error_detail = {"detail": f"Received non-JSON error response from backend (status: {response.status_code}): {error_body_bytes.decode(encoding='utf-8', errors='ignore')}"}
                            formatted_error_chunk = dumps(error_detail).encode(encoding="utf-8")
                        except Exception as e:
                            error_detail = {"detail": f"Error processing backend error response (status: {response.status_code}): {str(e)}"}
                            formatted_error_chunk = dumps(error_detail).encode(encoding="utf-8")
                        
                        yield formatted_error_chunk, response.status_code
                        return # Exit the generator

                    # If status is OK, proceed with streaming chunks
                    first_chunk = True
                    async for chunk in response.aiter_raw():
                        # Add additional data to the first chunk
                        if first_chunk and additional_data_value and additional_data_key:
                            # This logic might still be problematic for SSE, but keep for now
                            try:
                                decoded_chunk = chunk.decode(encoding="utf-8")
                                if decoded_chunk.startswith("data: "):
                                    parsed_chunk = loads(decoded_chunk.lstrip("data: ").rstrip("\n\n"))
                                    parsed_chunk[additional_data_key] = additional_data_value
                                    chunk = f"data: {dumps(parsed_chunk)}\n\n".encode("utf-8")
                                else:
                                     logger.warning("First chunk did not start with 'data: ', cannot add additional data reliably.")
                            except Exception as parse_e:
                                logger.error(f"Error processing first chunk to add data: {parse_e}")
                                # Yield original chunk if processing fails
                                pass 

                        first_chunk = False
                        yield chunk, response.status_code # Yield normal chunk

            except (httpx.TimeoutException, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as e:
                yield dumps({"detail": "Request timed out, model is too busy."}).encode(), 504
            except Exception as e:
                # This should ideally catch errors during stream setup
                logger.error(f"Exception during httpx stream setup or context: {type(e).__name__} - {e}")
                logger.error(traceback.format_exc())
                yield dumps({"detail": f"Stream Setup Error: {type(e).__name__}"}).encode(), 500
