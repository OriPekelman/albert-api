from typing import List, Tuple

from openai import OpenAI
import requests
import streamlit as st
import logging
import json

from ui.settings import settings

logger = logging.getLogger(__name__)


def generate_stream(messages: List[dict], params: dict, rag: bool, rerank: bool) -> Tuple[str, List[str]]:
    sources = []
    try:
        if rag:
            prompt = messages[-1]["content"]
            k = params["rag"]["k"] * 2 if rerank else params["rag"]["k"]
            data = {"collections": params["rag"]["collections"], "k": k, "prompt": messages[-1]["content"], "score_threshold": None}
            response = requests.post(
                url=f"{settings.playground.api_url}/v1/search", json=data, headers={"Authorization": f"Bearer {st.session_state['user'].api_key}"}
            )
            if response.status_code != 200:
                error_detail = response.json().get('detail', str(response.text))
                raise Exception(f"Search API error: {response.status_code} - {error_detail}")

            prompt_template = """Réponds à la question suivante de manière claire en te basant sur les extraits de documents ci-dessous. Si les documents ne sont pas pertinents pour répondre à la question, réponds que tu ne sais pas ou réponds directement la question à l'aide de tes connaissances. Réponds en français.
La question de l'utilisateur est : {prompt}

Les documents sont :

{chunks}
"""
            chunks = [chunk["chunk"] for chunk in response.json()["data"]]

            if rerank:
                data = {
                    "prompt": prompt,
                    "input": [chunk["content"] for chunk in chunks],
                }
                response = requests.post(
                    url=f"{settings.playground.api_url}/v1/rerank", json=data, headers={"Authorization": f"Bearer {st.session_state['user'].api_key}"}
                )
                if response.status_code != 200:
                    error_detail = response.json().get('detail', str(response.text))
                    raise Exception(f"Rerank API error: {response.status_code} - {error_detail}")

                rerank_scores = sorted(response.json()["data"], key=lambda x: x["score"])
                chunks = [chunks[result["index"]] for result in rerank_scores[: params["rag"]["k"]]]

            sources = list(set([chunk["metadata"]["document_name"] for chunk in chunks]))
            chunks = [chunk["content"] for chunk in chunks]
            prompt = prompt_template.format(prompt=prompt, chunks="\n\n".join(chunks))
            messages = messages[:-1] + [{"role": "user", "content": prompt}]

        client = OpenAI(base_url=f"{settings.playground.api_url}/v1", api_key=st.session_state["user"].api_key)
        request_params = {
            "stream": True,
            "messages": messages,
            "model": params["sampling_params"]["model"],
            "temperature": params["sampling_params"].get("temperature", 0.2),
        }
        if params["sampling_params"].get("max_tokens"):
            request_params["max_tokens"] = params["sampling_params"]["max_tokens"]
        
        try:
            # Create a streaming response
            response = client.chat.completions.create(**request_params)
            if not response:
                raise Exception("No response received from the chat completions API")
            
            # Return the response object directly
            return response, sources
        except Exception as e:
            logger.error(f"Error creating chat completion: {str(e)}")
            raise
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None, sources
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON response: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None, sources
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        return None, sources
