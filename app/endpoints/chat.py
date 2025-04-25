from typing import List, Tuple, Union

from fastapi import APIRouter, Depends, Request, Security
from sqlalchemy.ext.asyncio import AsyncSession

from app.helpers import Authorization, StreamingResponseWithStatusCode
from app.schemas.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionRequest
from app.schemas.search import Search
from app.sql.session import get_db as get_session
from app.utils.exceptions import CollectionNotFoundException
from app.utils.lifespan import context
from app.utils.variables import ENDPOINT__CHAT_COMPLETIONS
from app.utils.usage_decorator import log_usage

router = APIRouter()


@router.post(path=ENDPOINT__CHAT_COMPLETIONS, dependencies=[Security(dependency=Authorization())])
@log_usage
async def chat_completions(request: Request, body: ChatCompletionRequest, session: AsyncSession = Depends(get_session)) -> Union[ChatCompletion, ChatCompletionChunk]:  # fmt: off
    """Creates a model response for the given chat conversation.

    **Important**: any others parameters are authorized, depending of the model backend. For example, if model is support by vLLM backend, additional
    fields are available (see https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/protocol.py#L209). Similarly, some defined fields
    may be ignored depending on the backend used and the model support.
    """

    # retrieval augmentation generation
    async def retrieval_augmentation_generation(body: ChatCompletionRequest, session: AsyncSession) -> Tuple[dict, List[dict]]:
        results = []
        if body.search:
            if not context.documents:
                raise CollectionNotFoundException()

            results = await context.documents.search(
                session=session,
                collection_ids=body.search_args.collections,
                prompt=body.messages[-1]["content"],
                method=body.search_args.method,
                k=body.search_args.k,
                user_id=request.app.state.user.id,
                web_search=body.search_args.web_search,
                score_threshold=body.search_args.score_threshold,
            )
            if results:
                chunks = "\n".join([result.chunk.content for result in results])
                body.messages[-1]["content"] = body.search_args.template.format(prompt=body.messages[-1]["content"], chunks=chunks)

        body_dict = body.model_dump()
        body_dict.pop("search", None)
        body_dict.pop("search_args", None)

        results = [result.model_dump() for result in results]
        return body_dict, results

    rag_body_dict, rag_results = await retrieval_augmentation_generation(body=body, session=session)

    # select client
    model_router = context.models(model=body.model)
    client = model_router.get_client(endpoint=ENDPOINT__CHAT_COMPLETIONS)

    # Prepare arguments for the OpenAI client call
    openai_kwargs = {
        "model": client.model,
        "messages": rag_body_dict.get("messages", body.messages),
        "stream": body.stream,
        # Add optional parameters only if they are not None in the original request
    }
    if body.temperature is not None: openai_kwargs["temperature"] = body.temperature
    if body.max_completion_tokens is not None: openai_kwargs["max_tokens"] = body.max_completion_tokens
    if body.n is not None: openai_kwargs["n"] = body.n
    if body.stop is not None: openai_kwargs["stop"] = body.stop
    if body.presence_penalty is not None: openai_kwargs["presence_penalty"] = body.presence_penalty
    if body.frequency_penalty is not None: openai_kwargs["frequency_penalty"] = body.frequency_penalty
    if body.logit_bias is not None: openai_kwargs["logit_bias"] = body.logit_bias
    if body.logprobs is not None: openai_kwargs["logprobs"] = body.logprobs
    if body.top_logprobs is not None: openai_kwargs["top_logprobs"] = body.top_logprobs
    if body.response_format is not None: openai_kwargs["response_format"] = body.response_format
    if body.seed is not None: openai_kwargs["seed"] = body.seed
    if body.stream_options is not None: openai_kwargs["stream_options"] = body.stream_options
    if body.tools is not None:
        openai_kwargs["tools"] = body.tools
        # Only include tool_choice if tools are present
        if body.tool_choice is not None:
            openai_kwargs["tool_choice"] = body.tool_choice
    
    # No longer need the final filter, as we added keys conditionally
    # openai_kwargs = {k: v for k, v in openai_kwargs.items() if v is not None}

    # non-stream case
    if not body.stream:
        response = await client.forward_request(
            endpoint=ENDPOINT__CHAT_COMPLETIONS,
            method="POST",
            json=openai_kwargs,
            additional_data_value=rag_results,
            additional_data_key="search_results",
        )
        return ChatCompletion(**response.json())

    # stream case
    return StreamingResponseWithStatusCode(
        content=client.forward_stream(
            endpoint=ENDPOINT__CHAT_COMPLETIONS,
            method="POST",
            json=openai_kwargs,
            additional_data_value=rag_results,
            additional_data_key="search_results",
        ),
        media_type="text/event-stream",
    )
