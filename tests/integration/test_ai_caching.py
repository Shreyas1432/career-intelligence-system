import pytest

from src.core.ai.service import AIService
from src.core.cache import CacheManager
from src.core.config import settings
from src.core.prompts import prompt_registry
from tests.fixtures.ai import MockLLMClient


@pytest.mark.asyncio
async def test_ai_service_with_cache_integration(
    mock_llm_client: MockLLMClient,
    test_cache_manager: CacheManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verifies the integration of prompt loading, MockLLMClient generation,
    and SQLiteCacheStore persistence.
    """
    # 1. Enable caching for the duration of this test
    monkeypatch.setattr(settings.cache, "enabled", True)

    # 2. Register a prompt template dynamically in the registry
    prompt_registry.register_prompt_from_string(
        "test/cached_prompt_v1.0",
        "---\nsystem_prompt: 'You are an AI career coach'\n---\nHello {{ name }}! Assess skills: {{ skills }}.",
    )

    ai_service = AIService(llm_client=mock_llm_client)

    # Configure a distinct mock response
    mock_llm_client.add_response("Mock assessment of coding skills.")

    context = {"name": "Alice", "skills": "Python, SQL"}
    model = "test-coach-model"
    temperature = 0.5
    max_tokens = 50

    # Helper function to execute the template-based generate call via caching
    async def get_assessment() -> str:
        # Load and compile prompt to generate a cache key
        from src.core.prompts import prompt_manager

        compiled_prompt = prompt_manager.load_prompt("test/cached_prompt.md", context)
        system_prompt = "You are an AI career coach"

        cache_key = test_cache_manager.generate_ai_response_key(
            model=model,
            system_prompt=system_prompt,
            prompt=compiled_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        async def generate_call() -> str:
            return await ai_service.generate_from_template(
                template_path="test/cached_prompt.md",
                context=context,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        from typing import cast

        return cast(
            str,
            await test_cache_manager.get_or_set_async(
                key=cache_key,
                value_type="response",
                creator_fn=generate_call,
            ),
        )

    # First Call: Cache Miss (should query MockLLMClient and write to Cache)
    res1 = await get_assessment()
    assert res1 == "Mock assessment of coding skills."
    assert len(mock_llm_client.calls) == 1

    # Second Call: Cache Hit (should fetch from Cache without invoking MockLLMClient)
    mock_llm_client.add_response("This should NOT be called!")
    res2 = await get_assessment()
    assert res2 == "Mock assessment of coding skills."
    assert len(mock_llm_client.calls) == 1  # Still 1 call

    # Clear cache and verify it queries the client again
    test_cache_manager.clear()
    res3 = await get_assessment()
    assert res3 == "This should NOT be called!"
    assert len(mock_llm_client.calls) == 2  # Incremented to 2 calls
