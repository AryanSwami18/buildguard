from app.intelligence import LLMProvider, LLMRequest, LLMResponse


def test_llm_request_can_be_created() -> None:
    request = LLMRequest(
        system_prompt="Interpret the supplied context.",
        user_prompt="Review this build step.",
        structured_output_instructions="Return JSON.",
    )

    assert request.system_prompt == "Interpret the supplied context."
    assert request.user_prompt == "Review this build step."
    assert request.structured_output_instructions == "Return JSON."


def test_llm_response_can_be_created() -> None:
    response = LLMResponse(content='{"result": "ok"}')

    assert response.content == '{"result": "ok"}'


def test_provider_protocol_can_be_implemented_by_a_fake_provider() -> None:
    class FakeProvider:
        def generate(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content=f"test response: {request.user_prompt}")

    provider: LLMProvider = FakeProvider()
    request = LLMRequest(
        system_prompt="Interpret the supplied context.",
        user_prompt="Review this build step.",
    )

    response = provider.generate(request)

    assert response.content == "test response: Review this build step."
