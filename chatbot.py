"""대화 문맥 관리와 Streamlit에서 재사용하는 날씨 에이전트."""

from datetime import datetime

import streamlit as st
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import AIMessageChunk

from weather import KST, LOCATIONS, WeatherError, fetch_weather, resolve_location


def conversation_context(messages: list[dict], max_turns: int = 10, max_chars: int = 16000) -> list[dict]:
    """최근 사용자 질문을 포함하되 완결된 이전 대화 단위로 입력량을 제한합니다."""
    turns = []
    for message in messages:
        if message["role"] == "user":
            turns.append([])
        if turns and message["role"] in {"user", "assistant"}:
            turns[-1].append({"role": message["role"], "content": message["content"]})
    selected, size = [], 0
    for turn in reversed(turns[-max_turns:]):
        length = sum(len(message["content"]) for message in turn)
        if selected and size + length > max_chars:
            break
        selected.append(turn)
        size += length
    return [message for turn in reversed(selected) for message in turn]


@st.cache_data(ttl=300, max_entries=256, show_spinner=False)
def cached_weather(api_key: str, location: str, base_time: str) -> dict:
    # 발표 시각도 캐시 키에 포함하여 새 발표 자료를 이전 캐시로 가리지 않습니다.
    # 예외는 캐시되지 않아 설정/네트워크 오류를 고친 뒤 즉시 재시도할 수 있습니다.
    return fetch_weather(api_key, location)


@st.cache_resource(show_spinner=False)
def get_agent(model_name: str, openai_api_key: str, kma_api_key: str, default_location: str):
    from weather import latest_base_time

    @tool
    def check_weather(location: str = "") -> dict:
        """지정한 한국 도시의 최신 관측 날씨를 조회합니다. 빈 지역은 화면의 기본 지역입니다.

        현재 기온·습도·강수·풍속만 제공하며 미래 예보와 하늘 상태는 제공하지 않습니다.
        """
        try:
            name = resolve_location(location or default_location)
            return cached_weather(kma_api_key, name, latest_base_time().isoformat())
        except WeatherError as exc:
            return {"error": str(exc)}

    @tool
    def current_time() -> str:
        """현재 한국 날짜와 시간(KST)을 조회합니다."""
        return datetime.now(KST).isoformat(timespec="minutes")

    model = init_chat_model(
        model_name, model_provider="openai", api_key=openai_api_key,
        timeout=30, max_retries=1,
    )
    return create_agent(
        model, tools=[check_weather, current_time],
        system_prompt=(
            "당신은 한국어로 간결하고 친절하게 답하는 날씨 챗봇입니다. "
            f"기본 조회 지역은 {default_location}입니다. 지원 지역: {', '.join(LOCATIONS)}. "
            "실시간 날씨 질문에는 반드시 check_weather를 사용하세요. "
            "이전 대화에서 정한 지역을 후속 질문에도 적용하고 지역을 임의로 바꾸지 마세요. "
            "조회 결과의 지역과 관측 시각(한국 시간), 수치와 단위를 알려주세요. "
            "실황은 현재 관측 자료이며 미래 예보가 아닙니다. 내일/미래의 날씨, "
            "강수확률, 미세먼지, 하늘 상태는 조회할 수 없다고 안내하고 추측하지 마세요. "
            "지원하지 않는 지역은 지원 지역 중 선택하도록 요청하세요. "
            "도구 오류가 나면 원인과 해결 방법을 안내하고 날씨 수치를 만들지 마세요. "
            "같은 질문에서 동일 지역 도구를 반복 호출하지 마세요."
        ),
    )


def stream_answer(agent, messages: list[dict]):
    for chunk, metadata in agent.stream(
        {"messages": conversation_context(messages)},
        config={"recursion_limit": 8}, stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk) and metadata.get("langgraph_node") == "model":
            if chunk.text:
                yield chunk.text
