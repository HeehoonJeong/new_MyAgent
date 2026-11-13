import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError
from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, RateLimitError

from chatbot import get_agent, stream_answer
from weather import LOCATIONS


def main():
    load_dotenv(Path(__file__).resolve().parent / ".env")
    st.set_page_config(page_title="날씨 정보 챗봇", page_icon="🌤️")
    st.title("날씨 정보 챗봇")
    st.caption("기상청 최신 관측 자료 · 한국 시간 기준 · 동일 지역 조회는 최대 5분간 캐시")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    kma_key = os.getenv("KMA_API_KEY", "").strip()
    model_name = os.getenv("OPENAI_MODEL", "gpt-6-astra").strip() or "gpt-6-astra"

    with st.sidebar:
        location = st.selectbox("기본 조회 지역", list(LOCATIONS))
        st.caption("각 도시의 대표 격자 기준입니다. 대화에서 다른 지원 도시를 지정할 수 있습니다.")
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("chat_error", None)
            st.rerun()
        st.caption(f"모델: {model_name}")
        st.caption("현재 관측 기온·습도·강수·바람을 제공합니다.")

    if not kma_key:
        st.warning(
            "기상청 인증키가 없습니다. 프로젝트 .env에 KMA_API_KEY를 설정하세요. "
            "기상청 API허브(apihub.kma.go.kr)에서 인증키를 발급하고 "
            "초단기실황 API 활용신청을 완료해야 합니다."
        )
    if not openai_key:
        st.warning("프로젝트 .env에 OPENAI_API_KEY를 설정하세요.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending = bool(st.session_state.messages and st.session_state.messages[-1]["role"] == "user")
    if st.session_state.get("chat_error"):
        st.error(st.session_state.chat_error)
    retry = pending and st.button("응답 다시 시도", disabled=not openai_key)
    prompt = st.chat_input(
        "예: 서울 지금 날씨 알려줘", disabled=not openai_key or pending, max_chars=4000,
    )
    if not (prompt or retry):
        return
    st.session_state.pop("chat_error", None)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    try:
        agent = get_agent(model_name, openai_key, kma_key, location)
        with st.chat_message("assistant"):
            with st.spinner("답변을 준비하고 있습니다…"):
                answer = st.write_stream(stream_answer(agent, st.session_state.messages))
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("empty answer")
    except AuthenticationError:
        error = "OpenAI 인증에 실패했습니다. OPENAI_API_KEY를 확인하세요."
    except RateLimitError:
        error = "모델 사용 한도에 도달했습니다. 사용량/결제 설정을 확인한 뒤 다시 시도하세요."
    except (APITimeoutError, APIConnectionError):
        error = "모델 서버 연결이 지연되거나 실패했습니다. 잠시 후 다시 시도하세요."
    except APIStatusError:
        error = "모델 요청에 실패했습니다. OPENAI_MODEL의 모델명과 사용 권한을 확인하세요."
    except GraphRecursionError:
        error = "도구 실행 횟수를 초과했습니다. 질문을 구체적으로 바꾸거나 다시 시도하세요."
    except Exception:
        error = "답변 생성에 실패했습니다. 설정을 확인하고 다시 시도하세요."
    else:
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
        return

    st.session_state.chat_error = error
    st.rerun()


if __name__ == "__main__":
    main()
