import os
import streamlit as st
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path


load_dotenv()

# # app.py 파일이 위치한 폴더 기준
# BASE_DIR = Path(__file__).resolve().parent

# # .env 파일 경로 지정
# env_path = BASE_DIR / ".env"

# # .env 파일 불러오기
# load_dotenv(dotenv_path=env_path)

# # API 키 확인
# print("ENV 파일 존재:", env_path.exists())
# print("API KEY 로드:", bool(os.getenv("OPENAI_API_KEY")))

# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

# # API 키 존재 여부 확인
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")


# @tool
# def current_time() -> str:
#     '실시간 한국시간을 알려주는 도구입니다.'
#     # UTC 기준 시간
#     utc_now = datetime.datetime.now(datetime.timezone.utc)

#     # 한국 시간으로 변환 (UTC + 9시간)
#     kst_timezone = datetime.timezone(datetime.timedelta(hours=9))
#     kst_now = utc_now.astimezone(kst_timezone)
#     return f"현재 한국 시간: {kst_now.strftime('%Y-%m-%d %H:%M')}"


@tool
def check_weather() -> dict:
    """기상청 API로 최근 초단기실황 날씨를 조회합니다."""

    url = "https://apihub.kma.go.kr/api/typ02/openApi/VilageFcstInfoService_2.0/getUltraSrtNcst"

    # 기상청 발표 지연을 고려해 1시간 전 정시 조회
    now = datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(hours=1)

    params = {
        "pageNo": 1,
        "numOfRows": 100,
        "dataType": "JSON",
        "base_date": now.strftime("%Y%m%d"),
        "base_time": now.strftime("%H00"),
        "nx": 55,
        "ny": 127,
        "authKey": os.getenv("KMA_API_KEY")
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        return {"error": str(e)}
    except ValueError:
        return {"error": "JSON 응답을 해석하지 못했습니다."}


model = init_chat_model('gpt-6-astra')
agent = create_agent(model, 
                     tools=[check_weather], #current_time], 
                     system_prompt = '당신은 실시간 기상정보 조회 agent입니다.',
                     
                     )


if 'messages' not in st.session_state:
    st.session_state.messages = []

st.title('날씨 정보 챗봇입니다.')

prompt = st.chat_input('무엇을 도와드릴까요?')

# 이전 대화 화면에 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if prompt:
    st.session_state.messages.append({'role': 'user', 'content':prompt})
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": prompt}
        ]
    })

    answer = response["messages"][-1].content

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })








