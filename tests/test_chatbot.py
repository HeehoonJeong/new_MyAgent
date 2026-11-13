import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from streamlit.testing.v1 import AppTest

from chatbot import cached_weather, conversation_context, get_agent, stream_answer
from weather import WeatherError

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class ChatbotTests(unittest.TestCase):
    def tearDown(self):
        cached_weather.clear()
        get_agent.clear()

    def test_history_preserves_recent_complete_turns(self):
        messages = [
            {"role": "user", "content": "서울 날씨"},
            {"role": "assistant", "content": "서울은 10도입니다."},
            {"role": "user", "content": "부산은?"},
            {"role": "assistant", "content": "부산은 15도입니다."},
            {"role": "user", "content": "거기 바람은?"},
        ]
        self.assertEqual(conversation_context(messages, max_turns=2), messages[2:])
        self.assertEqual(conversation_context(messages, max_chars=8), messages[-1:])
        self.assertEqual(conversation_context(messages), messages)

    def test_weather_cache_and_failed_call_recovery(self):
        with patch("chatbot.fetch_weather", return_value={"temperature": 10}) as fetch:
            cached_weather("key", "서울", "12")
            cached_weather("key", "서울", "12")
            fetch.assert_called_once()
            cached_weather("key", "서울", "13")
            cached_weather("key", "부산", "13")
            self.assertEqual(fetch.call_count, 3)
        with patch("chatbot.fetch_weather", side_effect=[WeatherError("temporary"), {"ok": True}]) as fetch:
            with self.assertRaises(WeatherError):
                cached_weather("other-key", "서울", "12")
            self.assertEqual(cached_weather("other-key", "서울", "12"), {"ok": True})
            self.assertEqual(fetch.call_count, 2)

    def test_stream_filters_tool_output(self):
        agent = Mock()
        agent.stream.return_value = iter([
            (AIMessageChunk(content="", tool_call_chunks=[{"name": "check_weather", "args": "{}", "id": "1", "index": 0}]), {"langgraph_node": "model"}),
            (ToolMessage(content="raw weather", tool_call_id="1"), {"langgraph_node": "tools"}),
            (AIMessageChunk(content="서울 "), {"langgraph_node": "model"}),
            (AIMessageChunk(content="10도"), {"langgraph_node": "model"}),
        ])
        self.assertEqual("".join(stream_answer(agent, [{"role": "user", "content": "날씨"}])), "서울 10도")

    def test_real_agent_graph_uses_selected_location(self):
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "check_weather", "args": {}, "id": "weather-1", "type": "tool_call"}]),
            AIMessage(content="부산은 15도입니다."),
        ])
        with patch("chatbot.init_chat_model", return_value=model), patch("chatbot.cached_weather", return_value={"temperature": 15}) as fetch:
            agent = get_agent("test-model", "fake-openai-key", "fake-kma-key", "부산")
            result = agent.invoke({"messages": [{"role": "user", "content": "지금 날씨"}]})
        self.assertEqual(fetch.call_args.args[:2], ("fake-kma-key", "부산"))
        self.assertEqual(result["messages"][-1].content, "부산은 15도입니다.")

    def test_app_missing_key_warning(self):
        with patch("dotenv.load_dotenv"), patch.dict("os.environ", {"OPENAI_API_KEY": "", "KMA_API_KEY": ""}):
            app = AppTest.from_file(APP_PATH).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.warning), 2)
        self.assertTrue(app.chat_input[0].disabled)

    def test_app_followup_and_retry(self):
        def fake_stream(agent, messages):
            yield "서울은 10도입니다."

        with patch("dotenv.load_dotenv"), patch.dict("os.environ", {"OPENAI_API_KEY": "fake", "KMA_API_KEY": "fake"}), patch("chatbot.get_agent", return_value=Mock()), patch("chatbot.stream_answer", side_effect=fake_stream) as stream:
            app = AppTest.from_file(APP_PATH).run()
            app.chat_input[0].set_value("서울 날씨").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state.messages), 2)
            stream.side_effect = RuntimeError("private error")
            app.chat_input[0].set_value("거기 바람은?").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state.messages), 3)
            self.assertEqual(len(app.error), 1)
            self.assertNotIn("private error", app.error[0].value)
            stream.side_effect = fake_stream
            next(button for button in app.button if button.label == "응답 다시 시도").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.session_state.messages), 4)
            self.assertFalse(app.error)
            self.assertEqual(stream.call_args.args[1][0]["content"], "서울 날씨")
            next(button for button in app.button if button.label == "대화 초기화").click().run()
            self.assertEqual(app.session_state.messages, [])


if __name__ == "__main__":
    unittest.main()
