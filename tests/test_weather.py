import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import requests

from weather import KST, NoWeatherData, WeatherError, fetch_weather, latest_base_time


def response(items=None, code="00", status=200):
    result = Mock(status_code=status)
    result.json.return_value = {"response": {
        "header": {"resultCode": code},
        "body": {"items": {"item": items if items is not None else [
            {"category": "T1H", "obsrValue": "15.3"},
            {"category": "PTY", "obsrValue": "0"},
        ]}},
    }}
    return result


class WeatherTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("weather.requests.Session")
        self.session_class = patcher.start()
        self.addCleanup(patcher.stop)
        self.session = self.session_class.return_value.__enter__.return_value
        self.now = datetime(2026, 11, 13, 12, 45, tzinfo=KST)

    def test_missing_key_does_not_send_request(self):
        with self.assertRaisesRegex(WeatherError, "KMA_API_KEY"):
            fetch_weather("", "서울", self.now)
        self.session_class.assert_not_called()

    def test_publication_boundary_and_midnight(self):
        before = datetime(2026, 1, 1, 0, 39, tzinfo=KST)
        self.assertEqual(latest_base_time(before).strftime("%Y%m%d %H%M"), "20251231 2300")
        self.assertEqual(latest_base_time(before.replace(minute=40)).hour, 0)

    def test_normalized_response_and_city_grid(self):
        self.session.get.return_value = response()
        result = fetch_weather("test-key", "서울특별시", self.now)
        params = self.session.get.call_args.kwargs["params"]
        self.assertEqual((params["nx"], params["ny"]), (60, 127))
        self.assertEqual(params["base_time"], "1200")
        self.assertEqual(params["authKey"], "test-key")
        self.assertEqual(result["measurements"]["temperature"], {"value": "15.3", "unit": "°C"})
        self.assertEqual(result["measurements"]["precipitation_type"], "없음")
        self.assertNotIn("test-key", str(result))

    def test_no_data_falls_back_across_midnight(self):
        self.session.get.side_effect = [response(code="03"), response()]
        result = fetch_weather("key", "부산", self.now.replace(hour=0))
        params = self.session.get.call_args.kwargs["params"]
        self.assertEqual((params["base_date"], params["base_time"]), ("20261112", "2300"))
        self.assertEqual((params["nx"], params["ny"]), (98, 76))
        self.assertTrue(result["fallback"])
        self.assertEqual(result["observed_at"], "2026-11-12T23:00+09:00")

    def test_empty_data_retries_only_once(self):
        self.session.get.return_value = response(items=[])
        with self.assertRaises(NoWeatherData):
            fetch_weather("key", "서울", self.now)
        self.assertEqual(self.session.get.call_count, 2)

    def test_authentication_and_quota_fail_without_fallback(self):
        for status, message in [(401, "인증"), (403, "인증"), (429, "한도")]:
            with self.subTest(status=status):
                self.session.get.reset_mock()
                self.session.get.return_value = response(status=status)
                with self.assertRaisesRegex(WeatherError, message):
                    fetch_weather("secret", "서울", self.now)
                self.session.get.assert_called_once()

    def test_http_200_api_error_is_not_weather(self):
        self.session.get.return_value = response(code="30")
        with self.assertRaisesRegex(WeatherError, "인증키"):
            fetch_weather("key", "서울", self.now)
        self.session.get.assert_called_once()

    def test_hub_error_envelope(self):
        self.session.get.return_value = response()
        self.session.get.return_value.json.return_value = {"result": {"status": 401}}
        with self.assertRaisesRegex(WeatherError, "인증키"):
            fetch_weather("key", "서울", self.now)

    def test_malformed_responses(self):
        for payload in [[], {}, {"response": None}, {"response": {"header": {"resultCode": "00"}, "body": []}}]:
            with self.subTest(payload=payload):
                self.session.get.return_value = response()
                self.session.get.return_value.json.return_value = payload
                with self.assertRaisesRegex(WeatherError, "응답 형식"):
                    fetch_weather("key", "서울", self.now)

    def test_non_json_response(self):
        self.session.get.return_value = response()
        self.session.get.return_value.json.side_effect = ValueError("HTML or XML")
        with self.assertRaisesRegex(WeatherError, "응답 형식"):
            fetch_weather("key", "서울", self.now)

    def test_request_error_does_not_expose_key(self):
        for exception in [requests.Timeout("url?authKey=secret"), requests.ConnectionError("url?authKey=secret")]:
            with self.subTest(exception=type(exception)):
                self.session.get.side_effect = exception
                with self.assertRaises(WeatherError) as caught:
                    fetch_weather("secret", "서울", self.now)
                self.assertNotIn("secret", str(caught.exception))

    def test_unknown_city_is_not_silently_replaced(self):
        with self.assertRaisesRegex(WeatherError, "지원하지 않는 지역"):
            fetch_weather("key", "없는도시", self.now)
        self.session_class.assert_not_called()

    def test_missing_observations_are_not_reported(self):
        self.session.get.return_value = response(items=[
            {"category": "T1H", "obsrValue": "-999"},
            {"category": "REH", "obsrValue": "60"},
        ])
        result = fetch_weather("key", "서울", self.now)
        self.assertNotIn("temperature", result["measurements"])
        self.assertEqual(result["measurements"]["humidity"]["value"], "60")


if __name__ == "__main__":
    unittest.main()
