"""기상청 API허브 초단기실황 조회 및 응답 정규화."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

KST = ZoneInfo("Asia/Seoul")
WEATHER_URL = (
    "https://apihub.kma.go.kr/api/typ02/openApi/"
    "VilageFcstInfoService_2.0/getUltraSrtNcst"
)
# 각 시청 소재지 기준 대표 격자입니다.
LOCATIONS = {
    "서울": (60, 127), "부산": (98, 76), "대구": (89, 90),
    "인천": (55, 124), "광주": (58, 74), "대전": (67, 100),
    "울산": (102, 84), "세종": (66, 103), "수원": (60, 121),
    "춘천": (73, 134), "강릉": (92, 131), "청주": (69, 106),
    "전주": (63, 89), "목포": (50, 67), "여수": (73, 66),
    "포항": (102, 94), "창원": (90, 77), "제주": (52, 38),
}
PRECIPITATION = {
    "0": "없음", "1": "비", "2": "비/눈", "3": "눈",
    "5": "빗방울", "6": "빗방울/눈날림", "7": "눈날림",
}
FIELDS = {
    "T1H": ("temperature", "°C"), "REH": ("humidity", "%"),
    "RN1": ("precipitation_1h", "mm"), "WSD": ("wind_speed", "m/s"),
    "VEC": ("wind_direction", "°"),
}


class WeatherError(Exception):
    """인증키나 요청 URL을 포함하지 않는 사용자용 오류."""


class NoWeatherData(WeatherError):
    pass


def resolve_location(location: str) -> str:
    name = location.strip()
    for suffix in ("특별자치시", "특별자치도", "특별시", "광역시", "시"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    if name not in LOCATIONS:
        raise WeatherError("지원하지 않는 지역입니다. 지원 지역: " + ", ".join(LOCATIONS))
    return name


def latest_base_time(now: datetime | None = None) -> datetime:
    """매시 실황의 제공 지연(40분)을 고려하며 자정에도 전일로 이동합니다."""
    now = now or datetime.now(KST)
    return (now.astimezone(KST) - timedelta(minutes=40)).replace(
        minute=0, second=0, microsecond=0
    )


def _parse_items(response: requests.Response) -> list[dict]:
    if response.status_code in (401, 403):
        raise WeatherError(
            "기상청 인증에 실패했습니다. KMA_API_KEY에 API허브 인증키를 설정하고 "
            "해당 API의 활용신청/사용 권한을 확인하세요. 공공데이터포털 키와는 다릅니다."
        )
    if response.status_code == 429:
        raise WeatherError("기상청 API 호출 한도를 초과했습니다. 잠시 후 다시 시도하세요.")
    response.raise_for_status()
    try:
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError
        # API허브 인증 오류는 기상청 표준 response/header 구조와 다릅니다.
        hub_result = payload.get("result")
        if isinstance(hub_result, dict) and str(hub_result.get("status")) in ("401", "403"):
            raise WeatherError("기상청 인증키와 API 활용신청 상태를 확인하세요.")
        envelope = payload["response"]
        code = str(envelope["header"]["resultCode"])
        if code == "03":
            raise NoWeatherData("기상청 관측 자료가 아직 없습니다.")
        if code != "00":
            if code in {"20", "21", "22", "30", "31", "32", "33"}:
                raise WeatherError("기상청 인증키, 사용 권한 및 일일 호출 한도를 확인하세요.")
            raise WeatherError("기상청에서 조회 오류를 반환했습니다. 잠시 후 다시 시도하세요.")
        body = envelope["body"]
        items = (body.get("items") or {}).get("item", [])
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError
        if not items:
            raise NoWeatherData("기상청 관측 자료가 아직 없습니다.")
        return items
    except (ValueError, KeyError, TypeError, AttributeError):
        raise WeatherError(
            "기상청 응답 형식이 올바르지 않습니다(JSON/자료 구조 오류). "
            "API허브 인증키와 서비스 상태를 확인하세요."
        ) from None


def fetch_weather(api_key: str, location: str, now: datetime | None = None) -> dict:
    """최신 실황을 조회하고 자료 미발표 시 한 시간 전 자료를 한 번 조회합니다."""
    if not api_key or not api_key.strip():
        raise WeatherError("KMA_API_KEY가 없습니다. .env에 기상청 API허브 인증키를 설정하세요.")
    location = resolve_location(location)
    nx, ny = LOCATIONS[location]
    base = latest_base_time(now)
    retry = Retry(
        total=1, backoff_factor=0.3, status_forcelist=(502, 503, 504),
        allowed_methods=frozenset({"GET"}), raise_on_status=False,
        respect_retry_after_header=False,
    )
    try:
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))
            for offset in range(2):
                observed = base - timedelta(hours=offset)
                response = session.get(WEATHER_URL, params={
                    "pageNo": 1, "numOfRows": 100, "dataType": "JSON",
                    "base_date": observed.strftime("%Y%m%d"),
                    "base_time": observed.strftime("%H00"),
                    "nx": nx, "ny": ny, "authKey": api_key.strip(),
                }, timeout=(3.05, 10))
                try:
                    items = _parse_items(response)
                except NoWeatherData:
                    continue
                measurements = {}
                for item in items:
                    category, value = item.get("category"), item.get("obsrValue")
                    if value is None or str(value) in {"-998", "-999"}:
                        continue
                    if category in FIELDS:
                        name, unit = FIELDS[category]
                        measurements[name] = {"value": value, "unit": unit}
                    elif category == "PTY":
                        measurements["precipitation_type"] = PRECIPITATION.get(str(value), "알 수 없음")
                if not measurements:
                    continue
                return {
                    "source": "기상청 초단기실황", "location": location,
                    "grid": {"nx": nx, "ny": ny},
                    "observed_at": observed.isoformat(timespec="minutes"),
                    "fallback": bool(offset), "measurements": measurements,
                }
    except requests.Timeout:
        raise WeatherError("기상청 응답 시간이 초과되었습니다. 잠시 후 다시 시도하세요.") from None
    except requests.RequestException:
        # requests 예외에는 authKey가 들어간 URL이 포함될 수 있으므로 그대로 노출하지 않습니다.
        raise WeatherError("기상청에 연결하지 못했습니다. 네트워크와 서비스 상태를 확인하세요.") from None
    raise NoWeatherData("최근 두 시간의 기상청 관측 자료가 없습니다. 잠시 후 다시 시도하세요.")
