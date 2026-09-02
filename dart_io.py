"""DART OpenAPI 호출 유틸 (classifier 전용, 독립 모듈).

dart_alert_bot.py / lead_radar.py를 import하지 않는다. 이 폴더는 나중에
별도 public 레포로 분리될 예정이라 필요한 로직만 여기 복사해 둔다.

- _dart_get / _date_windows: lead_radar.py의 동작을 그대로 복사.
- fetch_list_range: list.json 페이지네이션을 캡슐화(중복 로직 제거).
- get_document_text: document.xml 원문 추출. 태그를 줄바꿈으로 치환해
  문단 구조를 보존한다(라벨 누출 라인 제거에 필요 — build_dataset.py 참고).
"""

import os
import re
import io
import time
import zipfile
from datetime import timedelta

import requests
from dotenv import load_dotenv

load_dotenv()

DART_API_KEY = os.getenv("DART_API_KEY")

LIST_URL = "https://opendart.fss.or.kr/api/list.json"
DOCUMENT_URL = "https://opendart.fss.or.kr/api/document.xml"

call_count = 0  # 이번 프로세스에서의 DART API 호출 횟수 (보고용)


def _dart_get(url, params, max_retries=3):
    """공통 GET + 지수 백오프. 응답 객체 반환(실패 시 None)."""
    global call_count
    delay = 1.0
    for attempt in range(max_retries):
        call_count += 1
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp
            print(f"  HTTP {resp.status_code} (재시도 {attempt + 1}/{max_retries})")
        except requests.RequestException as e:
            print(f"  요청 예외: {e} (재시도 {attempt + 1}/{max_retries})")
        time.sleep(delay)
        delay *= 2
    return None


def _date_windows(start, end, chunk_days):
    """[start, end]을 chunk_days 이하 구간들로 분할.

    OpenDART list.json은 검색기간이 최대 3개월이라 긴 구간을 한 번에
    못 받는다. 필요한 크기의 윈도우로 쪼개 각각 페이지네이션한다.
    """
    windows = []
    cur = start
    step = timedelta(days=chunk_days)
    while cur <= end:
        w_end = min(cur + step - timedelta(days=1), end)
        windows.append((cur, w_end))
        cur = w_end + timedelta(days=1)
    return windows


def fetch_list_range(bgn_de, end_de, page_count=100, max_pages=200, sleep_sec=0.2):
    """list.json에서 [bgn_de, end_de] 구간의 전체 공시를 페이지 끝까지 수집한다.

    pblntf_ty 필터를 걸지 않는다(분류기는 자율공시·대량보유보고서 등
    모든 유형을 다 봐야 한다). bgn_de/end_de는 "YYYYMMDD" 문자열.
    """
    items = []
    page = 1
    while page <= max_pages:
        params = {
            "crtfc_key": DART_API_KEY,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "page_no": page,
            "page_count": page_count,
        }
        resp = _dart_get(LIST_URL, params)
        if resp is None:
            print(f"  [{bgn_de}~{end_de}] page {page} 조회 실패 — 이 윈도우 중단")
            break

        data = resp.json()
        status = data.get("status")
        if status == "013":  # 조회된 데이터 없음
            break
        if status != "000":
            print(f"  [{bgn_de}~{end_de}] DART status={status}: {data.get('message')}")
            break

        items.extend(data.get("list", []))

        total_page = int(data.get("total_page", 1))
        if page >= total_page:
            break
        page += 1
        time.sleep(sleep_sec)

    return items


_BLOCK_TAG_RE = re.compile(r"</?(P|TD|TR|TITLE|SUBTITLE|TABLE)[^>]*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_STYLE_RE = re.compile(r"\.xforms[^}]+\}")


def get_document_text(rcept_no, max_chars=None):
    """DART 공시 원문 텍스트를 가져온다. 줄바꿈(문단 구조)을 보존한다.

    dart_alert_bot.get_document_text()와 달리 전체를 한 줄로 뭉개지 않는다.
    빈 줄은 제거하고, 각 줄 내부의 연속 공백만 정리한다.
    """
    params = {"crtfc_key": DART_API_KEY, "rcept_no": rcept_no}
    resp = _dart_get(DOCUMENT_URL, params)
    if resp is None:
        return None

    try:
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        names = z.namelist()
        if not names:
            return None
        xml = z.read(names[0]).decode("utf-8", errors="replace")

        xml = _BLOCK_TAG_RE.sub("\n", xml)
        text = _TAG_RE.sub(" ", xml)
        text = _STYLE_RE.sub("", text)

        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
        lines = [ln for ln in lines if ln]
        full_text = "\n".join(lines)

        if max_chars and len(full_text) > max_chars:
            full_text = full_text[:max_chars]
        return full_text
    except Exception as e:
        print(f"  원문 가져오기 실패({rcept_no}): {e}")
        return None
