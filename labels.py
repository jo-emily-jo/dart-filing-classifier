"""report_nm(공시 제목) → 학습 라벨 정규화.

라벨 8종은 build_dataset.py와 공유하는 LABELS 리스트로 고정한다.
"주요사항보고서(...)" 래핑 여부와 무관하게 같은 의사결정 유형은 같은
라벨로 모이게 하고, 그 외 "주요사항보고서(...)" 하위유형은 뭉뚱그려
"주요사항보고서(전체)"로 잡는다.
"""

import re

LABELS = [
    "단일판매ㆍ공급계약체결",
    "유상증자결정",
    "타법인주식및출자증권취득결정",
    "신규시설투자등",
    "자기주식취득결정",
    "주요사항보고서(전체)",
    "기타경영사항(자율공시)",
    "주식등의대량보유상황보고서",
]

# 주요사항보고서(...) 래핑 없이도 그 자체로 바로 라벨이 되는 의사결정 유형
_SPECIFIC_DECISIONS = {
    "유상증자결정",
    "타법인주식및출자증권취득결정",
    "신규시설투자등",
    "자기주식취득결정",
}

# 괄호를 포함한 문자열 자체가 라벨(괄호를 "부가정보"로 벗겨내면 안 됨)
_DIRECT_EXACT = {"기타경영사항(자율공시)"}

# 앞부분(첫 '(' 이전)만 일치하면 라벨로 인정 — 뒤에 (약식)/(추가) 등 부가정보가 붙어도 흡수
_DIRECT_PREFIX = {"단일판매ㆍ공급계약체결", "주식등의대량보유상황보고서"}

_LEADING_BRACKET_RE = re.compile(r"^\[[^\]]*\]\s*")
_MAJOR_REPORT_RE = re.compile(r"^주요사항보고서\((.*)\)$")


def normalize_label(report_nm):
    """report_nm을 8개 라벨 중 하나로, 해당 없으면 None으로 정규화한다."""
    if not report_nm:
        return None

    s = report_nm.strip()
    while True:
        m = _LEADING_BRACKET_RE.match(s)
        if not m:
            break
        s = s[m.end():]
    s = s.strip()

    s = re.sub(r"\s+", "", s)
    for dot in ("·", "‧", "⋅"):
        s = s.replace(dot, "ㆍ")

    if s in _DIRECT_EXACT:
        return s

    m = _MAJOR_REPORT_RE.match(s)
    if m:
        inner_core = re.split(r"[(]", m.group(1))[0]
        if inner_core in _SPECIFIC_DECISIONS:
            return inner_core
        return "주요사항보고서(전체)"

    core = re.split(r"[(]", s)[0]
    if core in _SPECIFIC_DECISIONS:
        return core
    if core in _DIRECT_PREFIX:
        return core

    return None


if __name__ == "__main__":
    import unittest

    class NormalizeLabelTest(unittest.TestCase):
        def test_direct_dot_variant(self):
            self.assertEqual(
                normalize_label("단일판매·공급계약체결"),
                "단일판매ㆍ공급계약체결",
            )

        def test_major_report_wrapped_specific_decision(self):
            self.assertEqual(
                normalize_label("[기재정정]주요사항보고서(유상증자결정)"),
                "유상증자결정",
            )

        def test_major_report_other_subtype_falls_to_catchall(self):
            self.assertEqual(
                normalize_label("주요사항보고서(전환사채권발행결정)"),
                "주요사항보고서(전체)",
            )

        def test_parens_are_part_of_label_not_stripped(self):
            self.assertEqual(
                normalize_label("기타경영사항(자율공시)"),
                "기타경영사항(자율공시)",
            )

        def test_unrelated_report_returns_none(self):
            self.assertIsNone(normalize_label("사업보고서(2025.12)"))

    unittest.main()
