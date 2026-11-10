#!/bin/bash
# 저장소에 포함된 라운드별 결과(src/Checkpoint/Results/EXP/{H12,S1,CAUSE})로 분석만 재실행한다(GPU 불필요, 수 분).
# 출력은 /tmp 에 쓰고 저장소의 결과 JSON과 같은지 비교한다.
set -e; cd "$(dirname "$0")/../src"; O=${TMPDIR:-/tmp}/fedmse_repro; mkdir -p $O
( cd ../analysis && sha256sum -c PRE_RULES_2.sha256 PRE_RULES_3.sha256 PRE_RULES_3_ADD1.sha256 )   # 동결 규칙이 그대로인지
python ../analysis/analyze_h1h2.py --exp H12 --runs 0-9 --out $O/h1h2_result.json > /dev/null
python ../analysis/analyze_cause0.py --out $O/cause0_result.json > /dev/null
python ../analysis/analyze_cause.py --exp CAUSE --out $O/cause_result.json > /dev/null
python - <<PY
import json
def same(x, y):
    if isinstance(x, dict): return x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, list): return len(x) == len(y) and all(same(u, v) for u, v in zip(x, y))
    if isinstance(x, float) or isinstance(y, float): return abs(x - y) < 1e-9
    return x == y
for n in ("h1h2_result", "cause0_result", "cause_result"):
    a = json.load(open("../analysis/%s.json" % n)); b = json.load(open("$O/%s.json" % n))
    print(n, "일치(허용오차 1e-9)" if same(a, b) else "다름(수치 확인 필요)")
PY
