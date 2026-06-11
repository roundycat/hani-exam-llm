export const meta = {
  name: 'hani-paper-results',
  description: '한의 국가시험 LLM 평가 논문 결과 섹션을 다중 에이전트로 작성·검증',
  phases: [
    { title: 'Draft', detail: '섹션별 병렬 초안' },
    { title: 'Verify', detail: '수치/주장 대조 검증' },
    { title: 'Synthesize', detail: '최종 통합' },
  ],
}

// ── 단일 진실원: 모든 수치는 여기서만 인용 (LLM이 새 숫자 만들지 말 것) ──
const FACTS = `
[연구 개요]
- 목적: 한의사·한약사 국가시험 5지선다 객관식을 LLM이 푸는 능력 평가(closed-book) + 소형 오픈모델 파인튜닝 학습 전후 비교.
- 데이터셋: 국시원 공개 기출(제81회 한의사 2026, 제27회 한약사 2026)에서 추출. 전체 587문항 중 그림 의존 70문항 제외 → 순수 텍스트 517문항(한의사 276, 한약사 241). 정답 100% 매칭, 보기 5개 검증.
- 누수 방지: 모델에는 정답 제거 문항만 제공, 정답(gold)은 분리 보관하여 채점에만 사용. gold↔원본 정답 불일치 0. 파인튜닝은 시드42로 학습466/검증51 분할(겹침 없음).
- 채점: 예측 보기번호(1~5)==정답. 95% CI는 Wilson 구간. 랜덤 베이스라인 20%.

[평가 대상 6모델 — 전체 517문항]
- GPT-5 84.91% (439/517), CI 81.6–87.7%
- Claude Sonnet 83.75% (433/517), CI 80.3–86.7%
- Claude Opus 81.24% (420/517), CI 77.6–84.4%
- GPT-4o(2024-08-06) 77.95% (403/517), CI 74.2–81.3%
- Claude Haiku 71.95% (372/517), CI 67.9–75.6%
- GPT-4o-mini 59.57% (308/517), CI 55.3–63.7%
- 상위 3개(GPT-5/Sonnet/Opus) 95% CI 상호 중첩 → 통계적 우열 단정 불가(사실상 동급 ~81–85%).
- 회차별: 모든 모델이 한의사(81회)를 한약사(27회)보다 약 10%p 어려워함. 예) GPT-5 한약사 89.6% vs 한의사 80.8%.

[과목별(6모델 평균) — 난이도]
- 최난도: 내과학2 44%(GPT-5조차 57%). 그 다음 외과학 61%(n=6), 보건의약관계법규 70%.
- 최易: 한의학기초 92%(n=107), 한방생리학 86%, 예방의학 83%.
- 약한 모델(GPT-4o-mini)은 지식형 과목에서 급락(본초학 31%, 침구학·부인과학 48%).

[방법 — 모델별 실행]
- Claude(Opus/Sonnet/Haiku): Claude Code 워크플로로 13문항×40배치×3티어=120 에이전트, 폐쇄형, 스키마 강제 출력, 미응답 0/오류 0.
- GPT(4o-mini/4o/5): 레포 원본 evaluate.py 무수정, 문항당 1 API 호출, temperature=0(비추론형), gpt-5는 reasoning_effort=low. gpt-5 빈응답 5개는 토큰 상향 재요청으로 복구(3개 정답) 후 최종 84.91%.
- 독립 재계산으로 전 모델 정답률 교차검증 통과.

[파인튜닝 학습 전후 — Qwen2.5-7B-Instruct, 검증 51문항(held-out)]
- 학습 전(베이스): 51.0% (26/51), CI 37.7–64.1%, 출력토큰 2.00/문항, 지연 0.386s.
- 학습 후(LoRA SFT, 466문항·3에폭): 49.0% (25/51), CI 35.9–62.3%, 출력토큰 2.00/문항, 지연 0.378s.
- 변화: 정답률 −2.0%p(맞음 −1), 출력토큰 0, 지연 거의 0. 두 CI 크게 중첩 → 유의한 개선 없음(사실상 차이 없음/노이즈).
- 학습 중 검증 손실 0.318→0.328→0.340으로 상승(과적합 신호).
- 해석: (1) 466개 소규모 객관식 SFT는 도메인 지식을 주입하지 못하고 형식만 학습. (2) "번호만 출력"은 베이스도 이미 수행 → 연산량(토큰) 절감 여지 없음. (3) 따라서 기대했던 정확도 향상·연산량 감소는 관찰되지 않음(정직한 음성 결과).

[인프라 메모(재현 관련)]
- OpenAI 파인튜닝 플랫폼은 종료(winding down)되어 GPT 파인튜닝 불가 → 오픈모델(Qwen)로 전환.
- Together에서 Qwen2.5-7B는 서버리스 불가, 추론에 2×H100 전용 엔드포인트 필요(고유 서빙명으로 호출).

[한계]
- 프로토콜 차이: Claude·GPT-5는 내부 추론 허용, GPT-4o/mini·Qwen은 번호만 즉답. 완전 동일 조건 아님.
- 파인튜닝 검증셋 n=51로 작아 과목별·전후 비교의 통계적 검정력 낮음(CI 넓음).
- 데이터는 비전 전사 기반이라 드물게 한자 오탈자 가능.
- 단일 회차(한의사81/한약사27)만 포함.
`

phase('Draft')
const SECTIONS = [
  { key: 'abstract', instr: '논문 "결과" 도입부로 쓸 5~7문장 초록 겸 핵심 요약. 6모델 벤치마크 핵심 순위와 파인튜닝 음성 결과를 균형 있게.' },
  { key: 'methods', instr: '방법 섹션. 데이터셋(517 텍스트 문항, 그림 제외), 누수 방지, 채점/CI, 모델별 실행 방식(Claude 워크플로/GPT evaluate.py/Qwen LoRA), 인프라 전환 사유를 학술적으로 서술. 8~14문장.' },
  { key: 'results_main', instr: '주요 결과 서술(표1·표2 해석). 모델 순위, 상위3 CI 중첩, 한의사>한약사 난이도, 과목별 난이도(내과학2 최난도, 한의학기초 최易), 약한 모델의 지식형 과목 붕괴. 표는 본문에 다시 적지 말고 "표1/표2 참조"로만. 8~14문장.' },
  { key: 'finetune', instr: '파인튜닝 학습 전후 분석(표3·표4). 51.0%→49.0%로 유의한 향상 없음, 토큰·지연 불변, 검증손실 상승(과적합), 왜 향상·연산절감이 없었는지(소규모 SFT는 형식만 학습/베이스가 이미 간결) 정직하게. 과대해석 금지. 6~10문장.' },
  { key: 'limitations', instr: '한계 및 타당성 위협. 프로토콜 차이, n=51 검정력, 전사 오탈자, 단일 회차. 5~8문장.' },
]
const drafts = await pipeline(
  SECTIONS,
  (s) => agent(
    `너는 의료 AI 논문을 쓰는 연구자다. 아래 FACTS의 수치만 사용해(새 숫자 절대 금지) "${s.key}" 섹션을 한국어 학술체로 작성하라. 표는 본문에 재작성하지 말 것. 정직하고 과장 없이.\n\n=== FACTS ===\n${FACTS}\n\n섹션: ${s.instr}`,
    { label: `draft:${s.key}`, phase: 'Draft' }
  ).then(text => ({ key: s.key, text })),
  // Verify 단계: 같은 아이템을 바로 검증으로 흘림
  (draft) => agent(
    `아래 섹션 초안의 모든 수치·주장을 FACTS와 대조 검증하라. FACTS에 없는 숫자나 과대주장(예: 파인튜닝이 향상됨, 통계적으로 유의함)을 찾아 교정한 \"최종본\"을 출력하라. 문제 없으면 원문 유지. 출력은 교정된 섹션 본문만.\n\n=== FACTS ===\n${FACTS}\n\n=== 섹션(${draft.key}) 초안 ===\n${draft.text}`,
    { label: `verify:${draft.key}`, phase: 'Verify', schema: {
        type: 'object', additionalProperties: false,
        required: ['key', 'final', 'issues_found'],
        properties: {
          key: { type: 'string' },
          final: { type: 'string', description: '교정된 최종 섹션 본문' },
          issues_found: { type: 'array', items: { type: 'string' }, description: '발견·교정한 문제 목록(없으면 빈 배열)' },
        }
      } }
  ).then(v => ({ key: draft.key, final: v.final, issues: v.issues_found || [] }))
)

phase('Synthesize')
const verified = drafts.filter(Boolean)
const byKey = {}
for (const d of verified) byKey[d.key] = d.final
const allIssues = verified.flatMap(d => (d.issues || []).map(i => `[${d.key}] ${i}`))
log(`검증 완료: ${verified.length}개 섹션, 교정 이슈 ${allIssues.length}건`)

return {
  sections: byKey,
  issues: allIssues,
}
