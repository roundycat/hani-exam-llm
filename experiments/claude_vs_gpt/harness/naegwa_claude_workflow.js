export const meta = {
  name: 'naegwa-claude-sc',
  description: '내과학 96문항 Claude Opus+Sonnet 변증추론 자기일관성(k=5)',
  phases: [
    { title: 'opus', detail: 'Opus k=5 자기일관성' },
    { title: 'sonnet', detail: 'Sonnet k=5 자기일관성' },
  ],
}

const FILE = 'eval/naegwa_noanswer.jsonl' // nidx 0..95, fields: nidx/과목/question/options (정답 없음)
const N = 96, BATCH = 12, K = 5
const ranges = []
for (let s = 0; s < N; s += BATCH) ranges.push([s, Math.min(s + BATCH, N) - 1])

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['answers'],
  properties: {
    answers: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false, required: ['nidx', 'pred'],
        properties: { nidx: { type: 'integer' }, pred: { type: 'integer', minimum: 1, maximum: 5 } },
      },
    },
  },
}

const prompt = (lo, hi, sample) => `너는 한방내과 전문의이자 한의사·한약사 국가시험 풀이 전문가다.
${FILE} 파일을 Read로 열어라(JSONL, 각 줄 필드: nidx, 과목, question, options[5개]). 정답은 들어있지 않다.
그 중 **nidx ${lo}부터 ${hi}까지의 문항만** 풀어라(다른 nidx는 무시).

각 문항을 반드시 다음 한의학적 절차로 추론하라(샘플 ${sample}):
(1) 주소증·동반증·설진/맥진 등 핵심 단서 정리
(2) 팔강변증(표리·한열·허실·음양)·장부변증
(3) 치법(治法) 결정
(4) 각 보기(처방/본초/개념)를 변증·치법에 비추어 적합성 비교
(5) 가장 부합하는 보기 1개 선택
지식만 사용하라. 웹검색·외부도구로 답을 찾지 말 것.

각 문항의 최종 정답 보기번호(1~5)를 {nidx, pred}로 모아 출력하라. 배정 범위(${lo}~${hi}) 전부 포함.`

// 모델별 k라운드 자기일관성: 라운드마다 8배치 병렬, nidx->votes 누적
async function selfConsistency(model) {
  const votes = {} // nidx -> [pred...]
  for (let s = 0; s < K; s++) {
    const rounds = await parallel(ranges.map(([lo, hi]) => () =>
      agent(prompt(lo, hi, s + 1), { label: `${model}-s${s + 1}-${lo}`, phase: model, model, schema: SCHEMA })
    ))
    for (const r of rounds) {
      if (!r || !r.answers) continue
      for (const a of r.answers) {
        if (a.nidx == null || a.pred == null) continue
        ;(votes[a.nidx] = votes[a.nidx] || []).push(a.pred)
      }
    }
    log(`${model}: 샘플 ${s + 1}/${K} 완료`)
  }
  return votes
}

phase('opus')
const opus = await selfConsistency('opus')
phase('sonnet')
const sonnet = await selfConsistency('sonnet')

return { opus, sonnet, meta: { N, K, BATCH } }
