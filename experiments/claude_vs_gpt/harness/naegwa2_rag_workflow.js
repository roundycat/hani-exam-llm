export const meta = {
  name: 'naegwa2-rag',
  description: '내과학2 30문항 RAG(웹 한의학 레퍼런스 검색) + 변증추론 자기일관성 k=3',
  phases: [{ title: 'RAG', detail: 'Opus 에이전트가 웹검색 근거로 풀이' }],
}

const FILE = '/Users/jeonghamin/Desktop/국가데이터처 모니터링/hani-exam-llm/eval/naegwa2_noanswer.jsonl'
const NIDXS = []
for (let n = 66; n <= 95; n++) NIDXS.push(n)
const K = 3

const SCHEMA = {
  type: 'object', additionalProperties: false, required: ['nidx', 'pred', 'evidence'],
  properties: {
    nidx: { type: 'integer' },
    pred: { type: 'integer', minimum: 1, maximum: 5 },
    evidence: { type: 'string', description: '검색으로 찾은 핵심 근거 요약(출처 포함, 2-4줄)' },
  },
}

const prompt = (nidx, s) => `너는 한방내과 전문의다. 한의사 국가시험 내과학 5지선다를 푼다.

파일 \`${FILE}\` 에서 nidx==${nidx} 인 줄을 Read로 찾아 question과 options(보기 5개)를 읽어라(정답 없음).

반드시 **WebSearch로 한의학 레퍼런스를 조사**하라(샘플 ${s + 1}):
- 보기에 등장하는 처방(方劑)·본초의 구성·주치(主治)·적응증
- 문제 증상에 해당하는 변증(辨證)·치법(治法), 필요시 『傷寒論』『東醫寶鑑』 등 원전 조문
신뢰할 만한 한의학 자료를 우선하고, 2~4회 검색하라.

조사한 근거로 변증추론(주소증→팔강·장부변증→치법→보기별 처방 비교→정답)을 거쳐
정답 보기번호(1~5)를 확정하라. {nidx, pred, evidence}로 출력하라.`

phase('RAG')
const votes = {} // nidx -> [pred...]
const evid = {}  // nidx -> [evidence...]
for (let s = 0; s < K; s++) {
  const round = await parallel(NIDXS.map(n => () =>
    agent(prompt(n, s), { label: `rag-s${s + 1}-${n}`, phase: 'RAG', model: 'opus', schema: SCHEMA })
  ))
  for (const r of round) {
    if (!r || r.nidx == null || r.pred == null) continue
    ;(votes[r.nidx] = votes[r.nidx] || []).push(r.pred)
    ;(evid[r.nidx] = evid[r.nidx] || []).push(r.evidence || '')
  }
  log(`RAG 샘플 ${s + 1}/${K} 완료`)
}

return { votes, evidence: evid, meta: { K, n: NIDXS.length } }
