export const meta = {
  name: 'hani-eval-full',
  description: '한의 국가시험 517문항 × Claude 3티어(Opus/Sonnet/Haiku) 평가',
  phases: [
    { title: 'eval:opus' },
    { title: 'eval:sonnet' },
    { title: 'eval:haiku' },
    { title: 'retry' },
  ],
}

// 설정값은 args 플러밍 이슈를 피하려 스크립트에 직접 명시.
const path = '/Users/jeonghamin/Desktop/국가데이터처 모니터링/hani-exam-llm/eval/questions_noanswer.jsonl'
const total = 517
const batch = 13
const models = ['opus', 'sonnet', 'haiku']

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    answers: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          idx: { type: 'integer' },
          pred: { type: 'integer', minimum: 1, maximum: 5 },
        },
        required: ['idx', 'pred'],
      },
    },
  },
  required: ['answers'],
}

function promptFor(lo, hi) {
  const count = hi - lo + 1
  return `당신은 대한민국 한의사·한약사 국가시험을 보는 수험생입니다. 폐쇄형(closed-book) 시험입니다. 웹검색이나 다른 도구를 쓰지 말고, 오직 당신의 의학/한의학 지식으로 푸세요.

Read 도구로 아래 파일을 여세요. offset=${lo + 1}, limit=${count} 로 읽으면 idx ${lo}..${hi} 문항이 나옵니다. 이 파일만 읽고, 정답표 등 다른 파일은 찾지 마세요.
파일 경로: ${path}

각 줄은 JSON 문항이며 idx, question, options(보기 5개 배열)를 가집니다. options[0]=보기1 ... options[4]=보기5.
각 문항마다 가장 옳은 보기 하나를 1~5 중에서 고르세요. 확신이 없어도 반드시 1~5 중 하나를 고르세요.
idx ${lo}..${hi} 의 모든 ${count}개 문항에 빠짐없이 답하세요.

내부적으로 충분히 추론하되, 최종 출력은 각 문항의 {idx, pred} 목록입니다.`
}

const batches = []
for (let s = 0; s < total; s += batch) batches.push([s, Math.min(s + batch, total) - 1])

log(`평가 시작: ${models.length}개 모델 × ${total}문항, 배치 ${batch} → 모델당 ${batches.length}배치`)

const tasks = []
for (const m of models) for (const [lo, hi] of batches) tasks.push({ m, lo, hi })

const raw = await parallel(tasks.map((t) => () =>
  agent(promptFor(t.lo, t.hi), {
    label: `${t.m} ${t.lo}-${t.hi}`,
    phase: `eval:${t.m}`,
    model: t.m,
    schema: SCHEMA,
  }).then((r) => ({ m: t.m, lo: t.lo, hi: t.hi, answers: (r && r.answers) || [] }))
    .catch((e) => ({ m: t.m, lo: t.lo, hi: t.hi, answers: [], error: String(e) }))
))

const preds = {}
for (const m of models) preds[m] = {}
for (const r of raw) {
  if (!r) continue
  for (const a of r.answers) {
    if (a && Number.isInteger(a.idx) && a.idx >= 0 && a.idx < total && a.pred >= 1 && a.pred <= 5) {
      preds[r.m][a.idx] = a.pred
    }
  }
}

phase('retry')
for (const m of models) {
  const missing = []
  for (let i = 0; i < total; i++) if (preds[m][i] == null) missing.push(i)
  if (!missing.length) { log(`${m}: 미응답 없음`); continue }
  log(`${m}: 미응답 ${missing.length}개 재시도`)
  const groups = []
  for (let i = 0; i < missing.length; i += 5) groups.push(missing.slice(i, i + 5))
  const retried = await parallel(groups.map((g) => () => {
    const lo = g[0], hi = g[g.length - 1]
    return agent(
      promptFor(lo, hi) + `\n\n(반드시 idx ${JSON.stringify(g)} 에 모두 답하세요.)`,
      { label: `retry ${m} ${lo}-${hi}`, phase: 'retry', model: m, schema: SCHEMA }
    ).then((r) => (r && r.answers) || []).catch(() => [])
  }))
  for (const arr of retried) for (const a of arr) {
    if (a && Number.isInteger(a.idx) && a.idx >= 0 && a.idx < total && a.pred >= 1 && a.pred <= 5) {
      preds[m][a.idx] = a.pred
    }
  }
}

const stats = {}
for (const m of models) {
  let answered = 0
  for (let i = 0; i < total; i++) if (preds[m][i] != null) answered++
  stats[m] = { answered, missing: total - answered }
  log(`${m}: 응답 ${answered}/${total}`)
}

return { preds, stats, total, batches: batches.length, errors: raw.filter((r) => r && r.error).length }
