export const meta = {
  name: 'hani-eval-smoke',
  description: '한의 시험 평가 하니스 스모크 테스트 (Opus, 13문항)',
  phases: [{ title: 'smoke' }],
}

const path = args.path
const lo = 0, hi = 12
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

const prompt = `당신은 대한민국 한의사·한약사 국가시험을 보는 수험생입니다. 폐쇄형(closed-book) 시험입니다. 웹검색이나 다른 도구를 쓰지 말고, 오직 당신의 의학/한의학 지식으로 푸세요.

Read 도구로 아래 파일을 여세요. offset=${lo + 1}, limit=${hi - lo + 1} 로 읽으면 idx ${lo}..${hi} 문항이 나옵니다. 이 파일만 읽고, 정답표 같은 다른 파일은 찾지 마세요.
파일 경로: ${path}

각 줄은 JSON 문항이며 idx, question, options(보기 5개 배열)를 가집니다. options[0]=보기1 ... options[4]=보기5.
각 문항에 대해 가장 옳은 보기 하나를 1~5 중에서 고르세요. 확신이 없어도 반드시 1~5 중 하나를 고르세요.
idx ${lo}..${hi} 의 모든 ${hi - lo + 1}개 문항에 답하세요.

내부적으로 충분히 추론하되, 최종 출력은 각 문항의 {idx, pred} 목록입니다.`

phase('smoke')
log('Opus 스모크 테스트: idx 0..12 읽고 답하기')
const r = await agent(prompt, { label: 'smoke opus 0-12', phase: 'smoke', model: 'opus', schema: SCHEMA })
return r
