// End-to-end OCR through the real transformers.js pipeline (processor, chat template, MRoPE indexing,
// generation loop) on the WebGPU EP, compared with the PyTorch greedy outputs in golden/summary.json.
//
//   node e2e_transformers.mjs [--model ../output | user/repo] [--dtype vision=fp16,embed=q4f16,decoder=q4f16] [--samples text table]
//                             [--max-new 512] [--out results.json]
import fs from 'node:fs';
import path from 'node:path';
import { AutoProcessor, Qwen2_5_VLForConditionalGeneration, RawImage, env } from '@huggingface/transformers';

const args = { dtype: ['vision=fp16,embed=q4f16,decoder=q4f16'], samples: null, 'max-new': ['512'], model: ['../output'], out: null, device: ['webgpu'] };
{
  let key = null;
  for (const a of process.argv.slice(2)) if (a.startsWith('--')) { key = a.slice(2); args[key] = []; } else args[key].push(a);
}
const SAMPLES = {
  text: ['assets/text.png', 'Please output the text content from the image.', null],
  formula: ['assets/formula.png', 'Please write out the expression of the formula in the image using LaTeX format.', null],
  table: ['assets/table.png', 'This is the image of a table. Please output the table in OTSL format.', null],
  layout: ['assets/layout.jpg', 'Analyze the image layout.', [1036, 1036]],
};
const golden = JSON.parse(fs.readFileSync('../golden/summary.json', 'utf8'));
const samples = args.samples ?? Object.keys(SAMPLES);

// --model is a local folder, or a Hub repo id (private repos: set HF_TOKEN).
let modelId = args.model[0];
if (fs.existsSync(modelId)) {
  const modelDir = path.resolve(modelId);
  env.localModelPath = path.dirname(modelDir) + '/';
  env.allowRemoteModels = false;
  modelId = path.basename(modelDir);
} else {
  env.allowLocalModels = false;
  env.cacheDir = path.resolve('../build/hf-js-cache');
}

const parts = Object.fromEntries(args.dtype[0].split(',').map((kv) => kv.split('=')));
const dtype = { vision_encoder: parts.vision, embed_tokens: parts.embed, decoder_model_merged: parts.decoder };

function levenshtein(a, b) {
  const prev = new Array(b.length + 1).fill(0).map((_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let diag = prev[0]; prev[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = prev[j];
      prev[j] = Math.min(prev[j] + 1, prev[j - 1] + 1, diag + (a[i - 1] === b[j - 1] ? 0 : 1));
      diag = tmp;
    }
  }
  return prev[b.length];
}

let t0 = performance.now();
const processor = await AutoProcessor.from_pretrained(modelId);
const model = await Qwen2_5_VLForConditionalGeneration.from_pretrained(modelId, { device: args.device[0], dtype });
console.log(`loaded ${modelId} dtype=${JSON.stringify(dtype)} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

const results = {};
for (const name of samples) {
  const [file, prompt, resize] = SAMPLES[name];
  let image = await RawImage.read(path.join('../models/TeleOCR', file));
  image = image.rgb();
  if (resize) image = await image.resize(resize[0], resize[1], { resample: 3 /* bicubic, as in the README */ });
  const messages = [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: [{ type: 'image' }, { type: 'text', text: prompt }] },
  ];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  const inputs = await processor(text, image);
  const g = golden[name];
  const maxNew = Math.min(+args['max-new'][0], g.new_tokens);
  t0 = performance.now();
  const out = await model.generate({ ...inputs, max_new_tokens: maxNew, do_sample: false, repetition_penalty: 1.0 });
  const secs = (performance.now() - t0) / 1000;
  const newIds = out.slice(null, [inputs.input_ids.dims[1], null]);
  const decoded = processor.batch_decode(newIds, { skip_special_tokens: true })[0].trim();
  const ref = g.text.trim();
  const refCut = maxNew < g.new_tokens ? ref.slice(0, decoded.length) : ref;
  const ned = levenshtein(decoded, refCut) / Math.max(decoded.length, refCut.length, 1);
  const grid = Array.from(inputs.image_grid_thw.data, Number).join('x');
  const nNew = newIds.dims[1];
  results[name] = { grid, prompt_len: inputs.input_ids.dims[1], new_tokens: nNew, seconds: secs, ned, exact: decoded === refCut, text: decoded };
  console.log(`\n== ${name} grid=${grid} (golden ${g.grid_thw.flat().join('x')}) prompt_len=${inputs.input_ids.dims[1]} (golden ${g.prompt_len}) ` +
    `${nNew} tokens in ${secs.toFixed(1)}s (${(nNew / secs).toFixed(1)} tok/s)\n   exact_match=${decoded === refCut} normalized_edit_distance=${ned.toFixed(4)}`);
  console.log('   ' + decoded.slice(0, 300).replaceAll('\n', '\n   '));
}
if (args.out) fs.writeFileSync(args.out[0], JSON.stringify({ dtype, results }, null, 1));
await model.dispose();
