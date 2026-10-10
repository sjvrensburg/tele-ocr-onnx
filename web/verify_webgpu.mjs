// Numerical check of the exported ONNX components on the WebGPU EP (onnxruntime-node, Dawn).
//
//   node verify_webgpu.mjs [--vision "" _fp16] [--decoder "" _fp16 _q4f16] [--samples text table]
//                          [--max-new 64] [--onnx ../output/onnx]
//
// Vision: compares image_features with the PyTorch golden features (cosine per merged token).
// Decoder: feeds the exact fp32 golden inputs_embeds (text + image features), compares prefill logits
// with PyTorch, then greedy-decodes (embedding new tokens with the matching embed_tokens variant) and
// compares the generated ids with PyTorch's greedy output.
import fs from 'node:fs';
import path from 'node:path';
import * as ort from 'onnxruntime-node';

const args = { dump: null, vision: [], decoder: [], samples: null, 'max-new': ['64'], onnx: ['../output/onnx'], golden: ['../golden/bin'], ep: ['webgpu'] };
{
  let key = null;
  for (const a of process.argv.slice(2)) {
    if (a.startsWith('--')) { key = a.slice(2); args[key] = []; } else args[key].push(a === '""' || a === 'fp32' ? '' : a);
  }
}
const ONNX = args.onnx[0], GOLD = args.golden[0], EP = args.ep[0], MAX_NEW = +args['max-new'][0];
const meta = JSON.parse(fs.readFileSync(path.join(GOLD, 'meta.json'), 'utf8'));
const samples = args.samples ?? Object.keys(meta);
const EOS = new Set([151645, 151643]);

const TYPED = { float32: Float32Array, int64: BigInt64Array, int32: Int32Array };
function load(sample, key) {
  const { dtype, shape } = meta[sample][key];
  const buf = fs.readFileSync(path.join(GOLD, `${sample}.${key}.bin`));
  const C = TYPED[dtype];
  return { data: new C(buf.buffer, buf.byteOffset, buf.byteLength / C.BYTES_PER_ELEMENT), dims: shape, dtype };
}
const tensor = (t) => new ort.Tensor(t.dtype, t.data, t.dims);

function half2float(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}
function toF32(t) {
  if (t.type === 'float32') return t.data;
  if (t.type === 'float16') {
    if (typeof Float16Array !== 'undefined' && t.data instanceof Float16Array) return Float32Array.from(t.data);
    return Float32Array.from(t.data, half2float);
  }
  throw new Error(`unexpected ${t.type}`);
}
function cosine(a, b, off = 0, n = a.length) {
  let ab = 0, aa = 0, bb = 0;
  for (let i = off; i < off + n; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
  return ab / Math.sqrt(aa * bb);
}
function argmax(a, off = 0, n = a.length) {
  let best = off;
  for (let i = off; i < off + n; i++) if (a[i] > a[best]) best = i;
  return best - off;
}
const opts = { executionProviders: [EP], logSeverityLevel: 3, graphOptimizationLevel: 'all' };
const fmt = (x, d = 6) => Number(x).toFixed(d);
let failures = 0;

for (const vsfx of args.vision) {
  const s = await ort.InferenceSession.create(path.join(ONNX, `vision_encoder${vsfx}.onnx`), opts);
  for (const sample of samples) {
    const pv = load(sample, 'pixel_values'), grid = load(sample, 'image_grid_thw'), gold = load(sample, 'image_features');
    const feeds = { pixel_values: tensor(pv), image_grid_thw: tensor(grid) };
    let t0 = performance.now();
    let out = (await s.run(feeds)).image_features;
    const first = performance.now() - t0;
    t0 = performance.now();
    out = (await s.run(feeds)).image_features;
    const second = performance.now() - t0;
    const o = toF32(out), D = gold.dims[1], rows = gold.dims[0];
    if (args.dump) fs.writeFileSync(path.join(args.dump[0], `vision${vsfx}.${sample}.f32`), Buffer.from(o.buffer, o.byteOffset, o.byteLength));
    let minc = 1, sumc = 0;
    for (let r = 0; r < rows; r++) { const c = cosine(o, gold.data, r * D, D); minc = Math.min(minc, c); sumc += c; }
    // Calibrated on the authors' own bf16 PyTorch inference vs fp32 (min cos 0.876, mean >= 0.9989 on these samples).
    const ok = out.dims[0] === rows && minc > 0.85 && sumc / rows > 0.998;
    failures += !ok;
    console.log(`${ok ? 'OK ' : 'BAD'} vision${vsfx || '(fp32)'} ${sample.padEnd(8)} grid=${Array.from(grid.data).join('x')} feats=${rows} ` +
      `min_cos=${fmt(minc)} mean_cos=${fmt(sumc / rows)}  ${first.toFixed(0)}ms first / ${second.toFixed(0)}ms warm`);
  }
  await s.release();
}

for (const dsfx of args.decoder) {
  const dec = await ort.InferenceSession.create(path.join(ONNX, `decoder_model_merged${dsfx}.onnx`), opts);
  const emb = await ort.InferenceSession.create(path.join(ONNX, `embed_tokens${dsfx}.onnx`), opts);
  const kvType = dsfx === '' ? 'float32' : 'float16';
  const nLayers = dec.inputNames.filter((n) => n.endsWith('.key')).length;
  for (const sample of samples) {
    const embeds = load(sample, 'inputs_embeds'), pos = load(sample, 'position_ids');
    const goldLogits = load(sample, 'prefill_logits_last'), goldIds = Array.from(load(sample, 'generated_ids').data, Number);
    const S = embeds.dims[1];
    const kvEmpty = () => new ort.Tensor(kvType, kvType === 'float32' ? new Float32Array(0) : new Uint16Array(0), [1, 8, 0, 128]);
    let feeds = { inputs_embeds: tensor(embeds), attention_mask: new ort.Tensor('int64', new BigInt64Array(S).fill(1n), [1, S]), position_ids: tensor(pos) };
    for (let i = 0; i < nLayers; i++) { feeds[`past_key_values.${i}.key`] = kvEmpty(); feeds[`past_key_values.${i}.value`] = kvEmpty(); }
    let t0 = performance.now();
    let out = await dec.run(feeds);
    const prefillMs = performance.now() - t0;
    let logits = toF32(out.logits);
    const V = goldLogits.dims[0], last = logits.length - V;
    const c = cosine(logits, goldLogits.data, 0, V) && cosine(logits.subarray(last), goldLogits.data);
    const top = argmax(logits, last, V), goldTop = argmax(goldLogits.data);

    // greedy decode
    let nextPos = 0n;
    for (const p of pos.data) if (p > nextPos) nextPos = p;
    nextPos += 1n;
    const gen = [top];
    t0 = performance.now();
    while (gen.length < Math.min(MAX_NEW, goldIds.length) && !EOS.has(gen.at(-1))) {
      const e = (await emb.run({ input_ids: new ort.Tensor('int64', BigInt64Array.from([BigInt(gen.at(-1))]), [1, 1]) })).inputs_embeds;
      const T = S + gen.length;
      const p = nextPos + BigInt(gen.length - 1);
      feeds = { inputs_embeds: e, attention_mask: new ort.Tensor('int64', new BigInt64Array(T).fill(1n), [1, T]), position_ids: new ort.Tensor('int64', BigInt64Array.from([p, p, p]), [3, 1, 1]) };
      for (let i = 0; i < nLayers; i++) { feeds[`past_key_values.${i}.key`] = out[`present.${i}.key`]; feeds[`past_key_values.${i}.value`] = out[`present.${i}.value`]; }
      out = await dec.run(feeds);
      logits = toF32(out.logits);
      gen.push(argmax(logits, logits.length - V, V));
    }
    const decodeMs = (performance.now() - t0) / Math.max(1, gen.length - 1);
    let match = 0;
    while (match < gen.length && gen[match] === goldIds[match]) match++;
    const ok = top === goldTop && match >= Math.min(gen.length, 16);
    failures += !ok;
    console.log(`${ok ? 'OK ' : 'BAD'} decoder${dsfx || '(fp32)'} ${sample.padEnd(8)} S=${S} prefill_cos=${fmt(c)} argmax ${top === goldTop ? 'match' : `MISMATCH ${top} vs ${goldTop}`} ` +
      `greedy prefix match ${match}/${gen.length} (golden len ${goldIds.length})  prefill ${prefillMs.toFixed(0)}ms, ${decodeMs.toFixed(1)}ms/token`);
  }
  await dec.release(); await emb.release();
}
process.exit(failures ? 1 : 0);
