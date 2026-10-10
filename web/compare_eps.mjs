// node compare_eps.mjs model.onnx name:dtype:d1xd2[:kind] ...   -- runs CPU and WebGPU EPs, prints max|diff| per output
import * as ort from 'onnxruntime-node';
const [model, ...specs] = process.argv.slice(2);
let seed = 1; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const feeds = {};
for (const sp of specs) {
  const [name, dtype, dimstr, kind = 'randn'] = sp.split(':');
  const dims = dimstr.split('x').map(Number), n = dims.reduce((a, b) => a * b, 1);
  let data;
  if (dtype === 'float32') data = Float32Array.from({ length: n }, (_, i) => kind === 'mask' ? (rnd() < 0.3 ? -1e4 : 0) : kind === 'angle' ? rnd() * 6 : (rnd() - 0.5) * 2);
  else if (dtype === 'int32') data = Int32Array.from({ length: n }, () => Math.floor(rnd() * +kind));
  else if (dtype === 'int64') data = BigInt64Array.from({ length: n }, (_, i) => kind === 'arange' ? BigInt(i % dims.at(-1)) : BigInt(Math.floor(rnd() * +kind)));
  feeds[name] = new ort.Tensor(dtype, data, dims);
}
const out = {};
for (const ep of ['cpu', 'webgpu']) out[ep] = await (await ort.InferenceSession.create(model, { executionProviders: [ep], logSeverityLevel: 3 })).run(feeds);
for (const k of Object.keys(out.cpu)) {
  const a = out.cpu[k].data, b = out.webgpu[k].data; let m = 0, mi = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m || Number.isNaN(d)) { m = d; mi = i; } }
  console.log(model.split('/').at(-1).padEnd(20), k, 'max|diff|', m.toExponential(2), 'at', mi, 'of', a.length);
}
