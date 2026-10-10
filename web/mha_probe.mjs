import * as ort from 'onnxruntime-node';
const model = process.argv[2], S = +process.argv[3], scaleIn = +process.argv[4];
let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 2 * scaleIn;
const mk = () => new ort.Tensor('float32', Float32Array.from({ length: S * 1280 }, rnd), [1, S, 1280]);
const feeds = { q: mk(), k: mk(), v: new ort.Tensor('float32', Float32Array.from({ length: S * 1280 }, () => rnd() / scaleIn), [1, S, 1280]) };
// reference in JS (float64)
const H = 16, D = 80, q = feeds.q.data, k = feeds.k.data, v = feeds.v.data, ref = new Float64Array(S * 1280);
for (let h = 0; h < H; h++) for (let i = 0; i < S; i++) {
  const sc = new Float64Array(S); let mx = -Infinity;
  for (let j = 0; j < S; j++) { let s = 0; for (let d = 0; d < D; d++) s += q[i * 1280 + h * D + d] * k[j * 1280 + h * D + d]; sc[j] = s / Math.sqrt(D); mx = Math.max(mx, sc[j]); }
  let z = 0; for (let j = 0; j < S; j++) { sc[j] = Math.exp(sc[j] - mx); z += sc[j]; }
  for (let d = 0; d < D; d++) { let o = 0; for (let j = 0; j < S; j++) o += sc[j] * v[j * 1280 + h * D + d]; ref[i * 1280 + h * D + d] = o / z; }
}
for (const ep of ['cpu', 'webgpu']) {
  const y = (await (await ort.InferenceSession.create(model, { executionProviders: [ep], logSeverityLevel: 3 })).run(feeds)).y.data;
  let m = 0, r = 0; for (let i = 0; i < y.length; i++) { m = Math.max(m, Math.abs(y[i] - ref[i])); r = Math.max(r, Math.abs(ref[i])); }
  console.log(`S=${S} logit-scale=${scaleIn} ${ep.padEnd(6)} max|err|=${m.toExponential(2)} (max|y|=${r.toFixed(3)})`);
}
