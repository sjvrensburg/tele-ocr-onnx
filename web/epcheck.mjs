// Usage: node epcheck.mjs model.onnx [ep]  -- fails if any node can't run on the EP
import * as ort from 'onnxruntime-node';
const [path, ep='webgpu'] = process.argv.slice(2);
try {
  const s = await ort.InferenceSession.create(path, { executionProviders: [ep], extra: { session: { disable_cpu_ep_fallback: '1' } } , logSeverityLevel: 3});
  console.log('OK', path, ep, s.inputNames, s.outputNames);
} catch (e) { console.log('FAIL', path, ep, String(e.message ?? e).slice(0, 600)); }
