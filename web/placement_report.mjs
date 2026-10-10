// Lists nodes that ORT does NOT place on the WebGPU EP (they fall back to CPU) for each model.
//   node placement_report.mjs ../output/onnx/*.onnx
import { spawnSync } from 'node:child_process';
if (process.argv[2] === '--child') {
  const ort = await import('onnxruntime-node');
  await ort.InferenceSession.create(process.argv[3], { executionProviders: ['webgpu'], logSeverityLevel: 0, logVerbosityLevel: 0 });
  process.exit(0);
}
for (const model of process.argv.slice(2)) {
  const r = spawnSync(process.execPath, [new URL(import.meta.url).pathname, '--child', model], { encoding: 'utf8', maxBuffer: 1 << 30 });
  const lines = (r.stdout + r.stderr).split('\n');  // ORT logs to stderr
  const i = lines.findIndex((l) => l.includes('Node placements'));
  const cpu = [];
  let inCpu = false;
  for (const l of lines.slice(i + 1)) {
    if (!l.includes('VerifyEachNodeIsAssignedToAnEp')) break;
    const body = l.split('VerifyEachNodeIsAssignedToAnEp]')[1] ?? '';
    if (/placed on \[/.test(body)) { inCpu = body.includes('[CPUExecutionProvider]'); continue; }
    if (inCpu && body.trim()) cpu.push(body.trim());
  }
  const ops = {};
  for (const c of cpu) { const op = c.split(' ')[0]; ops[op] = (ops[op] ?? 0) + 1; }
  console.log(`${model.split('/').at(-1).padEnd(36)} CPU nodes: ${cpu.length}  ${JSON.stringify(ops)}`);
}
