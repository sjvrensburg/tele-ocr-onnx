// Drives the demo page in real Google Chrome (headless, WebGPU enabled) and runs OCR samples.
//   node browser_test.mjs [--dtype q4f16] [--samples text table formula layout] [--headful]
import fs from 'node:fs';
import puppeteer from 'puppeteer-core';

const args = { dtype: ['q4f16'], samples: ['text', 'table', 'formula', 'layout'], url: ['http://127.0.0.1:8765/web/demo/index.html'], out: null };
{ let k = null; for (const a of process.argv.slice(2)) if (a.startsWith('--')) { k = a.slice(2); args[k] = []; } else args[k].push(a); }
const golden = JSON.parse(fs.readFileSync('../golden/summary.json', 'utf8'));
const FILES = { text: 'text.png', table: 'table.png', formula: 'formula.png', layout: 'layout.jpg', code: 'code.png', figure: 'scientific_figure.png' };

function levenshtein(a, b) {
  const prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let diag = prev[0]; prev[0] = i;
    for (let j = 1; j <= b.length; j++) { const t = prev[j]; prev[j] = Math.min(prev[j] + 1, prev[j - 1] + 1, diag + (a[i - 1] === b[j - 1] ? 0 : 1)); diag = t; }
  }
  return prev[b.length];
}

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome',
  headless: !('headful' in args),
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPUDeveloperFeatures', '--use-angle=vulkan', '--ignore-gpu-blocklist', '--no-first-run'],
  protocolTimeout: 1_800_000,
});
const page = await browser.newPage();
page.on('console', (m) => { const t = m.text(); if (/error|warn/i.test(m.type()) && !/powerPreference|DevTools/.test(t)) console.log(`[console.${m.type()}]`, t.slice(0, 300)); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(args.url[0], { waitUntil: 'load' });
const info = await page.evaluate(async () => {
  if (!navigator.gpu) return { webgpu: false };
  const a = await navigator.gpu.requestAdapter();
  const i = a.info ?? {};
  return { webgpu: true, vendor: i.vendor, arch: i.architecture, description: i.description, f16: a.features.has('shader-f16'),
           maxBuffer: a.limits.maxBufferSize, maxStorageBinding: a.limits.maxStorageBufferBindingSize, maxStorageBuffersPerStage: a.limits.maxStorageBuffersPerShaderStage };
});
console.log('browser:', await browser.version(), JSON.stringify(info));
if (!info.webgpu) { await browser.close(); process.exit(2); }

const results = {};
for (const name of args.samples) {
  const r = await page.evaluate(async (name, file, dtype) => {
    window.teleocr.setImage('/models/TeleOCR/assets/' + file);
    document.getElementById('task').value = name; document.getElementById('task').onchange();
    try { return await window.teleocr.run({ task: name, dtype, maxNew: 2048 }); } catch (e) { return { error: String(e?.message ?? e) }; }
  }, name, FILES[name], args.dtype[0]);
  if (r.error) { console.log(`== ${name}: ERROR ${r.error}`); results[name] = r; continue; }
  // PyTorch golden outputs were capped at 512 new tokens; compare on the common prefix in that case.
  const g = golden[name];
  let cmp = '';
  let exact = null, ned = null;
  if (g) {
    const ref = g.text.trim();
    const hyp = g.new_tokens >= 512 ? r.result.slice(0, ref.length) : r.result;
    exact = hyp === ref;
    ned = levenshtein(hyp, ref) / Math.max(hyp.length, ref.length, 1);
    cmp = `  vs PyTorch: exact=${exact} NED=${ned.toFixed(4)}${g.new_tokens >= 512 ? ' (prefix)' : ''}`;
  }
  console.log(`== ${name}: ${JSON.stringify(r.stats)}${cmp}\n   ${r.result.slice(0, 160).replaceAll('\n', ' | ')}`);
  results[name] = { ...r, exact, ned };
}
if (args.out) fs.writeFileSync(args.out[0], JSON.stringify({ info, dtype: args.dtype[0], results }, null, 1));
await page.screenshot({ path: '../results/browser_demo.png', fullPage: true });
await browser.close();
