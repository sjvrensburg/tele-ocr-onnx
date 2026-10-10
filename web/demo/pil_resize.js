// Bit-exact port of Pillow's bicubic resampling (libImaging/Resample.c, 8-bit path) and of
// Qwen2-VL's `smart_resize`, so browser preprocessing matches the Python reference.
//
// Why: in browsers transformers.js resizes through canvas drawImage, which ignores the requested
// filter and does not antialias like PIL. TeleOCR is sensitive to this (it changes OCR output), so
// the demo resizes images itself to the exact size the processor wants; the processor's own resize
// then becomes a no-op.

const PRECISION_BITS = 32 - 8 - 2;
const SCALE = 2 ** PRECISION_BITS;

function bicubic(x) {
  const a = -0.5;
  x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

// Integer fixed-point coefficients exactly as precompute_coeffs + normalize_coeffs_8bpc.
function coeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 2.0 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Float64Array(outSize * ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    const k = new Float64Array(xmax);
    let ww = 0;
    for (let x = 0; x < xmax; x++) { const w = bicubic((x + xmin - center + 0.5) * ss); k[x] = w; ww += w; }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * SCALE) : Math.trunc(0.5 + v * SCALE);
    }
    bounds[xx * 2] = xmin; bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, kk, ksize };
}

const clip8 = (ss) => { const v = Math.floor(ss / SCALE); return v < 0 ? 0 : v > 255 ? 255 : v; };

/** Resize interleaved uint8 pixels [h, w, c] like PIL `Image.resize((w2, h2), Image.BICUBIC)`. */
export function resizeBicubicPIL(src, w, h, c, w2, h2) {
  let cur = src, cw = w;
  if (w2 !== w) {  // horizontal pass first (as ImagingResampleInner does)
    const { bounds, kk, ksize } = coeffs(w, w2);
    const out = new Uint8ClampedArray(w2 * h * c);
    for (let y = 0; y < h; y++) {
      for (let xx = 0; xx < w2; xx++) {
        const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1], ko = xx * ksize;
        for (let ch = 0; ch < c; ch++) {
          let ss = 2 ** (PRECISION_BITS - 1);
          for (let x = 0; x < xmax; x++) ss += cur[(y * cw + x + xmin) * c + ch] * kk[ko + x];
          out[(y * w2 + xx) * c + ch] = clip8(ss);
        }
      }
    }
    cur = out; cw = w2;
  }
  if (h2 !== h) {
    const { bounds, kk, ksize } = coeffs(h, h2);
    const out = new Uint8ClampedArray(cw * h2 * c);
    for (let yy = 0; yy < h2; yy++) {
      const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1], ko = yy * ksize;
      for (let xx = 0; xx < cw; xx++) {
        for (let ch = 0; ch < c; ch++) {
          let ss = 2 ** (PRECISION_BITS - 1);
          for (let y = 0; y < ymax; y++) ss += cur[((y + ymin) * cw + xx) * c + ch] * kk[ko + y];
          out[(yy * cw + xx) * c + ch] = clip8(ss);
        }
      }
    }
    cur = out;
  }
  return cur;
}

// Python's round(): half to even.
function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** transformers' Qwen2-VL smart_resize: target (height, width), multiples of `factor`. */
export function smartResize(height, width, factor = 28, minPixels = 56 * 56, maxPixels = 14 * 14 * 4 * 1280) {
  let hBar = pyRound(height / factor) * factor;
  let wBar = pyRound(width / factor) * factor;
  if (hBar * wBar > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    hBar = Math.max(factor, Math.floor(height / beta / factor) * factor);
    wBar = Math.max(factor, Math.floor(width / beta / factor) * factor);
  } else if (hBar * wBar < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    hBar = Math.ceil((height * beta) / factor) * factor;
    wBar = Math.ceil((width * beta) / factor) * factor;
  }
  return [hBar, wBar];
}
