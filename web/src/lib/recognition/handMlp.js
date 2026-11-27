// Fitur landmark tangan + inferensi MLP. HARUS identik dengan ml/landmark_mlp/common.py dan train.py.
export const FEATURE_DIM = 131;

/**
 * hands: array landmark MediaPipe (tiap tangan: 21 titik {x,y,z} ternormalisasi 0..1).
 * width/height: ukuran frame video (koordinat dikali ukuran agar bebas rasio gambar, sama seperti extract.py).
 */
export function featurize(hands, width, height) {
  const f = new Float64Array(FEATURE_DIM);
  const hs = (hands || []).slice(0, 2).map(h => h.map(p => [p.x * width, p.y * height, p.z * width]));
  hs.sort((a, b) => a[0][0] - b[0][0]);
  const scales = [];
  hs.forEach((h, i) => {
    const w = h[0];
    const s = Math.hypot(h[9][0] - w[0], h[9][1] - w[1]) || 1e-6;
    scales.push(s);
    for (let j = 0; j < 21; j++) for (let k = 0; k < 3; k++) f[i * 63 + j * 3 + k] = (h[j][k] - w[k]) / s;
    f[126 + i] = 1;
  });
  if (hs.length === 2) for (let k = 0; k < 3; k++) f[128 + k] = (hs[1][0][k] - hs[0][0][k]) / scales[0];
  return f;
}

export function createMlp(model) {
  const { mean, std, layers } = model;
  return function predict(features) {
    let h = Float64Array.from(features, (v, i) => (v - mean[i]) / std[i]);
    layers.forEach((layer, li) => {
      const out = Float64Array.from(layer.b);
      for (let i = 0; i < h.length; i++) {
        const hi = h[i];
        if (hi === 0) continue;
        const row = layer.W[i];
        for (let j = 0; j < out.length; j++) out[j] += hi * row[j];
      }
      if (li < layers.length - 1) for (let j = 0; j < out.length; j++) if (out[j] < 0) out[j] = 0;
      h = out;
    });
    let max = -Infinity; for (const v of h) if (v > max) max = v;
    let sum = 0; const p = new Float64Array(h.length);
    for (let i = 0; i < h.length; i++) { p[i] = Math.exp(h[i] - max); sum += p[i]; }
    for (let i = 0; i < p.length; i++) p[i] /= sum;
    return p;
  };
}

export function validateMlpModel(model) {
  if (!model || model.feature_dim !== FEATURE_DIM) throw new Error(`Model harus memiliki feature_dim=${FEATURE_DIM}.`);
  if (!Array.isArray(model.classes) || model.classes.length < 2 || model.classes.some(x => typeof x !== 'string' || !x.trim())) {
    throw new Error('Daftar kelas model tidak valid.');
  }
  if (!Array.isArray(model.mean) || model.mean.length !== FEATURE_DIM || !Array.isArray(model.std) || model.std.length !== FEATURE_DIM) {
    throw new Error('Mean/std model harus berisi 131 angka.');
  }
  if (![...model.mean, ...model.std].every(Number.isFinite) || model.std.some(x => x <= 0)) {
    throw new Error('Mean/std model mengandung nilai tidak valid.');
  }
  if (!Array.isArray(model.layers) || model.layers.length < 2) throw new Error('Struktur layer model tidak valid.');
  let inputSize = FEATURE_DIM;
  for (const [i, layer] of model.layers.entries()) {
    if (!Array.isArray(layer.W) || !Array.isArray(layer.b) || layer.W.length !== inputSize || !layer.b.length) {
      throw new Error(`Dimensi layer ${i} tidak valid.`);
    }
    const outputSize = layer.b.length;
    if (layer.W.some(row => !Array.isArray(row) || row.length !== outputSize || !row.every(Number.isFinite)) || !layer.b.every(Number.isFinite)) {
      throw new Error(`Bobot layer ${i} tidak valid.`);
    }
    inputSize = outputSize;
  }
  if (inputSize !== model.classes.length) throw new Error('Jumlah output model tidak sama dengan jumlah kelas.');
  if (model.activation !== 'relu' || model.output !== 'softmax') throw new Error('Aktivasi model tidak didukung web.');
  return model;
}

export async function loadMlpModel(base = '/') {
  const root = base.endsWith('/') ? base : `${base}/`;
  const res = await fetch(`${root}models/hand_mlp.json`, { cache: 'no-cache' });
  if (!res.ok) throw new Error('Model 25 kata belum tersedia. Jalankan train.py dan pastikan hand_mlp.json ada di web/public/models/.');
  const model = validateMlpModel(await res.json());
  if (model.classes.length !== 25) throw new Error(`Model harus berisi 25 kelas kata, ditemukan ${model.classes.length}.`);
  if (new Set(model.classes).size !== 25) throw new Error('Nama kelas model harus unik.');
  return model;
}
