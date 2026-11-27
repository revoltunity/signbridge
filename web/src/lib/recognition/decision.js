import { DECISION } from './config.js';

/** Ambil kandidat teratas dari probabilitas. */
export function topCandidates(probs, classNames, k = 3) {
  const idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, k);
  return idx.map(i => ({ index: i, label: classNames[i], probability: probs[i] }));
}

/** Rata-rata probabilitas dari beberapa hipotesis kecepatan (test-time augmentation). */
export function averageProbabilities(list) {
  const out = new Float32Array(list[0].length);
  for (const p of list) for (let i = 0; i < out.length; i++) out[i] += p[i] / list.length;
  return out;
}

/**
 * Keputusan: "confirmed" hanya bila keyakinan dan selisih dengan kandidat kedua melewati ambang.
 * Selain itu "uncertain" (kandidat tetap dilaporkan, tetapi tidak boleh dipakai sebagai terjemahan).
 */
export function decide(probs, classNames, options = {}) {
  const cfg = { ...DECISION, ...options };
  const top = topCandidates(probs, classNames, 3);
  const margin = top[0].probability - (top[1]?.probability ?? 0);
  const confirmed = top[0].probability >= cfg.minConfidence && margin >= cfg.minMargin;
  return { status: confirmed ? 'confirmed' : 'uncertain', label: top[0].label, index: top[0].index,
           confidence: top[0].probability, margin, candidates: top };
}

/** Penekan duplikat: label yang sama dalam jendela waktu singkat setelah isyarat sebelumnya diabaikan. */
export function createDuplicateGuard(windowSec = DECISION.duplicateSuppressSec) {
  let last = null;
  return {
    reset() { last = null; },
    isDuplicate(label, t) { return !!last && last.label === label && t - last.t < windowSec; },
    record(label, t) { last = { label, t }; }
  };
}
