import { STATIC, DECISION } from './config.js';
import { featurize, createMlp } from './handMlp.js';
import { decide, averageProbabilities, createDuplicateGuard } from './decision.js';

/**
 * Mesin pose statis: tiap frame -> probabilitas -> rata-rata jendela pendek -> label dikonfirmasi bila stabil.
 * Satu isyarat = satu hasil; setelah dikonfirmasi, tangan harus turun dulu sebelum isyarat berikutnya.
 * Event: {type:'state', state}, {type:'result', status:'confirmed'|'uncertain', label, confidence, margin, candidates, duplicate, activeSec}.
 */
export function createStaticEngine({ model, onEvent = () => {}, decision = {}, config = STATIC }) {
  const predict = createMlp(model);
  const classNames = model.classes;
  const guard = createDuplicateGuard(config.duplicateSuppressSec);
  let state = 'idle', startT = 0, lastHandT = 0, window = [], done = false;
  let cand = -1, candSince = 0, best = null, lastT = 0;

  const setState = s => { if (s !== state) { state = s; onEvent({ type: 'state', state: s }); } };
  const resetSeg = () => { window = []; done = false; cand = -1; best = null; };
  const cfg = () => ({ ...DECISION, ...decision });

  function emit(status, d, t, dup) {
    onEvent({ type: 'result', ...d, status, duplicate: dup, reason: status === 'confirmed' ? 'held' : 'ended', activeSec: t - startT });
  }

  return {
    push({ t, hands, width, height }) {
      lastT = t;
      if (!hands || hands.length === 0) {
        if (state === 'recording' && t - lastHandT >= config.endGapSec) {
          if (!done && lastHandT - startT >= config.minActiveSec && best) emit('uncertain', { ...best, status: 'uncertain' }, lastHandT, false);
          resetSeg(); setState('idle');
        }
        return;
      }
      lastHandT = t;
      if (state === 'idle') { resetSeg(); startT = t; setState('recording'); }
      window.push(predict(featurize(hands, width, height)));
      if (window.length > config.windowFrames) window.shift();
      if (done || window.length < config.minFrames) return;
      const d = decide(averageProbabilities(window), classNames, cfg());
      if (!best || d.confidence > best.confidence) best = d;
      const idx = d.status === 'confirmed' ? d.index : -1;
      if (idx !== cand) { cand = idx; candSince = t; }
      if (cand >= 0 && t - candSince >= config.holdSec) {
        const dup = guard.isDuplicate(d.label, startT);
        if (!dup) guard.record(d.label, t);
        emit('confirmed', d, t, dup); done = true;
      }
    },
    get state() { return state; },
    get recordedSec() { return state === 'recording' ? lastT - startT : 0; },
    setDecision(next) { Object.assign(decision, next); },
    reset() { resetSeg(); guard.reset(); setState('idle'); }
  };
}
