// Uji: fitur & probabilitas JS == referensi Python; mesin mengonfirmasi pose stabil dan tidak mengonfirmasi saat tangan hilang.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { featurize, createMlp, validateMlpModel } from '../src/lib/recognition/handMlp.js';
import { createStaticEngine } from '../src/lib/recognition/staticEngine.js';

const fx = JSON.parse(fs.readFileSync(new URL('./fixtures/handmlp.json', import.meta.url), 'utf8'));
const toLm = hands => hands.map(h => h.map(([x, y, z]) => ({ x, y, z })));
const predict = createMlp(fx.model);
const validFixtureModel = { ...fx.model, activation: 'relu', output: 'softmax' };
assert.equal(validateMlpModel(validFixtureModel), validFixtureModel);
assert.throws(() => validateMlpModel({ ...validFixtureModel, feature_dim: 195 }), /feature_dim/);
assert.throws(() => validateMlpModel({ ...validFixtureModel, classes: ['only-one'] }), /Daftar kelas/);
let maxF = 0, maxP = 0;
for (const c of fx.cases) {
  const f = featurize(toLm(c.hands), c.width, c.height);
  f.forEach((v, i) => { maxF = Math.max(maxF, Math.abs(v - c.features[i])); });
  const p = predict(f);
  p.forEach((v, i) => { maxP = Math.max(maxP, Math.abs(v - c.probs[i])); });
}
console.log(`maks selisih fitur=${maxF.toExponential(2)} probabilitas=${maxP.toExponential(2)}`);
assert.ok(maxF < 1e-5, 'fitur JS berbeda dari Python');
assert.ok(maxP < 1e-5, 'probabilitas JS berbeda dari Python');

// invarian rasio gambar: skala x,y berbeda tidak mengubah fitur bila landmark dikali ukuran
const c0 = fx.cases[0];
const a = featurize(toLm(c0.hands), 1280, 720);
const sq = c0.hands.map(h => h.map(([x, y, z]) => [x * 1280 / 720, y, z * 1280 / 720]));
const b = featurize(toLm(sq), 720, 720);
a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-9, 'fitur bergantung rasio gambar'));

// mesin
const events = [];
const eng = createStaticEngine({ model: fx.model, onEvent: e => events.push(e), decision: { minConfidence: 0.6, minMargin: 0.3 } });
const sample = fx.cases[1];
let t = 0;
for (let i = 0; i < 30; i++) { eng.push({ t: (t += 1 / 30), hands: toLm(sample.hands), width: 1280, height: 720 }); }
for (let i = 0; i < 30; i++) { eng.push({ t: (t += 1 / 30), hands: [], width: 1280, height: 720 }); }
const res = events.filter(e => e.type === 'result');
assert.equal(res.length, 1, 'harus tepat satu hasil per isyarat');
assert.equal(res[0].status, 'confirmed');
assert.equal(res[0].label, fx.model.classes[sample.label]);
assert.deepEqual(events.filter(e => e.type === 'state').map(e => e.state), ['recording', 'idle']);

// tangan hanya sekilas (< minActiveSec) tidak boleh menghasilkan apa-apa
events.length = 0; eng.reset(); events.length = 0;
for (let i = 0; i < 3; i++) eng.push({ t: (t += 1 / 30), hands: toLm(sample.hands), width: 1280, height: 720 });
for (let i = 0; i < 30; i++) eng.push({ t: (t += 1 / 30), hands: [], width: 1280, height: 720 });
assert.equal(events.filter(e => e.type === 'result').length, 0, 'sekilas tidak boleh dilaporkan');
console.log('handmlp.test: OK');
