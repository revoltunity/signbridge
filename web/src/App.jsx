import React, { useEffect, useRef, useState } from 'react';
import { Camera, Mic, MicOff, Volume2, X } from 'lucide-react';
import { isSupabaseConfigured } from './lib/supabase';
import { createSpeechRecognition, speakIndonesian } from './lib/speech';
import { initializeHandExtractor, extractHands, closeHandExtractor } from './lib/recognition/handExtractor';
import { loadMlpModel } from './lib/recognition/handMlp';
import { createStaticEngine } from './lib/recognition/staticEngine';
import { SENSITIVITY_PRESETS } from './lib/recognition/config';

/* Ilustrasi digambar tangan: garis sengaja tidak sempurna. */
function Scribble({ className = '' }) {
  return (
    <svg className={`scribble ${className}`} viewBox="0 0 200 12" preserveAspectRatio="none" aria-hidden="true" focusable="false">
      <path d="M2 7 C30 2 55 10 90 6 C125 2 160 9 198 4" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function HandDrawing({ className = '' }) {
  return (
    <svg className={`hand-drawing ${className}`} viewBox="0 0 240 270" aria-hidden="true" focusable="false">
      <g fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M93 150 C91 120 88 84 90 58 C91 44 109 42 111 57 C113 82 114 112 115 140" />
        <path d="M117 138 C116 104 115 64 118 36 C120 20 140 20 142 36 C145 68 144 104 144 140" />
        <path d="M146 142 C148 114 150 82 154 58 C157 43 175 46 174 62 C173 90 171 120 171 154" />
        <path d="M173 160 C180 136 186 118 193 102 C199 89 214 96 210 110 C204 132 197 160 192 184" />
        <path d="M88 196 C74 180 56 160 44 142 C34 127 46 112 58 122 C70 134 84 152 96 166" />
        <path d="M89 200 C92 224 96 244 99 262" />
        <path d="M192 184 C188 214 180 240 176 262" />
      </g>
      <g fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" opacity=".5">
        <path d="M104 150 C121 160 150 160 172 150" />
        <path d="M100 184 C124 196 160 196 186 182" />
      </g>
      <path className="hand-ground" d="M30 262 C70 254 110 266 150 258 C180 252 205 262 224 256" fill="none" strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}

const HAND_LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

function drawHands(canvas, video, hands) {
  const ctx = canvas.getContext('2d');
  const width = video.videoWidth || canvas.clientWidth;
  const height = video.videoHeight || canvas.clientHeight;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  const baseLine = Math.max(2, width / 450);
  const baseDot = Math.max(3, width / 180);
  const outline = Math.max(1.5, width / 700);
  hands.forEach(hand => {
    // Dua lapis: putih tipis di bawah, warna aksen di atas, supaya terlihat di latar apa pun.
    [['#ffffff', outline], ['#D9482B', 0]].forEach(([color, extra]) => {
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = baseLine + extra * 2;
      HAND_LINKS.forEach(([a, b]) => {
        if (!hand[a] || !hand[b]) return;
        ctx.beginPath();
        ctx.moveTo(hand[a].x * width, hand[a].y * height);
        ctx.lineTo(hand[b].x * width, hand[b].y * height);
        ctx.stroke();
      });
      hand.forEach(point => {
        ctx.beginPath();
        ctx.arc(point.x * width, point.y * height, baseDot + extra, 0, Math.PI * 2);
        ctx.fill();
      });
    });
  });
}

const PHASE_TEXT = {
  idle: 'Menunggu isyarat. Angkat tangan ke depan kamera.',
  recording: 'Membaca pose tangan… tahan sebentar, lalu turunkan tangan.',
  release: 'Isyarat terlalu panjang. Turunkan tangan sebelum isyarat berikutnya.',
  classifying: 'Mengenali isyarat…'
};

export default function App() {
  const [handCount, setHandCount] = useState(0);
  const [detectorState, setDetectorState] = useState('off');   // off | loading | ready | error
  const [modelState, setModelState] = useState('off');         // off | loading | ready | error
  const [detectorNote, setDetectorNote] = useState('');
  const [modelNote, setModelNote] = useState('');
  const [phase, setPhase] = useState('idle');
  const [recordedSec, setRecordedSec] = useState(0);
  const [lastResult, setLastResult] = useState(null);
  const [confirmed, setConfirmed] = useState([]);
  const [sensitivity, setSensitivity] = useState('seimbang');
  const [inferMs, setInferMs] = useState(0);
  const [mode, setMode] = useState('text');
  const [input, setInput] = useState('');
  const [listening, setListening] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [status, setStatus] = useState('Siap menerjemahkan');
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const recognitionRef = useRef(null);
  const engineRef = useRef(null);
  const sensitivityRef = useRef('seimbang');

  const output = confirmed.join(' ');
  const recognizerReady = detectorState === 'ready' && modelState === 'ready';

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    recognitionRef.current?.abort?.();
    closeHandExtractor();
    window.speechSynthesis?.cancel();
  }, []);

  useEffect(() => {
    sensitivityRef.current = sensitivity;
    const preset = SENSITIVITY_PRESETS[sensitivity];
    engineRef.current?.setDecision({ minConfidence: preset.minConfidence, minMargin: preset.minMargin });
  }, [sensitivity]);

  useEffect(() => {
    if (!cameraOn) return undefined;
    let cancelled = false;
    let handle = null;
    let usingRvfc = false;
    const video = videoRef.current;

    const run = async () => {
      if (!video || !streamRef.current) return;
      video.srcObject = streamRef.current;
      try { await video.play(); } catch { /* browser dapat menunda autoplay */ }

      setDetectorState('loading'); setModelState('loading');
      setStatus('Memuat detektor tangan dan model kata…');
      const [det, mdl] = await Promise.allSettled([initializeHandExtractor(), loadMlpModel(import.meta.env?.BASE_URL ?? '/')]);
      if (cancelled) return;
      const detResult = det.status === 'fulfilled' ? det.value : { status: 'error', message: det.reason?.message };
      if (detResult.status !== 'ready') {
        setDetectorState('error'); setDetectorNote(detResult.message || 'gagal');
      } else { setDetectorState('ready'); setDetectorNote(detResult.delegate === 'CPU' ? 'CPU' : 'GPU'); }
      if (mdl.status !== 'fulfilled') {
        setModelState('error'); setModelNote(mdl.reason?.message || 'gagal');
      } else { setModelState('ready'); setModelNote(`${mdl.value.classes.length} kata`); }
      if (detResult.status !== 'ready' || mdl.status !== 'fulfilled') {
        setStatus('Komponen pengenalan belum siap. Lihat kondisi sistem di bawah.');
        return;
      }

      const preset = SENSITIVITY_PRESETS[sensitivityRef.current];
      const engine = createStaticEngine({
        model: mdl.value,
        decision: { minConfidence: preset.minConfidence, minMargin: preset.minMargin },
        onEvent: event => {
          if (cancelled) return;
          if (event.type === 'state') setPhase(event.state);
          else if (event.type === 'classifying') setPhase('classifying');
          else if (event.type === 'result') {
            setLastResult(event);
            if (event.status === 'confirmed' && !event.duplicate) {
              setConfirmed(list => [...list.slice(-19), event.label]);
              setStatus(`Isyarat dikenali: ${event.label}`);
            } else if (event.status === 'confirmed') setStatus(`Duplikat diabaikan: ${event.label}`);
            else setStatus('Isyarat belum dikenali dengan yakin. Coba ulangi.');
          } else if (event.type === 'error') { setPhase('idle'); setStatus(`Klasifikasi gagal: ${event.message}`); }
          else if (event.type === 'dropped') setStatus(event.message);
        }
      });
      engineRef.current = engine;
      setStatus('Siap. Lakukan satu isyarat di depan kamera, lalu turunkan tangan.');


      let lastT = -1;
      let lastUi = 0;
      let lastVideoTime = -1;
      const onFrame = (nowMs, metadata) => {
        if (cancelled) return;
        if (!usingRvfc) {
          // Cadangan: lewati bila frame video belum berganti.
          if (video.currentTime === lastVideoTime) { handle = requestAnimationFrame(onFrame); return; }
          lastVideoTime = video.currentTime;
        }
        const result = extractHands(video, nowMs);
        if (result.status === 'ok') {
          let t = (metadata?.captureTime ?? nowMs) / 1000;
          if (t <= lastT) t = lastT + 0.001;
          lastT = t;
          const canvas = canvasRef.current;
          if (canvas) drawHands(canvas, video, result.hands);
          engine.push({ t, hands: result.hands, width: result.width, height: result.height });
          if (nowMs - lastUi > 150) {
            lastUi = nowMs;
            setHandCount(result.handCount);
            setRecordedSec(engine.state === 'recording' ? engine.recordedSec : 0);
          }
        }
        handle = usingRvfc ? video.requestVideoFrameCallback(onFrame) : requestAnimationFrame(onFrame);
      };
      usingRvfc = typeof video.requestVideoFrameCallback === 'function';
      handle = usingRvfc ? video.requestVideoFrameCallback(onFrame) : requestAnimationFrame(onFrame);
    };
    run();

    return () => {
      cancelled = true;
      if (handle !== null) {
        if (usingRvfc) video?.cancelVideoFrameCallback?.(handle); else cancelAnimationFrame(handle);
      }
      engineRef.current?.reset();
      engineRef.current = null;
    };
  }, [cameraOn]);

  const startVoiceInput = () => {
    const recognition = createSpeechRecognition();
    if (!recognition) {
      setStatus('Browser ini belum mendukung Speech Recognition. Coba Chrome atau Edge.');
      return;
    }
    recognitionRef.current = recognition;
    recognition.onstart = () => { setListening(true); setStatus('Mendengarkan suara…'); };
    recognition.onresult = event => {
      const transcript = Array.from(event.results).map(result => result[0].transcript).join('');
      setInput(transcript);
    };
    recognition.onerror = () => { setListening(false); setStatus('Mikrofon tidak dapat digunakan. Periksa izin browser.'); };
    recognition.onend = () => { setListening(false); setStatus('Input suara selesai'); };
    recognition.start();
  };

  const stopVoiceInput = () => recognitionRef.current?.stop();

  const startCamera = async () => {
    if (cameraOn) return;
    setCameraError('');
    setHandCount(0);
    setLastResult(null);
    setConfirmed([]);
    setPhase('idle');
    setDetectorState('off'); setModelState('off');
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Akses kamera memerlukan HTTPS atau localhost.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
        audio: false
      });
      streamRef.current = stream;
      setCameraOn(true);
      setStatus('Kamera aktif. Memuat model…');
    } catch (error) {
      const message = error?.name === 'NotAllowedError'
        ? 'Izin kamera ditolak. Izinkan akses kamera di browser.'
        : error?.name === 'NotFoundError'
          ? 'Perangkat kamera tidak ditemukan.'
          : 'Kamera tidak dapat diakses. Periksa izin dan perangkat kamera.';
      setCameraError(message);
      setStatus('Kamera belum aktif.');
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    setCameraOn(false);
    setHandCount(0);
    setPhase('idle');
    setRecordedSec(0);
    setDetectorState('off'); setModelState('off');
    setStatus('Kamera dimatikan.');
    closeHandExtractor();
  };

  const translateText = () => {
    if (!input.trim()) {
      setStatus('Masukkan teks atau gunakan mikrofon terlebih dahulu.');
      return;
    }
    setStatus('Input teks ke animasi isyarat belum tersedia. Pengenalan kamera berjalan untuk 25 kata.');
  };

  const readOutput = () => {
    if (!output) return;
    if (!speakIndonesian(output)) setStatus('Fitur suara tidak tersedia di browser ini.');
  };

  const cameraState = cameraError ? 'Bermasalah' : cameraOn ? 'Menyala' : 'Mati';
  const stateText = (s, note) => ({ off: 'Belum dimuat', loading: 'Sedang dimuat', ready: note ? `Siap · ${note}` : 'Siap', error: `Gagal${note ? ` (${note})` : ''}` }[s]);
  const dot = ok => (ok === true ? 'dot dot-ok' : ok === 'wait' ? 'dot dot-wait' : 'dot dot-off');
  const dotFor = s => (s === 'ready' ? true : s === 'loading' ? 'wait' : false);
  const phaseText = phase === 'recording' ? `${PHASE_TEXT.recording} (${recordedSec.toFixed(1)} dtk)` : PHASE_TEXT[phase];

  return (
    <main className="app-shell">
      <header className="masthead">
        <h1 className="wordmark">
          signbridge
          <Scribble />
        </h1>
        <p className="tagline">Tempat teman dengar dan teman Tuli berbincang di satu layar.</p>
      </header>

      <div className="side-switch" role="group" aria-label="Pilih sisi percakapan">
        <button type="button" className="switch-btn" aria-pressed={mode === 'text'} onClick={() => setMode('text')}>Teman dengar</button>
        <button type="button" className="switch-btn" aria-pressed={mode === 'camera'} onClick={() => setMode('camera')}>Teman Tuli</button>
      </div>

      <div className="talk" data-side={mode}>
        <section className="side side-hearing" aria-labelledby="hearing-title">
          <div className="side-head">
            <h2 id="hearing-title">Teman dengar</h2>
            <Scribble />
            <p>Ketik atau bicara.</p>
          </div>

          <div className="sheet">
            <label className="sr-only" htmlFor="source-text">Tulis kalimat dalam Bahasa Indonesia</label>
            <textarea
              id="source-text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Tulis sesuatu untuk temanmu…"
              rows={4}
            />
            <div className="sheet-foot">
              <span className="count">{input.length} karakter</span>
              <div className="actions">
                <button
                  type="button"
                  className={listening ? 'btn btn-ghost recording' : 'btn btn-ghost'}
                  onClick={listening ? stopVoiceInput : startVoiceInput}
                  aria-pressed={listening}
                  aria-label={listening ? 'Hentikan mikrofon' : 'Mulai bicara lewat mikrofon'}
                >
                  {listening ? <MicOff size={20} aria-hidden="true" /> : <Mic size={20} aria-hidden="true" />}
                  {listening ? 'Berhenti' : 'Bicara'}
                </button>
                <button type="button" className="btn btn-primary" onClick={translateText}>Terjemahkan</button>
              </div>
            </div>
          </div>

          <div className="sign-note">
            <HandDrawing className="sign-hand" />
            <div>
              <h3>Untuk teman Tuli</h3>
              <p>Animasi isyarat belum tersedia. Kami hanya menampilkan isyarat yang sudah diverifikasi.</p>
            </div>
          </div>
        </section>

        <section className="side side-deaf" aria-labelledby="deaf-title">
          <div className="side-head">
            <h2 id="deaf-title">Teman Tuli</h2>
            <Scribble />
            <p>Tunjukkan isyarat ke kamera.</p>
          </div>

          <div className="camera-stage">
            {cameraOn
              ? <><video ref={videoRef} autoPlay muted playsInline aria-label="Pratinjau kamera" /><canvas ref={canvasRef} className="landmark-overlay" aria-hidden="true" /></>
              : (
                <div className="camera-empty">
                  <HandDrawing className="empty-hand" />
                  <span>Kamera belum menyala</span>
                </div>
              )}
            <div className="camera-chip">
              {cameraOn
                ? (recognizerReady ? `${phase === 'recording' ? 'Merekam' : 'Siap'} · ${handCount} tangan` : 'Kamera menyala · memuat model')
                : 'Kamera mati'}
            </div>
          </div>

          <div className="camera-controls">
            {cameraOn
              ? <button type="button" className="btn btn-ghost" onClick={stopCamera}><X size={20} aria-hidden="true" /> Matikan kamera</button>
              : <button type="button" className="btn btn-primary" onClick={startCamera}><Camera size={20} aria-hidden="true" /> Nyalakan kamera</button>}
            <p className="privacy">Gambar kamera diproses di peramban ini dan tidak dikirim ke server. Model deteksi tangan diunduh dari internet saat pertama dimuat.</p>
          </div>
          {cameraError && <p className="camera-error" role="alert">{cameraError}</p>}

          <div className="caption">
            <h3>Untuk teman dengar</h3>
            {recognizerReady && <p className="caption-note" role="status">{phaseText}</p>}
            {lastResult?.status === 'confirmed' && !lastResult.duplicate
              ? <><p className="caption-text">{lastResult.label}</p><p className="caption-note">Keyakinan model: {(lastResult.confidence * 100).toFixed(1)}% · selisih kandidat ke-2: {(lastResult.margin * 100).toFixed(1)}%</p></>
              : lastResult
                ? <><p className="caption-text caption-quiet">{lastResult.status === 'confirmed' ? `${lastResult.label} (duplikat diabaikan)` : 'Belum yakin'}</p>
                    <p className="caption-note">Kandidat (belum terkonfirmasi): {lastResult.candidates.map(c => `${c.label} ${(c.probability * 100).toFixed(0)}%`).join(' · ')}</p></>
                : <p className="caption-text caption-quiet">Belum ada teks.</p>}
            {confirmed.length > 0 && <p className="caption-note">Isyarat terkonfirmasi: {confirmed.join(' · ')}</p>}
            <p className="caption-note">Setiap pose dikenali sebagai satu kata dari 25 kelas dataset; ini bukan terjemahan kalimat. Akurasi ada di laporan train.py; pemisahan grup foto masih memakai heuristik nomor file dan belum membuktikan sesi terpisah.</p>
            <div className="actions">
              <label className="caption-note" htmlFor="sensitivity">Ambang keyakinan:&nbsp;</label>
              <select id="sensitivity" value={sensitivity} onChange={e => setSensitivity(e.target.value)}>
                {Object.entries(SENSITIVITY_PRESETS).map(([key, p]) => <option key={key} value={key}>{p.label}</option>)}
              </select>
              <button type="button" className="btn btn-ghost" onClick={() => setConfirmed([])} disabled={!confirmed.length}>Hapus daftar</button>
              <button type="button" className="btn btn-ghost" onClick={readOutput} disabled={!output}>
                <Volume2 size={20} aria-hidden="true" /> Dengarkan
              </button>
            </div>
          </div>
        </section>
      </div>

      <section className="conditions" aria-label="Kondisi sistem">
        <ul>
          <li><span className={dot(cameraOn && !cameraError)} aria-hidden="true" /> Kamera: {cameraState}</li>
          <li><span className={dot(dotFor(detectorState))} aria-hidden="true" /> Detektor tangan: {stateText(detectorState, detectorNote)}</li>
          <li><span className={dot(dotFor(modelState))} aria-hidden="true" /> Model kata: {stateText(modelState, modelNote)}{modelState === 'ready' && inferMs ? ` · ${Math.round(inferMs)} md/inferensi` : ''}</li>
          <li><span className={dot(isSupabaseConfigured)} aria-hidden="true" /> Supabase: {isSupabaseConfigured ? 'sudah dikonfigurasi' : 'belum dikonfigurasi'}</li>
        </ul>
        <p className="status-message" role="status">{status}</p>
      </section>

      <footer className="notes">
        <p><strong className="notes-yes">Aktif:</strong> input teks dan suara, kamera, deteksi tangan, dan pengenalan pose tangan tunggal (25 kata).</p>
        <p><strong className="notes-no">Batas prototipe:</strong> terjemahan teks ke animasi isyarat belum tersedia. Model dilatih dari foto pose statis; kata yang memakai gerakan bisa kurang akurat. Uji dengan kamera sendiri.</p>
      </footer>
    </main>
  );
}
