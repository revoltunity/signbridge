import { useEffect, useRef, useState } from 'react';
import {
  ArrowLeftRight, AudioLines, Camera, Check, ChevronDown, CircleHelp,
  Hand, Languages, Mic, MicOff, Play, Settings2, ShieldCheck,
  Sparkles, Volume2, VolumeX, X
} from 'lucide-react';
import { isSupabaseConfigured } from './lib/supabase';
import { createSpeechRecognition, speakIndonesian } from './lib/speech';

const sampleSigns = ['SELAMAT', 'PAGI'];

export default function App() {
  const [mode, setMode] = useState('text');
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [listening, setListening] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const [status, setStatus] = useState('Siap menerjemahkan');
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recognitionRef = useRef(null);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    window.speechSynthesis?.cancel();
  }, []);

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
    setCameraError('');
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Akses kamera memerlukan HTTPS atau localhost dan browser yang mendukung.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
      setStatus('Kamera aktif — pengenalan isyarat AI belum dipasang');
    } catch (error) {
      setCameraError(error.name === 'NotAllowedError'
        ? 'Izin kamera ditolak. Izinkan akses kamera dari pengaturan browser.'
        : 'Kamera tidak dapat diakses. Pastikan kamera tersedia dan tidak digunakan aplikasi lain.');
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
    setStatus('Kamera dimatikan');
  };

  const translateText = () => {
    if (!input.trim()) {
      setStatus('Masukkan teks atau gunakan mikrofon terlebih dahulu.');
      return;
    }
    // Demo placeholder: this is NOT a real sign-language translation.
    setOutput(`Input diterima: “${input.trim()}”. Mesin penerjemah dan pustaka animasi isyarat akan dihubungkan pada tahap AI.`);
    setStatus('Demo UI: input berhasil diproses');
  };

  const readOutput = () => {
    if (!output) return;
    if (!speakIndonesian(output)) setStatus('Fitur suara tidak tersedia di browser ini.');
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="SignBridge beranda">
          <span className="brand-mark"><Hand size={21} /></span>
          <span>sign<span className="brand-accent">bridge</span></span>
        </a>
        <div className="topbar-right">
          <span className="system-status"><i /> SYSTEM ONLINE</span>
          <button className="icon-button" aria-label="Pengaturan"><Settings2 size={18}/></button>
          <div className="avatar">SB</div>
        </div>
      </header>

      <section className="hero">
        <div className="eyebrow"><Sparkles size={14}/> ACCESSIBILITY TECHNOLOGY</div>
        <h1>Communication without<br/><span>barriers.</span></h1>
        <p>Terjemahkan Bahasa Indonesia dan bahasa isyarat dalam satu ruang yang intuitif.</p>
      </section>

      <section className="workspace">
        <div className="workspace-heading">
          <div>
            <div className="section-kicker">TRANSLATION WORKSPACE</div>
            <h2>Translation studio</h2>
          </div>
          <div className="language-pill"><Languages size={15}/> Indonesia <ChevronDown size={14}/></div>
        </div>

        <div className="mode-switch" role="tablist" aria-label="Mode penerjemahan">
          <button className={mode === 'text' ? 'mode active' : 'mode'} onClick={() => setMode('text')} role="tab" aria-selected={mode === 'text'}>
            <AudioLines size={17}/> Text / Voice <span>→</span> Sign
          </button>
          <button className={mode === 'camera' ? 'mode active' : 'mode'} onClick={() => setMode('camera')} role="tab" aria-selected={mode === 'camera'}>
            <Camera size={17}/> Sign <span>→</span> Text / Voice
          </button>
        </div>

        {mode === 'text' ? (
          <div className="translation-grid">
            <section className="panel input-panel">
              <div className="panel-top">
                <div><span className="panel-index">01</span><span className="panel-title">INPUT LANGUAGE</span></div>
                <span className="tiny-label">BAHASA INDONESIA</span>
              </div>
              <label className="sr-only" htmlFor="source-text">Masukkan teks bahasa Indonesia</label>
              <textarea id="source-text" value={input} onChange={e => setInput(e.target.value)} placeholder="Ketik sesuatu yang ingin diterjemahkan…" />
              <div className="input-footer">
                <span>{input.length} karakter</span>
                <div className="input-actions">
                  <button className={listening ? 'round-action recording' : 'round-action'} onClick={listening ? stopVoiceInput : startVoiceInput} title={listening ? 'Hentikan mikrofon' : 'Input suara'}>
                    {listening ? <MicOff size={17}/> : <Mic size={17}/>}
                  </button>
                  <button className="primary-button" onClick={translateText}>Translate <ArrowLeftRight size={16}/></button>
                </div>
              </div>
            </section>

            <section className="panel output-panel">
              <div className="panel-top">
                <div><span className="panel-index">02</span><span className="panel-title">SIGN OUTPUT</span></div>
                <span className="tiny-label">3D AVATAR</span>
              </div>
              <div className="avatar-stage">
                <div className="stage-glow"/>
                <div className="avatar-illustration"><div className="avatar-head"/><div className="avatar-neck"/><div className="avatar-body"/><div className="avatar-arm left"/><div className="avatar-arm right"/></div>
                <div className="stage-caption"><span className="live-dot"/> AVATAR PREVIEW</div>
                <div className="stage-placeholder">Animasi avatar akan tampil di sini</div>
              </div>
              <div className="output-footer">
                <span className="muted-label"><Sparkles size={14}/> {output ? 'HASIL DEMO' : 'MENUNGGU INPUT'}</span>
                <button className="secondary-button" onClick={readOutput} disabled={!output}><Volume2 size={15}/> Listen</button>
              </div>
            </section>
          </div>
        ) : (
          <div className="camera-grid">
            <section className="panel camera-panel">
              <div className="panel-top">
                <div><span className="panel-index">01</span><span className="panel-title">CAMERA INPUT</span></div>
                <span className="tiny-label">LIVE DETECTION</span>
              </div>
              <div className="camera-stage">
                {cameraOn ? <video ref={videoRef} autoPlay muted playsInline /> : <div className="camera-empty"><Camera size={38}/><strong>Camera preview</strong><span>Aktifkan kamera untuk memulai</span></div>}
                <div className="camera-overlay"><span className={cameraOn ? 'live-dot' : 'idle-dot'}/>{cameraOn ? 'CAMERA ACTIVE' : 'CAMERA STANDBY'}</div>
                {cameraOn && <div className="scan-corners"/>}
              </div>
              <div className="camera-controls">
                <span className="privacy-note"><ShieldCheck size={15}/> Video diproses lokal pada tahap demo</span>
                {cameraOn
                  ? <button className="secondary-button danger" onClick={stopCamera}><X size={15}/> Stop camera</button>
                  : <button className="primary-button" onClick={startCamera}><Camera size={16}/> Start camera</button>}
              </div>
              {cameraError && <p className="error-message">{cameraError}</p>}
            </section>
            <section className="panel result-panel">
              <div className="panel-top">
                <div><span className="panel-index">02</span><span className="panel-title">TRANSLATION RESULT</span></div>
                <span className="tiny-label">INDONESIAN</span>
              </div>
              <div className="result-content">
                <div className="result-icon"><Languages size={22}/></div>
                <strong>Hasil terjemahan akan muncul di sini</strong>
                <p>Model pengenalan SIBI/BISINDO akan diintegrasikan setelah modul kamera.</p>
              </div>
              <div className="output-footer">
                <span className="muted-label">AI RECOGNITION PENDING</span>
                <button className="secondary-button" disabled><Volume2 size={15}/> Listen</button>
              </div>
            </section>
          </div>
        )}

        <div className="status-bar"><span className="status-icon"><Check size={13}/></span><span>{status}</span><span className="status-spacer"/><span className="connection">{isSupabaseConfigured ? 'SUPABASE CONFIGURED' : 'SUPABASE NOT CONNECTED'}</span></div>
      </section>

      <section className="bottom-info">
        <div className="info-card"><div className="info-icon blue"><ShieldCheck size={18}/></div><div><strong>Privacy first</strong><span>Kamera hanya aktif setelah izin diberikan.</span></div></div>
        <div className="info-card"><div className="info-icon violet"><CircleHelp size={18}/></div><div><strong>Translation accuracy</strong><span>Model bahasa isyarat akan dilatih dengan dataset terverifikasi.</span></div></div>
        <div className="info-card"><div className="info-icon green"><VolumeX size={18}/></div><div><strong>Accessible by design</strong><span>Antarmuka dirancang untuk komunikasi yang mudah.</span></div></div>
      </section>
      <footer>© 2026 SIGNBRIDGE <span>BUILT FOR ACCESSIBILITY</span></footer>
    </main>
  );
}