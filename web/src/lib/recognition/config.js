// Semua konstanta pengenalan di satu tempat.
export const DECISION = Object.freeze({
  minConfidence: 0.70,   // probabilitas teratas minimal
  minMargin: 0.40,       // selisih dengan kandidat kedua minimal
  duplicateSuppressSec: 1.0
});

export const SENSITIVITY_PRESETS = Object.freeze({
  ketat:    { label: 'Ketat',    minConfidence: 0.85, minMargin: 0.55 },
  seimbang: { label: 'Seimbang', minConfidence: 0.70, minMargin: 0.40 },
  longgar:  { label: 'Longgar',  minConfidence: 0.50, minMargin: 0.25 }
});

// Mesin pose statis (satu pose tangan = satu kata).
export const STATIC = Object.freeze({
  windowFrames: 8,        // rata-rata probabilitas dari N frame terakhir
  minFrames: 4,           // minimal frame sebelum dinilai
  holdSec: 0.35,          // label yang sama harus bertahan selama ini sebelum dikonfirmasi
  endGapSec: 0.5,         // tangan hilang selama ini = isyarat selesai
  minActiveSec: 0.3,      // tangan terlalu singkat tidak dilaporkan
  duplicateSuppressSec: 1.0
});
