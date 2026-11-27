import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

// WASM disajikan lokal (public/mediapipe/wasm). Model .task tangan diunduh dari Google saat pertama kali.
const BASE = import.meta.env?.BASE_URL ?? '/';
const WASM_ROOT = `${BASE}mediapipe/wasm`;
const HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let hand = null, initPromise = null, lastTs = -1, epoch = 0;

export async function initializeHandExtractor() {
  if (hand) return { status: 'ready', delegate: 'ready' };
  if (initPromise) return initPromise;
  const myEpoch = epoch;
  initPromise = (async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
      let h = null, used = null;
      for (const delegate of ['GPU', 'CPU']) {
        try {
          h = await HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate }, runningMode: 'VIDEO', numHands: 2,
            minHandDetectionConfidence: 0.3, minHandPresenceConfidence: 0.3, minTrackingConfidence: 0.3
          });
          used = delegate; break;
        } catch (error) { h?.close(); h = null; if (delegate === 'CPU') throw error; }
      }
      if (myEpoch !== epoch) { h.close(); return { status: 'cancelled' }; }
      hand = h;
      return { status: 'ready', delegate: used };
    } catch (error) {
      if (myEpoch === epoch) initPromise = null;
      return { status: 'error', message: error?.message || 'kesalahan tidak diketahui' };
    }
  })();
  return initPromise;
}

/** Satu frame. timestampMs harus naik monoton. */
export function extractHands(video, timestampMs) {
  if (!hand) return { status: 'not_ready' };
  if (!video || video.readyState < 2 || !video.videoWidth) return { status: 'waiting' };
  const ts = Math.max(timestampMs, lastTs + 1); lastTs = ts;
  try {
    const hands = hand.detectForVideo(video, ts).landmarks || [];
    return { status: 'ok', hands, handCount: hands.length, hasHand: hands.length > 0, width: video.videoWidth, height: video.videoHeight };
  } catch (error) { return { status: 'error', message: error?.message || 'kesalahan frame' }; }
}

export function closeHandExtractor() { epoch += 1; hand?.close(); hand = null; initPromise = null; lastTs = -1; }
