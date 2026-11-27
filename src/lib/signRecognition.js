/**
 * Integration point for MediaPipe + a trained SIBI/BISINDO classifier.
 *
 * Important:
 * - MediaPipe detects hand/body landmarks; it does not translate a language by itself.
 * - A production recognizer needs a labeled dataset, temporal sequence handling,
 *   and a model trained for the selected sign-language standard.
 * - Keep camera frames local by default; only transmit them with explicit consent.
 */
export async function initializeSignRecognizer() {
  // Next implementation step:
  // 1. Install @mediapipe/tasks-vision
  // 2. Initialize HandLandmarker (and optionally PoseLandmarker)
  // 3. Extract landmark sequences
  // 4. Pass sequences into a trained classifier
  return {
    status: 'not-configured',
    message: 'MediaPipe/model integration is the next development stage.'
  };
}