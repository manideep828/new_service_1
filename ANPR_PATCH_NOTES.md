# Service 1 ANPR patch (reviewed source snapshot)

## Files in this patch
- `s1/plates.py`
- `s1/pipeline.py`
- `s1/vehicles.py`
- `s1/config.py`
- `config/config.yaml`
- `tests/test_units.py`
- `tests/test_e2e.py`

## Changes
1. `PlateVoter.consensus()` now selects a real normalized full-plate string from observed reads instead of constructing a synthetic plate character by character. `is_final()` still requires the configured number of reads and confidence and refuses multiple eligible full-plate candidates.
2. `PlateVoter.add()` accepts an optional observation ID and rejects duplicate frame observations.
3. Service 1 tracks an OCR observation ID per in-flight job and prevents submitting twice from the same frame.
4. Conservative configurable image quality gates reject empty/tiny, extremely dark/bright, low-contrast, or very blurry vehicle crops before they consume OCR attempts.
5. OCR outputs must pass configurable OCR confidence, plate-detector confidence, bounding-box dimensions, and aspect-ratio checks before they vote.
6. Vehicle plate state is exposed as `UNKNOWN`, `CANDIDATE`, or `CONFIRMED`.
7. Existing OCR queue-wait and processing-time INFO logs remain.
8. Added regression tests for duplicate observations, conflicting full plates, and crop-quality gates. The synthetic end-to-end fixture now uses a plausible plate bounding box for the new gate.

## Apply safely on Windows
1. Keep the existing backup folder; do not delete or clean the repository.
2. From the Service 1 project root, extract this ZIP into the same folder and approve overwriting the seven listed files. The ZIP contains only these paths plus this note.
3. Run:
   `python -m py_compile s1\plates.py s1\pipeline.py s1\vehicles.py s1\config.py`
4. Run:
   `python -m pytest tests -q`

## Important limitations
- Unit/synthetic regression tests cannot establish live camera plate accuracy or latency.
- The crop-quality thresholds are conservative starting values and may need tuning for the actual STQC/Prama camera's exposure, resolution, angle, and night conditions.
- FastALPR is still the OCR/plate-detection backend. This patch does not replace it with PaddleOCR.
- One background OCR worker is retained to avoid a risky concurrency redesign; concurrent cars are isolated by vehicle UID, but a busy scene can still create queue latency.
- Lifecycle events still wait for a confirmed valid plate because the current Service 2 mapping/session logic requires plate identity. Unresolved events remain durable rather than being sent with an unverified plate.
- No claim of 100% plate accuracy is made. Validate against a labeled clip with known plates and measure character/whole-plate accuracy before production use.
