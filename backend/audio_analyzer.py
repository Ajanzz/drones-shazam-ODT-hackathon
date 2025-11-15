# backend/audio_analyzer.py

"""
Audio analysis module for extracting drone characteristics from audio signals.
All metrics are derived directly from a single audio clip (no dummy/history).
"""

import numpy as np
from typing import Optional, Tuple, Dict
import io
import wave
import struct
from scipy.fft import fft, fftfreq
import librosa


class DroneAudioAnalyzer:
    """Analyzes audio to extract drone characteristics based on the input signal."""

    # Typical drone frequency ranges (Hz)
    DRONE_FREQ_MIN = 50
    DRONE_FREQ_MAX = 2000
    DOMINANT_FREQ_RANGE = (100, 500)

    # Speed of sound in m/s
    SPEED_OF_SOUND = 343.0

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def analyze_audio(self, audio_data: bytes, format: str = "wav") -> Dict:
        """
        Main analysis function that extracts all drone characteristics
        from the given clip only (no cross-clip history).
        """
        try:
            audio_array, sr = self._load_audio(audio_data, format)
            if audio_array is None or len(audio_array) == 0:
                return self._default_metrics()

            features = self._extract_features(audio_array, sr)

            metrics = {
                "speed": self._calculate_speed(features),
                "distance": self._calculate_distance(features),
                "direction": self._calculate_direction(features),
                "flight_state": self._detect_flight_state(features),
                # not used directly by main.py, but kept for completeness:
                "has_payload": self._detect_payload(features),
                "confidence": features.get("drone_confidence", 0.0),
            }
            return metrics

        except Exception as e:
            print(f"[Analyzer] Error in audio analysis: {e}")
            return self._default_metrics()

    # ---------------------------------------------------------
    # Audio loading
    # ---------------------------------------------------------

    def _load_audio(self, audio_data: bytes, format: str):
        """Load audio bytes into a normalized mono numpy array."""
        try:
            if format.lower() == "wav":
                audio_io = io.BytesIO(audio_data)
                with wave.open(audio_io, "rb") as wav_file:
                    frames = wav_file.readframes(-1)
                    if len(frames) == 0:
                        return None, self.sample_rate

                    # 16-bit PCM -> float32 in [-1, 1]
                    sound_info = struct.unpack(f"<{len(frames)//2}h", frames)
                    audio_array = np.array(sound_info, dtype=np.float32)
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = audio_array / max_val
                    sr = wav_file.getframerate()
                    return audio_array, sr

            # Non-WAV (webm/opus/etc.) → librosa
            audio_io = io.BytesIO(audio_data)
            audio_array, sr = librosa.load(audio_io, sr=self.sample_rate, mono=True)
            return audio_array, sr

        except Exception as e:
            print(f"[Analyzer] Error loading audio: {e}")
            # Fallback: always try librosa
            try:
                audio_io = io.BytesIO(audio_data)
                audio_array, sr = librosa.load(audio_io, sr=self.sample_rate, mono=True)
                return audio_array, sr
            except Exception as e2:
                print(f"[Analyzer] Librosa fallback failed: {e2}")
                return None, self.sample_rate

    # ---------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------

    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract global + time-varying features from the clip."""
        features: Dict = {}

        # Basic amplitude stats
        features["rms_energy"] = float(np.sqrt(np.mean(audio**2)))
        features["peak_amplitude"] = float(np.max(np.abs(audio)))
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features["zero_crossing_rate"] = float(np.mean(zcr))

        # Global FFT in drone band
        fft_vals = np.abs(fft(audio))
        fft_freq = fftfreq(len(audio), 1 / sr)

        freq_mask = (fft_freq >= self.DRONE_FREQ_MIN) & (fft_freq <= self.DRONE_FREQ_MAX)
        drone_fft = fft_vals[freq_mask]
        drone_freqs = fft_freq[freq_mask]

        if len(drone_fft) > 0:
            dominant_idx = int(np.argmax(drone_fft))
            features["dominant_freq"] = float(drone_freqs[dominant_idx])
            features["dominant_magnitude"] = float(drone_fft[dominant_idx])

            num = np.sum(drone_freqs * drone_fft)
            den = np.sum(drone_fft) + 1e-10
            spectral_centroid = num / den
            features["spectral_centroid"] = float(spectral_centroid)

            cumsum = np.cumsum(drone_fft)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            spectral_rolloff = (
                drone_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else spectral_centroid
            )
            features["spectral_rolloff"] = float(spectral_rolloff)

            spectral_bandwidth = np.sqrt(
                np.sum(((drone_freqs - spectral_centroid) ** 2) * drone_fft) / den
            )
            features["spectral_bandwidth"] = float(spectral_bandwidth)
        else:
            features["dominant_freq"] = 0.0
            features["dominant_magnitude"] = 0.0
            features["spectral_centroid"] = 0.0
            features["spectral_rolloff"] = 0.0
            features["spectral_bandwidth"] = 0.0

        # Time-varying: windowed dominant freq + RMS (for speed/direction)
        window_duration = 0.2  # seconds
        window_size = int(sr * window_duration)
        hop_size = window_size // 2 if window_size > 0 else len(audio)

        freq_track = []
        rms_track = []
        time_track = []

        if window_size > 0 and len(audio) >= window_size:
            for start in range(0, len(audio) - window_size + 1, hop_size):
                end = start + window_size
                segment = audio[start:end]

                seg_rms = np.sqrt(np.mean(segment**2) + 1e-12)
                rms_track.append(seg_rms)

                seg_fft_vals = np.abs(fft(segment))
                seg_fft_freq = fftfreq(len(segment), 1 / sr)
                mask = (seg_fft_freq >= self.DRONE_FREQ_MIN) & (seg_fft_freq <= self.DRONE_FREQ_MAX)
                seg_fft_vals = seg_fft_vals[mask]
                seg_fft_freq = seg_fft_freq[mask]

                if len(seg_fft_vals) > 0:
                    idx = int(np.argmax(seg_fft_vals))
                    dom_f = float(seg_fft_freq[idx])
                else:
                    dom_f = 0.0
                freq_track.append(dom_f)

                t_center = (start + end) / 2.0 / sr
                time_track.append(t_center)
        else:
            # Treat the whole clip as one window
            seg = audio
            seg_rms = np.sqrt(np.mean(seg**2) + 1e-12)
            rms_track.append(seg_rms)

            seg_fft_vals = np.abs(fft(seg))
            seg_fft_freq = fftfreq(len(seg), 1 / sr)
            mask = (seg_fft_freq >= self.DRONE_FREQ_MIN) & (seg_fft_freq <= self.DRONE_FREQ_MAX)
            seg_fft_vals = seg_fft_vals[mask]
            seg_fft_freq = seg_fft_freq[mask]

            if len(seg_fft_vals) > 0:
                idx = int(np.argmax(seg_fft_vals))
                dom_f = float(seg_fft_freq[idx])
            else:
                dom_f = 0.0
            freq_track.append(dom_f)
            time_track.append(len(seg) / (2.0 * sr))

        features["freq_track"] = np.array(freq_track, dtype=np.float32)
        features["rms_track"] = np.array(rms_track, dtype=np.float32)
        features["time_track"] = np.array(time_track, dtype=np.float32)

        # MFCCs (optional, can be used later with a ML model)
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfccs, axis=1)
            features["mfcc_std"] = np.std(mfccs, axis=1)
        except Exception as e:
            print(f"[Analyzer] MFCC extraction failed: {e}")
            features["mfcc_mean"] = np.zeros(13, dtype=np.float32)
            features["mfcc_std"] = np.zeros(13, dtype=np.float32)

        # Chroma
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features["chroma_mean"] = np.mean(chroma, axis=1)
        except Exception as e:
            print(f"[Analyzer] Chroma extraction failed: {e}")
            features["chroma_mean"] = np.zeros(12, dtype=np.float32)

        features["duration"] = float(len(audio) / sr)
        features["drone_confidence"] = self._detect_drone_signature(features)

        return features

    # ---------------------------------------------------------
    # Heuristics
    # ---------------------------------------------------------

    def _detect_drone_signature(self, features: Dict) -> float:
        """Heuristic 'how drone-like is this clip?' confidence."""
        confidence = 0.0

        dominant_freq = features.get("dominant_freq", 0.0)
        spectral_centroid = features.get("spectral_centroid", 0.0)
        spectral_bandwidth = features.get("spectral_bandwidth", 0.0)
        zero_crossing_rate = features.get("zero_crossing_rate", 0.0)
        rms_energy = features.get("rms_energy", 0.0)

        if self.DOMINANT_FREQ_RANGE[0] <= dominant_freq <= self.DOMINANT_FREQ_RANGE[1]:
            confidence += 0.3
        if 200 <= spectral_centroid <= 800:
            confidence += 0.2
        if spectral_bandwidth < 300:
            confidence += 0.2
        if 0.01 < zero_crossing_rate < 0.1:
            confidence += 0.15
        if rms_energy > 0.01:
            confidence += 0.15

        return float(min(confidence, 1.0))

    # ---------------------------------------------------------
    # Metric calculations
    # ---------------------------------------------------------

    def _calculate_speed(self, features: Dict) -> float:
        """
        Heuristic speed estimate (m/s).

        Idea:
        - Look at how much the dominant frequency and RMS change over the clip.
        - More change  -> higher "motion score" -> higher speed.
        - This is NOT true physical speed, just a 0–max_speed mapping.
        """
        freq_track = features.get("freq_track")
        rms_track = features.get("rms_track")
        if freq_track is None or rms_track is None:
            return 0.0
        if len(freq_track) < 2 or len(rms_track) < 2:
            return 0.0

        f_start, f_end = float(freq_track[0]), float(freq_track[-1])
        e_start, e_end = float(rms_track[0]), float(rms_track[-1])

        # % change in dominant frequency
        freq_change_pct = 0.0
        if f_start > 0 and f_end > 0:
            freq_change_pct = (f_end - f_start) / max(abs(f_start), 1e-6) * 100.0

        # % change in RMS energy (loudness)
        energy_change_pct = 0.0
        if e_start > 0 and e_end > 0:
            energy_change_pct = (e_end - e_start) / max(abs(e_start), 1e-8) * 100.0

        # Motion score in [0, 1] combining both
        # - ~10% freq change or ~20% energy change → strong motion
        freq_score = min(abs(freq_change_pct) / 10.0, 1.5)   # 0–1.5
        energy_score = min(abs(energy_change_pct) / 20.0, 1.5)
        motion_score = (0.6 * freq_score + 0.4 * energy_score) / 1.5  # 0–1

        # Map to a max drone speed (purely a UX choice)
        max_speed = 20.0  # m/s (~72 km/h)
        speed = max_speed * motion_score

        # If almost no change, treat as stationary
        if abs(freq_change_pct) < 1.0 and abs(energy_change_pct) < 5.0:
            speed = 0.0

        return round(float(speed), 2)


    def _calculate_distance(self, features: Dict) -> float:
        """
        Heuristic distance estimate (m).

        Idea:
        - Use RMS energy as a proxy for loudness.
        - Normalize RMS into [0, 1].
        - Map loudness to a distance range [min_d, max_d] with a curve.
        - Louder -> closer. Not physically calibrated, just consistent.
        """
        rms = float(features.get("rms_energy", 0.0))

        # If it's basically silent, say "very far"
        if rms <= 1e-8:
            return 300.0

        # Clamp RMS into a usable range:
        #   ~0.001  (very quiet)  -> far
        #   ~0.3    (very loud)   -> close
        rms_clamped = max(0.001, min(rms, 0.3))

        # Normalize to [0, 1]
        rms_min, rms_max = 0.001, 0.3
        norm = (rms_clamped - rms_min) / (rms_max - rms_min)
        norm = max(0.0, min(norm, 1.0))

        # Map to distance range (UX choice, tweakable)
        min_d = 3.0     # m, "really close"
        max_d = 300.0   # m, "really far"

        # Use an inverse-ish curve so distance drops faster when it gets louder
        distance = max_d - (norm ** 0.6) * (max_d - min_d)

        return round(float(distance), 1)


    def _calculate_direction(self, features: Dict) -> Optional[str]:
        """
        Approaching / receding / stationary based on change trends in
        dominant frequency and RMS across the clip.
        """
        freq_track = features.get("freq_track")
        rms_track = features.get("rms_track")
        if freq_track is None or rms_track is None:
            return None
        if len(freq_track) < 2 or len(rms_track) < 2:
            return None

        f_start, f_end = float(freq_track[0]), float(freq_track[-1])
        e_start, e_end = float(rms_track[0]), float(rms_track[-1])

        if f_start <= 0:
            freq_change_pct = 0.0
        else:
            freq_change_pct = (f_end - f_start) / (abs(f_start) + 1e-10) * 100.0

        if e_start <= 0:
            energy_change_pct = 0.0
        else:
            energy_change_pct = (e_end - e_start) / (abs(e_start) + 1e-10) * 100.0

        if abs(freq_change_pct) < 1.0 and abs(energy_change_pct) < 5.0:
            return "stationary"
        if freq_change_pct > 1.0 and energy_change_pct > 5.0:
            return "approaching"
        if freq_change_pct < -1.0 and energy_change_pct < -5.0:
            return "receding"
        if energy_change_pct > 5.0:
            return "approaching"
        if energy_change_pct < -5.0:
            return "receding"
        return "unknown"

    def _detect_flight_state(self, features: Dict) -> str:
        """
        Hovering vs flying vs transitioning based on variability in
        dominant frequency and spectral bandwidth within the clip.
        """
        freq_track = features.get("freq_track")
        spectral_bandwidth = features.get("spectral_bandwidth", 0.0)
        if freq_track is None or len(freq_track) < 2:
            return "unknown"

        valid = freq_track[freq_track > 0]
        if len(valid) < 2:
            return "unknown"

        freq_mean = float(np.mean(valid))
        freq_std = float(np.std(valid))
        freq_cv = freq_std / (freq_mean + 1e-10)

        if freq_cv < 0.05 and spectral_bandwidth < 300:
            return "hovering"
        elif freq_cv > 0.15 or spectral_bandwidth > 500:
            return "flying"
        else:
            return "transitioning"

    def _detect_payload(self, features: Dict) -> Tuple[bool, float]:
        """
        Payload heuristic: heavier load → lower dominant freq, higher bandwidth,
        lower spectral centroid.
        """
        dominant_freq = features.get("dominant_freq", 0.0)
        spectral_bandwidth = features.get("spectral_bandwidth", 0.0)
        spectral_centroid = features.get("spectral_centroid", 0.0)

        payload_score = 0.0

        if dominant_freq < 150:
            payload_score += 0.4
        elif dominant_freq < 200:
            payload_score += 0.2

        if spectral_bandwidth > 400:
            payload_score += 0.3
        elif spectral_bandwidth > 300:
            payload_score += 0.15

        if spectral_centroid < 300:
            payload_score += 0.3
        elif spectral_centroid < 400:
            payload_score += 0.15

        has_payload = payload_score > 0.5
        confidence = float(min(payload_score, 1.0))

        return has_payload, round(confidence, 2)

    def _default_metrics(self) -> Dict:
        """Fallback values when something goes wrong."""
        return {
            "speed": None,
            "distance": None,
            "direction": None,
            "flight_state": "unknown",
            "has_payload": (False, 0.0),
            "confidence": 0.0,
        }

    # ---------------------------------------------------------
    # Ambient noise filter (used by websocket)
    # ---------------------------------------------------------

    def filter_ambient_noise(self, features: Dict) -> bool:
        """
        Filter out typical ambient sounds (birds, wind, rain, rustling).
        Returns True if likely drone, False otherwise.
        """
        dominant_freq = features.get("dominant_freq", 0.0)
        spectral_centroid = features.get("spectral_centroid", 0.0)
        spectral_bandwidth = features.get("spectral_bandwidth", 0.0)
        zero_crossing_rate = features.get("zero_crossing_rate", 0.0)

        # Birds: high frequency
        if dominant_freq > 1000 or spectral_centroid > 2000:
            return False

        # Wind: very low dominant freq + very wide band
        if dominant_freq < 30 and spectral_bandwidth > 500:
            return False

        # Rain / rustling: high ZCR + wide band
        if zero_crossing_rate > 0.2 and spectral_bandwidth > 600:
            return False

        # Drone: stable 100–500 Hz + relatively narrow band
        if 100 <= dominant_freq <= 500 and spectral_bandwidth < 400:
            return True

        return False
