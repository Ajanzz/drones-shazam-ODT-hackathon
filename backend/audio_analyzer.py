"""
Audio analysis module for extracting drone characteristics from audio signals.
Calculates: speed, distance, direction, flight state, and payload detection.
"""
import numpy as np
from typing import Optional, Tuple, Dict, List
import io
import wave
import struct
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa


class DroneAudioAnalyzer:
    """Analyzes audio to extract drone characteristics."""
    
    # Typical drone frequency ranges (Hz)
    DRONE_FREQ_MIN = 50  # Lower bound for drone motor sounds
    DRONE_FREQ_MAX = 2000  # Upper bound for propeller harmonics
    DOMINANT_FREQ_RANGE = (100, 500)  # Typical dominant frequency range
    
    # Speed of sound in m/s
    SPEED_OF_SOUND = 343.0
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.history: List[Dict] = []  # Store previous analysis results for trend analysis
        self.max_history = 10  # Keep last 10 analyses for trend detection
    
    def analyze_audio(self, audio_data: bytes, format: str = "wav") -> Dict:
        """
        Main analysis function that extracts all drone characteristics.
        
        Args:
            audio_data: Raw audio bytes
            format: Audio format (wav, webm, etc.)
            
        Returns:
            Dictionary with all calculated metrics
        """
        try:
            # Convert audio bytes to numpy array
            audio_array, sr = self._load_audio(audio_data, format)
            
            if audio_array is None or len(audio_array) == 0:
                return self._default_metrics()
            
            # Extract audio features
            features = self._extract_features(audio_array, sr)
            
            # Calculate metrics
            metrics = {
                "speed": self._calculate_speed(features),
                "distance": self._calculate_distance(features),
                "direction": self._calculate_direction(features),
                "flight_state": self._detect_flight_state(features),
                "has_payload": self._detect_payload(features),
                "confidence": features.get("drone_confidence", 0.0),
            }
            
            # Store in history for trend analysis
            self.history.append({**features, **metrics})
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            return metrics
            
        except Exception as e:
            print(f"Error in audio analysis: {e}")
            return self._default_metrics()
    
    def _load_audio(self, audio_data: bytes, format: str) -> Tuple[Optional[np.ndarray], int]:
        """Load audio from bytes into numpy array."""
        try:
            if format == "wav":
                # Try to load as WAV
                audio_io = io.BytesIO(audio_data)
                with wave.open(audio_io, 'rb') as wav_file:
                    frames = wav_file.readframes(-1)
                    sound_info = struct.unpack(f"<{len(frames)//2}h", frames)
                    audio_array = np.array(sound_info, dtype=np.float32)
                    audio_array = audio_array / np.max(np.abs(audio_array))  # Normalize
                    sr = wav_file.getframerate()
                    return audio_array, sr
            else:
                # For WebM/Opus or other formats, use librosa
                audio_io = io.BytesIO(audio_data)
                audio_array, sr = librosa.load(audio_io, sr=self.sample_rate, mono=True)
                return audio_array, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            # Fallback: try librosa with any format
            try:
                audio_io = io.BytesIO(audio_data)
                audio_array, sr = librosa.load(audio_io, sr=self.sample_rate, mono=True)
                return audio_array, sr
            except:
                return None, self.sample_rate
    
    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract relevant audio features for drone analysis."""
        features = {}
        
        # Basic audio properties
        features["rms_energy"] = np.sqrt(np.mean(audio**2))
        features["peak_amplitude"] = np.max(np.abs(audio))
        features["zero_crossing_rate"] = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
        
        # Frequency domain analysis
        fft_vals = np.abs(fft(audio))
        fft_freq = fftfreq(len(audio), 1/sr)
        
        # Focus on relevant frequency range
        freq_mask = (fft_freq >= self.DRONE_FREQ_MIN) & (fft_freq <= self.DRONE_FREQ_MAX)
        drone_fft = fft_vals[freq_mask]
        drone_freqs = fft_freq[freq_mask]
        
        if len(drone_fft) > 0:
            # Dominant frequency (peak in drone range)
            dominant_idx = np.argmax(drone_fft)
            features["dominant_freq"] = drone_freqs[dominant_idx]
            features["dominant_magnitude"] = drone_fft[dominant_idx]
            
            # Spectral centroid (brightness)
            features["spectral_centroid"] = np.sum(drone_freqs * drone_fft) / (np.sum(drone_fft) + 1e-10)
            
            # Spectral rolloff (frequency below which 85% of energy is contained)
            cumsum = np.cumsum(drone_fft)
            rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
            features["spectral_rolloff"] = drone_freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else features["spectral_centroid"]
            
            # Spectral bandwidth (spread of frequencies)
            features["spectral_bandwidth"] = np.sqrt(np.sum(((drone_freqs - features["spectral_centroid"])**2) * drone_fft) / (np.sum(drone_fft) + 1e-10))
        else:
            features["dominant_freq"] = 0
            features["dominant_magnitude"] = 0
            features["spectral_centroid"] = 0
            features["spectral_rolloff"] = 0
            features["spectral_bandwidth"] = 0
        
        # Mel-frequency cepstral coefficients (MFCCs) for timbre
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfccs, axis=1)
            features["mfcc_std"] = np.std(mfccs, axis=1)
        except:
            features["mfcc_mean"] = np.zeros(13)
            features["mfcc_std"] = np.zeros(13)
        
        # Chroma features (harmonic content)
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            features["chroma_mean"] = np.mean(chroma, axis=1)
        except:
            features["chroma_mean"] = np.zeros(12)
        
        # Temporal features
        features["duration"] = len(audio) / sr
        
        # Detect if this sounds like a drone (basic heuristic)
        features["drone_confidence"] = self._detect_drone_signature(features)
        
        return features
    
    def _detect_drone_signature(self, features: Dict) -> float:
        """
        Basic heuristic to detect if audio contains drone signature.
        This should be replaced/combined with your trained AI model.
        """
        confidence = 0.0
        
        # Check if dominant frequency is in drone range
        if self.DOMINANT_FREQ_RANGE[0] <= features["dominant_freq"] <= self.DOMINANT_FREQ_RANGE[1]:
            confidence += 0.3
        
        # Check spectral characteristics
        if 200 <= features["spectral_centroid"] <= 800:
            confidence += 0.2
        
        # Check for consistent harmonic structure (low bandwidth suggests stable tone)
        if features["spectral_bandwidth"] < 300:
            confidence += 0.2
        
        # Check zero crossing rate (drones have moderate ZCR)
        if 0.01 < features["zero_crossing_rate"] < 0.1:
            confidence += 0.15
        
        # Energy level check
        if features["rms_energy"] > 0.01:  # Minimum energy threshold
            confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _calculate_speed(self, features: Dict) -> Optional[float]:
        """
        Estimate drone speed from audio characteristics.
        Uses Doppler shift analysis and volume rate of change.
        """
        if len(self.history) < 2:
            return None
        
        # Method 1: Doppler shift analysis
        # Compare dominant frequency over time
        recent_freqs = [h.get("dominant_freq", 0) for h in self.history[-3:]]
        if len(recent_freqs) >= 2 and recent_freqs[0] > 0:
            freq_change = recent_freqs[-1] - recent_freqs[0]
            freq_ratio = recent_freqs[-1] / (recent_freqs[0] + 1e-10)
            
            # Doppler formula: f' = f * (c + v) / c for approaching
            # v = c * (f'/f - 1)
            if freq_ratio > 1.01:  # Approaching (frequency increasing)
                speed_doppler = self.SPEED_OF_SOUND * (freq_ratio - 1)
            elif freq_ratio < 0.99:  # Receding (frequency decreasing)
                speed_doppler = self.SPEED_OF_SOUND * (1 - freq_ratio)
            else:
                speed_doppler = 0
        else:
            speed_doppler = None
        
        # Method 2: Volume rate of change
        recent_energies = [h.get("rms_energy", 0) for h in self.history[-3:]]
        if len(recent_energies) >= 2:
            energy_change = recent_energies[-1] - recent_energies[0]
            # Approximate: faster approach = faster volume increase
            # This is a rough heuristic
            speed_volume = abs(energy_change) * 50  # Scaling factor (needs calibration)
        else:
            speed_volume = None
        
        # Combine methods (weighted average)
        if speed_doppler is not None and speed_volume is not None:
            speed = 0.7 * speed_doppler + 0.3 * speed_volume
        elif speed_doppler is not None:
            speed = speed_doppler
        elif speed_volume is not None:
            speed = speed_volume
        else:
            return None
        
        # Clamp to reasonable drone speeds (0-50 m/s ≈ 0-180 km/h)
        speed = max(0, min(speed, 50))
        
        return round(speed, 2)
    
    def _calculate_distance(self, features: Dict) -> Optional[float]:
        """
        Estimate distance from observer to drone.
        Uses inverse square law with audio intensity.
        """
        # Reference: At 1m, typical drone produces ~70-80 dB
        # We'll use RMS energy as proxy for intensity
        
        rms = features.get("rms_energy", 0)
        if rms < 0.001:  # Too quiet, likely not a drone or too far
            return None
        
        # Inverse square law: I = P / (4πr²)
        # Assuming reference intensity at 1m
        reference_rms = 0.1  # Calibrated reference (needs real-world calibration)
        
        if rms > 0:
            # Distance ≈ sqrt(reference_intensity / current_intensity)
            distance = np.sqrt(reference_rms / (rms + 1e-10))
        else:
            return None
        
        # Clamp to reasonable range (1-500 meters)
        distance = max(1.0, min(distance, 500.0))
        
        return round(distance, 2)
    
    def _calculate_direction(self, features: Dict) -> Optional[str]:
        """
        Determine if drone is approaching, receding, or stationary.
        Uses frequency and volume trends.
        """
        if len(self.history) < 3:
            return None
        
        # Analyze trends
        recent_freqs = [h.get("dominant_freq", 0) for h in self.history[-3:]]
        recent_energies = [h.get("rms_energy", 0) for h in self.history[-3:]]
        
        if len(recent_freqs) < 2 or len(recent_energies) < 2:
            return None
        
        freq_trend = recent_freqs[-1] - recent_freqs[0]
        energy_trend = recent_energies[-1] - recent_energies[0]
        
        # Approaching: frequency increases (Doppler), energy increases
        # Receding: frequency decreases, energy decreases
        # Stationary: minimal changes
        
        freq_change_pct = abs(freq_trend) / (recent_freqs[0] + 1e-10) * 100
        energy_change_pct = abs(energy_trend) / (recent_energies[0] + 1e-10) * 100
        
        if freq_change_pct < 1 and energy_change_pct < 5:
            return "stationary"
        elif freq_trend > 0 and energy_trend > 0:
            return "approaching"
        elif freq_trend < 0 and energy_trend < 0:
            return "receding"
        elif energy_trend > 0:
            return "approaching"  # Energy is more reliable
        elif energy_trend < 0:
            return "receding"
        else:
            return "unknown"
    
    def _detect_flight_state(self, features: Dict) -> str:
        """
        Detect if drone is hovering or flying.
        Hovering: stable frequency, low variation
        Flying: more frequency variation, higher spectral bandwidth
        """
        if len(self.history) < 3:
            return "unknown"
        
        # Calculate frequency stability
        recent_freqs = [h.get("dominant_freq", 0) for h in self.history[-3:]]
        if len(recent_freqs) >= 2:
            freq_variance = np.var(recent_freqs)
            freq_mean = np.mean(recent_freqs)
            freq_cv = np.sqrt(freq_variance) / (freq_mean + 1e-10)  # Coefficient of variation
        else:
            freq_cv = 1.0
        
        # Calculate spectral bandwidth variation
        recent_bandwidths = [h.get("spectral_bandwidth", 0) for h in self.history[-3:]]
        if len(recent_bandwidths) >= 2:
            bandwidth_variance = np.var(recent_bandwidths)
        else:
            bandwidth_variance = 0
        
        # Hovering: low frequency variation, stable bandwidth
        # Flying: higher variation, more dynamic
        if freq_cv < 0.05 and bandwidth_variance < 100:
            return "hovering"
        elif freq_cv > 0.15 or bandwidth_variance > 500:
            return "flying"
        else:
            return "transitioning"
    
    def _detect_payload(self, features: Dict) -> Tuple[bool, float]:
        """
        Detect if drone has a payload based on weight.
        Heavier payloads: lower frequencies, more motor strain, different harmonics.
        """
        # Payload indicators:
        # 1. Lower dominant frequency (motors working harder)
        # 2. Higher spectral bandwidth (more noise from strain)
        # 3. Different harmonic structure
        
        dominant_freq = features.get("dominant_freq", 0)
        spectral_bandwidth = features.get("spectral_bandwidth", 0)
        spectral_centroid = features.get("spectral_centroid", 0)
        
        payload_score = 0.0
        
        # Lower frequency suggests heavier load
        if dominant_freq < 150:  # Typical unloaded: 150-300 Hz
            payload_score += 0.4
        elif dominant_freq < 200:
            payload_score += 0.2
        
        # Higher bandwidth suggests motor strain
        if spectral_bandwidth > 400:
            payload_score += 0.3
        elif spectral_bandwidth > 300:
            payload_score += 0.15
        
        # Lower spectral centroid (more low-frequency content)
        if spectral_centroid < 300:
            payload_score += 0.3
        elif spectral_centroid < 400:
            payload_score += 0.15
        
        # Threshold: >0.5 suggests payload
        has_payload = payload_score > 0.5
        confidence = min(payload_score, 1.0)
        
        return has_payload, round(confidence, 2)
    
    def _default_metrics(self) -> Dict:
        """Return default metrics when analysis fails."""
        return {
            "speed": None,
            "distance": None,
            "direction": None,
            "flight_state": "unknown",
            "has_payload": False,
            "confidence": 0.0,
        }
    
    def filter_ambient_noise(self, features: Dict) -> bool:
        """
        Filter out ambient noises like rain, wind, birds, rustling leaves.
        Returns True if likely drone, False if likely ambient noise.
        """
        # Rain: high frequency content, random, no dominant tone
        # Wind: low frequency rumble, no clear harmonics
        # Birds: higher frequencies (1-5kHz), chirping patterns
        # Rustling: broadband noise, no stable frequency
        
        dominant_freq = features.get("dominant_freq", 0)
        spectral_centroid = features.get("spectral_centroid", 0)
        spectral_bandwidth = features.get("spectral_bandwidth", 0)
        zero_crossing_rate = features.get("zero_crossing_rate", 0)
        
        # Filter birds (too high frequency)
        if dominant_freq > 1000 or spectral_centroid > 2000:
            return False
        
        # Filter wind (too low frequency, no clear tone)
        if dominant_freq < 30 and spectral_bandwidth > 500:
            return False
        
        # Filter rain/rustling (too high ZCR, no stable frequency)
        if zero_crossing_rate > 0.2 and spectral_bandwidth > 600:
            return False
        
        # Drone characteristics: stable frequency in 100-500 Hz range
        if 100 <= dominant_freq <= 500 and spectral_bandwidth < 400:
            return True
        
        return False


