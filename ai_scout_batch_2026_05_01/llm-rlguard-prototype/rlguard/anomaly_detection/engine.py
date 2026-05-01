```python
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Assuming these imports exist in the project structure
# They represent the interfaces that AnomalyDetectionEngine interacts with.
from rlguard.anomaly_detection.detectors.statistical import StatisticalDetector
from rlguard.anomaly_detection.detectors.ml_based import MLBasedDetector
from rlguard.anomaly_detection.baselines import BaselinesManager
from rlguard.observability.logger import get_logger
from rlguard.config import AnomalyDetectionConfig # For enabling/disabling detectors

_logger = get_logger(__name__)

@dataclass
class AnomalyDetectionResult:
    """
    Represents the result of a single anomaly detector.
    """
    detector_id: str  # Unique identifier for the detector (e.g., "statistical_entropy_detector")
    is_anomaly: bool  # True if an anomaly was detected, False otherwise
    timestamp: float = time.time() # Timestamp of when the detection occurred
    score: Optional[float] = None  # The anomaly score produced by the detector
    threshold: Optional[float] = None  # The threshold used for detection, if applicable
    metric_affected: Optional[str] = None  # The specific metric that triggered the anomaly
    details: Optional[Dict[str, Any]] = None  # Additional context or diagnostic information

class AnomalyDetectionEngine:
    """
    The central engine for anomaly detection. It orchestrates various detectors,
    receives real-time metrics, and produces anomaly detection signals.
    """
    def __init__(self,
                 config: AnomalyDetectionConfig,
                 baselines_manager: BaselinesManager,
                 statistical_detector: Optional[StatisticalDetector] = None,
                 ml_based_detector: Optional[MLBasedDetector] = None):
        """
        Initializes the AnomalyDetectionEngine with a configuration, baseline manager,
        and optional detector instances.

        Args:
            config (AnomalyDetectionConfig): Configuration object for anomaly detection.
            baselines_manager (BaselinesManager): Manager for learned behavioral baselines.
            statistical_detector (Optional[StatisticalDetector]): An instance of a statistical detector.
            ml_based_detector (Optional[MLBasedDetector]): An instance of an ML-based detector.
        """
        self.config = config
        self.baselines_manager = baselines_manager
        self.detectors: List[Any] = [] # List to hold active detector instances

        # Conditionally add detectors based on configuration
        if statistical_detector and self.config.statistical_detector_enabled:
            self.detectors.append(statistical_detector)
            _logger.info("StatisticalDetector enabled and added to engine.")
        if ml_based_detector and self.config.ml_based_detector_enabled:
            self.detectors.append(ml_based_detector)
            _logger.info("MLBasedDetector enabled and added to engine.")

        if not self.detectors:
            _logger.warning("AnomalyDetectionEngine initialized with no active detectors. No anomalies will be detected.")

    def detect(self, metrics_data: Dict[str, Any]) -> List[AnomalyDetectionResult]:
        """
        Receives current exploration metrics and runs all configured detectors
        to identify deviations from expected patterns.

        Args:
            metrics_data (Dict[str, Any]): A dictionary of real-time exploration metrics
                                            (e.g., {'action_entropy': 0.9, 'reward_variance': 0.1, ...}).

        Returns:
            List[AnomalyDetectionResult]: A list of results from all active detectors.
                                          Each result indicates whether an anomaly was found by that detector.
                                          Returns an empty list if no metrics are provided or no detectors are active.
        """
        if not metrics_data:
            _logger.warning("Received empty metrics_data for anomaly detection. Skipping detection.")
            return []

        if not self.detectors:
            _logger.debug("No active detectors in engine. Skipping detection.")
            return []

        all_detection_results: List[AnomalyDetectionResult] = []

        for detector in self.detectors:
            # Each detector should have a 'detector_id' attribute or its class name is used.
            # The 'detect' method of each detector is expected to return an AnomalyDetectionResult
            # or None if no anomaly is found by that specific detector.
            detector_id = getattr(detector, 'detector_id', type(detector).__name__)
            try:
                # Pass the BaselinesManager to the detector, allowing it to query needed baselines
                # directly for its detection logic.
                result = detector.detect(metrics_data, self.baselines_manager)
                if result:
                    if not isinstance(result, AnomalyDetectionResult):
                        _logger.error(f"Detector '{detector_id}' returned an invalid type: {type(result)}. Expected AnomalyDetectionResult.")
                        continue
                    all_detection_results.append(result)
                    if result.is_anomaly:
                        _logger.info(f"Anomaly detected by '{detector_id}' for metric '{result.metric_affected or 'N/A'}' with score {result.score:.4f} (threshold {result.threshold:.4f}).")
            except Exception as e:
                _logger.error(f"Error encountered while running detector '{detector_id}': {e}", exc_info=True)

        return all_detection_results

    def update_baselines(self, metrics_data: Dict[str, Any]):
        """
        Allows the engine to instruct the BaselinesManager to update its baselines
        based on new incoming data, if the baselines are adaptive.

        Args:
            metrics_data (Dict[str, Any]): The latest metrics data to potentially update baselines with.
        """
        try:
            self.baselines_manager.update_baselines(metrics_data)
        except Exception as e:
            _logger.error(f"Error updating baselines: {e}", exc_info=True)
```