import logging
from typing import Dict, Any, Optional

class Streamer:
    """
    Pushes collected exploration metrics to the Anomaly Detection Engine,
    simulating real-time data flow.
    """

    def __init__(self, anomaly_detection_engine_instance: Any):
        """
        Initializes the Streamer with a reference to the Anomaly Detection Engine.

        Args:
            anomaly_detection_engine_instance: An instance of the AnomalyDetectionEngine
                                               to which metrics will be streamed.
        """
        self.anomaly_detection_engine = anomaly_detection_engine_instance
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Telemetry Streamer initialized.")

        if self.anomaly_detection_engine and not hasattr(self.anomaly_detection_engine, 'process_metrics'):
            self.logger.warning(
                "AnomalyDetectionEngine instance does not have a 'process_metrics' method. "
                "Ensure the engine is correctly configured to receive metrics."
            )

    def stream_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Streams a batch of collected metrics to the Anomaly Detection Engine.

        Args:
            metrics (Dict[str, Any]): A dictionary of collected exploration metrics.
                                      e.g., {'action_entropy': 0.8, 'reward_variance': 0.15}
            step (Optional[int]): The current training step or episode number, for context.
        """
        if not metrics:
            self.logger.warning("Attempted to stream empty metrics. Skipping.")
            return

        self.logger.debug(f"Streaming metrics for step {step}: {metrics.keys()}")
        
        try:
            if self.anomaly_detection_engine:
                self.anomaly_detection_engine.process_metrics(metrics, step=step)
                self.logger.info(f"Metrics successfully streamed for step {step}.")
            else:
                self.logger.error("AnomalyDetectionEngine is not initialized. Cannot stream metrics.")
        except AttributeError as ae:
            self.logger.error(
                f"AnomalyDetectionEngine object '{type(self.anomaly_detection_engine).__name__}' "
                f"does not have the expected 'process_metrics' method: {ae}", exc_info=True
            )
        except Exception as e:
            self.logger.error(f"Error streaming metrics for step {step}: {e}", exc_info=True)