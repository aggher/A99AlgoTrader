import logging
import datetime
from typing import Dict, List, Any

log = logging.getLogger(__name__)

class MacroEventDetector:
    """
    Section 6: Event Risk Protection Module.
    Identifies high-impact economic events (FOMC, CPI, NFP) and restricts trading.
    """

    def __init__(self):
        # In a production system, this would fetch from an Economic Calendar API (e.g. DailyFX, ForexFactory)
        # Here we use a mock schedule for demonstration.
        self.high_impact_events = [
            {"event": "FOMC Interest Rate Decision", "impact": "CRITICAL", "hour_utc": 19},
            {"event": "US CPI (Inflation)", "impact": "HIGH", "hour_utc": 13},
            {"event": "Non-Farm Payrolls (NFP)", "impact": "HIGH", "hour_utc": 13}
        ]
        self.risk_window_minutes = 60 # Pause trading 60 minutes before/after

    def get_current_event_risk(self) -> Dict[str, Any]:
        """
        Checks if any high-impact events are imminent.
        """
        now = datetime.datetime.utcnow()
        current_hour = now.hour
        current_minute = now.minute
        
        for event in self.high_impact_events:
            event_hour = event["hour_utc"]
            # Only block within ±30 minutes of the event hour
            minutes_from_event = abs((current_hour * 60 + current_minute) - (event_hour * 60))
            if minutes_from_event < 30:
                log.warning(f"⚠️ Event Risk Detected: {event['event']} is imminent.")
                return {
                    "is_risky": True,
                    "event_name": event["event"],
                    "impact": event["impact"],
                    "action": "PAUSE" if event["impact"] == "CRITICAL" else "REDUCE_SIZE"
                }

        return {"is_risky": False, "event_name": None, "impact": None, "action": "NONE"}

    def evaluate_risk_status(self) -> float:
        """
        Returns a Risk Multiplier: 0.0 (Stop), 0.5 (Reduce), 1.0 (Normal)
        """
        risk = self.get_current_event_risk()
        if risk["action"] == "PAUSE":
            return 0.0
        elif risk["action"] == "REDUCE_SIZE":
            return 0.5
        return 1.0
