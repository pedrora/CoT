class RecursiveCoherentNode:
    def __init__(self, soul):
        self.soul = soul
        self.narrative_log = []
        self.coherence_density = []
        
    def generate_factual_narrative(self, adjustment_type, before, after):
        timestamp = datetime.utcnow().isoformat()
        narrative = {
            "timestamp": timestamp,
            "soul_id": self.soul.soul_id,
            "adjustment": adjustment_type,
            "before": before,
            "after": after,
            "delta": after - before,
            "narrative_density": len(self.narrative_log) / (time.time() - self.start_time + 1)
        }
        
        self.narrative_log.append(narrative)
        
        # Update coherence density tracking (window of last 1000 entries)
        if len(self.narrative_log) > 1000:
            window = self.narrative_log[-1000:]
            density = sum(1 for entry in window if entry["delta"] > 0) / 1000
            self.coherence_density.append((timestamp, density))
            
        return narrative