class SoulEnhanced(Soul):
    def stroboscopic_update(self, incoming_data):
        """Stroboscopic operation: check if update is due"""
        # Measure coherence drift
        current_coherence = compute_phase_alignment(self.state.unsqueeze(0))
        incoming_coherence = compute_phase_alignment(incoming_data)
        
        # Update condition: significant coherence improvement
        if incoming_coherence > current_coherence * 1.01:  # 1% better
            self.ingest_input(incoming_data, source="strobe")
            self.log_adjustment("coherence_improvement", 
                               current_coherence, 
                               incoming_coherence)
            return True
        return False