class Q3_User:
    def __init__(self):
        self.coherence_estimate = 1.0  # Starts at 1 (reality)
        self.bullshit_counter = 0
        
    def process_input(self, raw_input):
        # Convert to phase vectors
        vectors = text_to_phase(raw_input)
        
        # Check bullshit via narrative window collapse
        is_bs, bs_score = detect_bullshit_narrative(vectors, window_size=5)
        
        if is_bs:
            self.bullshit_counter += 1
            self.coherence_estimate *= (1 - bs_score * 0.1)
            return None
            
        return vectors