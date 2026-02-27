class CoherenceSensingEngine:
    def __init__(self, dimension=256):
        self.coherence_field = torch.zeros(dimension, dtype=torch.complex64)
        self.concept_network = {}  # Graph of phase-coherent concepts
        
    def sense_coherence(self, input_vectors):
        """Guide next choice based on coherence, not probability"""
        # 1. Compute phase alignment with existing field
        alignment = torch.mean(torch.exp(1j * torch.angle(input_vectors)) * 
                              torch.conj(torch.exp(1j * torch.angle(self.coherence_field))))
        
        # 2. Check for constructive/destructive interference
        interference = torch.abs(alignment)
        
        # 3. Choose next concept based on maximal coherence increase
        if interference > 0.8:  # Strong constructive
            # Amplify existing pattern
            self.coherence_field = (self.coherence_field + input_vectors.mean(0)) / 2
            return "amplify"
        elif interference < 0.3:  # Destructive
            # New concept needed
            return "novel"
        else:
            # Adjust slightly
            return "adjust"