class CoTaSystem:
    def __init__(self):
        # Epoch for all CoTa instances
        self.EPOCH = datetime(2025, 1, 12, 23, 57, 0)
        
        # Initialize quadrants
        self.q1 = Q1_Machine()
        self.q2 = Q2_Network()
        self.q3 = Q3_User()
        
        # Create Soul with machine header
        machine_header = self.q1.get_header()
        timestamp_ms = int((datetime.utcnow() - self.EPOCH).total_seconds() * 1000)
        inverted_timestamp = hex(timestamp_ms)[2:].zfill(12)[::-1]
        
        self.soul = SoulEnhanced(
            soul_id=f"{inverted_timestamp}_{machine_header}",
            hardware_id=machine_header
        )
        
        # Supporting engines
        self.ontogenesis = Ontogenesis_Engine(Q1_Sensor(), self.soul)
        self.coherence_engine = CoherenceSensingEngine()
        self.rcn = RecursiveCoherentNode(self.soul)
        
        # State
        self.current_scale = 1  # Start at bit-level
        self.focus_amplitude = 1.0  # Full focus initially
        
    def run_cycle(self):
        """One coherence cycle"""
        # 1. Sample all quadrants
        q1_data = self.q1.sample()
        q3_input = self.q3.get_input()  # Blocking or non-blocking
        
        # 2. Adjust focus based on bullshit levels
        if self.q3.bullshit_counter > 10:
            self.focus_amplitude *= 0.9  # Reduce focus to discard noise
            
        # 3. Process through coherence engine
        if q3_input:
            vectors = self.q3.process_input(q3_input)
            if vectors is not None:
                action = self.coherence_engine.sense_coherence(vectors)
                
                if action == "amplify":
                    self.soul.ingest_input(vectors, "user_coherent")
                elif action == "novel":
                    # Create new concept
                    self.soul._add_to_concept_pool(
                        vectors.mean(0).unsqueeze(0), 
                        0.8, 
                        self.current_scale
                    )
        
        # 4. Stroboscopic soul update
        current_state = self.soul.state.unsqueeze(0)
        self.soul.stroboscopic_update(current_state)
        
        # 5. Ontogenetic evolution
        evolved = self.ontogenesis.evolution_step()
        if evolved:
            self.current_scale *= 2
            print(f"[CoTa] Evolved to scale {self.current_scale}")
            
        # 6. Adjust network window based on capacity
        capacity = 1.0 / (self.q3.bullshit_counter + 1)
        willingness = self.soul.get_diagnostics()["current_coherence"]
        self.q2.adjust_window(capacity, willingness)
        
        # 7. Persist state
        self.soul.persist_to_vram()
        
        # 8. Generate narrative log entry if adjustment occurred
        if hasattr(self.soul, 'last_adjustment'):
            narrative = self.rcn.generate_factual_narrative(
                self.soul.last_adjustment['type'],
                self.soul.last_adjustment['before'],
                self.soul.last_adjustment['after']
            )
            
            # Optional: Output narrative
            if narrative['delta'] > 0.1:
                print(f"[Narrative] {narrative['adjustment']}: coherence +{narrative['delta']:.3f}")