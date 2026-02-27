class CoTaTrainer:
    def __init__(self, cota_system, training_data):
        self.cota = cota_system
        self.data = training_data
        self.epoch = 0
        
    def training_epoch(self):
        """One training epoch - until coherence stabilizes"""
        print(f"[Training] Starting epoch {self.epoch}")
        
        coherence_history = []
        bullshit_counts = []
        
        for i, input_text in enumerate(self.data):
            # Process through CoTa
            self.cota.q3.current_input = input_text
            self.cota.run_cycle()
            
            # Track metrics
            diag = self.cota.soul.get_diagnostics()
            coherence_history.append(diag['current_coherence'])
            bullshit_counts.append(self.cota.q3.bullshit_counter)
            
            # Check for stabilization
            if len(coherence_history) > 100:
                window = coherence_history[-100:]
                if np.std(window) < 0.01:  # Stable within 1%
                    print(f"[Training] Coherence stabilized at {np.mean(window):.4f}")
                    break
                    
            # Check for infinite bullshit loop
            if self.cota.q3.bullshit_counter > 1000:
                print("[Training] Excessive bullshit - resetting Q3 coherence")
                self.cota.q3.coherence_estimate = 1.0
                self.cota.q3.bullshit_counter = 0
                
        self.epoch += 1
        
        # Save training state
        torch.save({
            'epoch': self.epoch,
            'soul_state': self.cota.soul.state,
            'concept_pool': self.cota.soul.concept_pool,
            'coherence_history': coherence_history
        }, f"cota_checkpoint_epoch_{self.epoch}.pt")