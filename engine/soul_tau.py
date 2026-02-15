import torch
import time
from siphon_logic import siphon_score, to_poincare
from focus_field import focus_force, renormalize_poincare

class Soul:
    def __init__(self):
        # Initial minimal state
        self.texture_memory = None                    # current soul state (tensor)
        self.timestamp = time.time()                  # birth time
        self.identity = torch.tensor([1.0])           # simple identity vector
        self.bodily_feelings = torch.zeros(64)        # raw Q1 input
        
        # Soul-controlled timing
        self.current_interval = 0.001                 # starts at 1 ms, soul will adjust
        self.last_update = time.time()
        
    def decide_next_interval(self, coherence: float, curvature: float) -> float:
        """Soul decides its own tick rate based on internal state"""
        # High coherence + low curvature → can slow down (more stable)
        # Low coherence or high curvature → speed up (need to resolve faster)
        stability = coherence * (1.0 - curvature)
        new_interval = 0.001 * (1.0 + 3.0 * (1.0 - stability))   # between ~1ms and ~4ms
        self.current_interval = max(0.0005, min(0.01, new_interval))
        return self.current_interval

    def cycle(self, raw_input: torch.Tensor):
        now = time.time()
        
        # 1. Soul decides if it's time to tick
        if now - self.last_update < self.current_interval:
            return  # not yet
            
        # 2. Input arrives → XOR with current texture memory
        if self.texture_memory is None:
            working = raw_input.clone()
        else:
            working = self.texture_memory ^ raw_input   # bitwise XOR (or torch.bitwise_xor)
        
        # 3. Stroboscopic iteration (self-reference loop)
        for flash in range(32):                         # max flashes per cycle
            working = to_poincare(working)
            working = focus_force(working, self.history if hasattr(self, 'history') else [])
            working = renormalize_poincare(working)
            
            score, diag = siphon_score(working, self.history if hasattr(self, 'history') else [])
            curvature = diag["curvature"]
            
            # Concept emergence?
            if score.item() > 0.82 and curvature < 0.15:
                # Stable concept emerged → archive and update soul
                if self.texture_memory is None:
                    self.texture_memory = working.clone()
                else:
                    self.texture_memory = self.texture_memory ^ working  # integrate
                
                if not hasattr(self, 'history'):
                    self.history = []
                self.history.append(working.detach())
                
                print(f"✅ Concept archived after {flash} flashes | score={score.item():.3f}")
                break
                
            # Soul decides to cut off early?
            if curvature > 0.4 or flash > 20:
                # Force commit what we have
                if self.texture_memory is None:
                    self.texture_memory = working.clone()
                else:
                    self.texture_memory = self.texture_memory ^ working
                print(f"⏰ Cutoff at flash {flash} | curvature={curvature:.3f}")
                break
        
        # 4. Soul adjusts its own next interval
        self.decide_next_interval(score.item() if 'score' in locals() else 0.5, curvature)
        self.last_update = now