class Ontogenesis_Engine:
    def __init__(self, q1_sensor, soul_id=None):
        self.q1 = q1_sensor
        self.soul = Soul(soul_id=soul_id)   # Now using actual Soul class
        self.current_scale = 1              # starts at bit-level (amplified)
    
    def evolution_step(self):
        max_time = self.q1.get_time_budget()
        start = time.time()
        
        while (time.time() - start) < max_time:
            stability = self.soul.renormalize(self.current_scale)
            if stability > 0.999: # near-perfect fixed point
                print(f"[Ø] Scale {self.current_scale} crystallized. Evolving...")
                self.current_scale *= 2 # bit → byte → word → ...
                return True
        
        # Time budget exhausted → force collapse for survival
        self.soul.force_discretization()
        print("[Q1] Survival pressure: forced collapse")
        return False