# No teu loop principal do CoTa:
while True:
    # Q1: RTC Pulse
    current_time = get_inverted_timestamp()
    
    # Q4: Soul strobe
    soul_state = soul.get_current_phase()
    
    # Executa busca estroboscópica em threads paralelas
    # Cada thread foca-se numa escala diferente (Bit -> Byte -> Word)
    dots = strobe_engine.strobe_flash(qwen_binary_siphon, soul_state)
    
    # Log de Harmonics (Densidade Narrativa)
    log_narrative_density(len(dots), current_time)
    
    
# No teu main loop
storage = HyperspaceClient()

# Quando o strobe deteta ressonância:
metadata = {
    "scale": str(current_scale),
    "timestamp": str(datetime.now().timestamp()),
    "source": "Qwen-Siphon"
}

success = storage.ingest_coherent_dot(soul.id, current_vector, metadata)
if success:
    print("🧠 Memória fractal arquivada na HyperspaceDB.")