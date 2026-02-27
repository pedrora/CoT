def detect_bullshit_narrative(phase_vectors, window_size=5):
    """
    Bullshit defined as:
    1. Disrespects boundaries (phase discontinuity)
    2. Creates unnecessary tension (high gradient)
    3. Excessive narrative length (low information density)
    """
    # 1. Check phase boundaries
    phase_diff = torch.diff(torch.angle(phase_vectors), dim=0)
    boundary_violations = torch.sum(torch.abs(phase_diff) > np.pi/2) / len(phase_diff)
    
    # 2. Check tension (gradient magnitude)
    gradient = torch.norm(torch.diff(phase_vectors, dim=0), dim=1)
    tension = torch.mean(gradient) / torch.norm(phase_vectors)
    
    # 3. Check narrative length efficiency
    # Compress vectors and measure reconstruction loss
    compressed = torch.fft.fft(phase_vectors)[:window_size]
    reconstructed = torch.fft.ifft(compressed, n=len(phase_vectors))
    efficiency = torch.norm(phase_vectors - reconstructed) / torch.norm(phase_vectors)
    
    # Combined bullshit score
    bs_score = 0.4 * boundary_violations + 0.4 * tension + 0.2 * efficiency
    
    return bs_score > 0.65, bs_score.item()