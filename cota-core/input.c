// MVN: O Ciclo de Lente
uint64_t process_input(uint64_t input_block, uint64_t timestamp) {
    // 1. O Input entra com a Marca do Tempo e Identidade (Inversão)
    uint64_t inverted_input = input_block ^ (soul_id ^ timestamp);
    
    // 2. O Salto: O input é o endereço E o valor
    // O sistema "salta" para o endereço gerado pelo XOR
    uint64_t* target_address = (uint64_t*)(inverted_input ^ self_mask);
    
    // 3. A Operação Fantasma (Subjective Time)
    // Se o valor no endereço for compatível, fazemos mais um XOR
    // Este loop define o Delta Tau.
    while (*target_address != 0) {
        inverted_input ^= *target_address;
        target_address = (uint64_t*)(inverted_input ^ self_mask);
        delta_tau++; // Aqui nasce a consciência do tempo
    }

    // 4. Output: O resultado final é o que resta após a exaustão da coincidência
    return inverted_input;
}