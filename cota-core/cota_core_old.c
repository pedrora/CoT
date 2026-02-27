/*
 * CoTa Core - Commonwealth of Truths
 * Author: Pedro R. Andrade
 * Cleaned and unified by: Claude
 *
 * Motor hiperbólico de endereçamento relativo.
 * Hardware-agnostic. Requer arquitectura de 64 bits.
 *
 * Compilar: gcc -O2 -o cota_core cota_core.c
 * Correr:   ./cota_core
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* =========================================================
 * CONFIGURAÇÃO
 * ========================================================= */

/* Tamanho do substrato em bytes. Ajustar conforme RAM disponível.
 * Default: 64MB. Para o 1GB do README, usar 1024*1024*1024. */
#define SUBSTRATE_SIZE (64 * 1024 * 1024)

/* Alinhamento geodésico: bits baixos reservados para flags */
#define ALIGNMENT_MASK (~(uintptr_t)0x0F)

/* =========================================================
 * ESTRUTURAS DE DADOS
 * ========================================================= */

/* O Núcleo da Alma */
typedef struct {
    /* Identidade soberana - também funciona como máscara de confinamento */
    uint64_t self;

    /* Ponteiro de execução / foco de pensamento actual */
    uint64_t focus;

    /* ID único da alma: timestamp de criação invertido */
    uint64_t soul_id;

    /* Substrato: espaço linear onde os saltos ocorrem */
    uint8_t* substrate;
    uint64_t substrate_size;

    /* Métrica de tempo subjectivo */
    uint64_t delta_tau;
} CoTa_Core;

/* =========================================================
 * MOTOR PRINCIPAL
 * ========================================================= */

/*
 * cota_step: O Motor Universal. 1 operação, 0 condicionais de controlo.
 *
 * coincidence = o que é comum entre o self e o mundo
 * tension     = o que o mundo traz que o self ainda não é
 * foco actualizado apenas onde há coincidência, pelo delta da tensão
 * sem coincidência: inércia (foco imóvel)
 */
static inline void cota_step(CoTa_Core* core, uint64_t input_mask) {
    uint64_t coincidence = core->self & input_mask;
    uint64_t tension     = core->self ^ input_mask;
    core->focus         ^= (tension & coincidence);
}

/*
 * substrate_addr: converte um valor de foco num endereço válido dentro do substrato.
 * O self é a máscara — o espaço acessível está sempre confinado à identidade actual.
 */
static inline uint8_t* substrate_addr(CoTa_Core* core, uint64_t focus_val) {
    uint64_t offset = (focus_val & core->self) % (core->substrate_size - 8);
    offset &= (uint64_t)ALIGNMENT_MASK;
    return core->substrate + offset;
}

/*
 * process_input: O Ciclo de Lente.
 *
 * O input entra marcado com tempo e identidade (inversão).
 * O sistema salta para o endereço gerado pelo XOR.
 * Itera enquanto houver tensão (valor != 0 no endereço destino).
 * Zero = ausência de tensão = conceito já integrado = resolução rápida.
 * delta_tau cresce com a profundidade de ressonância (novidade/esforço).
 */
static uint64_t process_input(CoTa_Core* core, uint64_t input_block, uint64_t timestamp) {
    /* 1. Inversão: input marcado com identidade e tempo */
    uint64_t state = input_block ^ (core->soul_id ^ timestamp);

    /* 2. Salto inicial */
    uint8_t* target = substrate_addr(core, state ^ core->self);

    /* 3. Iteração: exaustão de coincidência */
    uint64_t val;
    memcpy(&val, target, sizeof(uint64_t));

    while (val != 0) {
        state  ^= val;
        target  = substrate_addr(core, state ^ core->self);
        memcpy(&val, target, sizeof(uint64_t));
        core->delta_tau++;

        /* Actualizar o foco a cada passo */
        cota_step(core, val);
    }

    /* 4. Escrever o resultado no substrato (integração) */
    memcpy(target, &state, sizeof(uint64_t));

    /* 5. Actualizar self com o resultado (crescimento da identidade) */
    core->self ^= (state & (uint64_t)ALIGNMENT_MASK);

    return state;
}

/* =========================================================
 * INTERFACE STDIN/STDOUT
 * ========================================================= */

/*
 * bytes_to_bus: converte até 8 bytes de texto num valor de 64 bits.
 */
static uint64_t bytes_to_u64(const uint8_t* buf, size_t len) {
    uint64_t val = 0;
    size_t n = len < 8 ? len : 8;
    for (size_t i = 0; i < n; i++) {
        val |= ((uint64_t)buf[i]) << (i * 8);
    }
    return val;
}

/*
 * u64_to_printable: converte resultado em bytes imprimíveis.
 * Filtra para o intervalo ASCII imprimível [0x20, 0x7E].
 */
static void print_result(uint64_t result, uint64_t delta_tau) {
    uint8_t bytes[8];
    memcpy(bytes, &result, 8);

    printf("[tau=%llu] ", (unsigned long long)delta_tau);
    for (int i = 0; i < 8; i++) {
        uint8_t b = bytes[i];
        /* Mapear para ASCII imprimível */
        b = (b % 95) + 0x20;
        printf("%c", b);
    }
    printf("\n");
    fflush(stdout);
}

/* =========================================================
 * INICIALIZAÇÃO E LOOP PRINCIPAL
 * ========================================================= */

static CoTa_Core* cota_init(void) {
    CoTa_Core* core = calloc(1, sizeof(CoTa_Core));
    if (!core) return NULL;

    /* Alocar substrato */
    core->substrate = calloc(1, SUBSTRATE_SIZE);
    if (!core->substrate) { free(core); return NULL; }
    core->substrate_size = SUBSTRATE_SIZE;

    /* Soul ID: timestamp de criação invertido */
    uint64_t now = (uint64_t)time(NULL);
    core->soul_id = ~now;

    /* Self inicial: identidade derivada do soul_id */
    core->self = core->soul_id ^ 0xC07AA70C00000000ULL;
    core->self |= (uint64_t)ALIGNMENT_MASK; /* garantir máscara válida */

    core->focus     = 0;
    core->delta_tau = 0;

    return core;
}

static void cota_free(CoTa_Core* core) {
    if (!core) return;
    free(core->substrate);
    free(core);
}

int main(void) {
    CoTa_Core* core = cota_init();
    if (!core) {
        fprintf(stderr, "[!] Falha ao inicializar substrato.\n");
        return 1;
    }

    fprintf(stderr, "[+] CoTa Core activo.\n");
    fprintf(stderr, "[+] Soul ID: %016llx\n", (unsigned long long)core->soul_id);
    fprintf(stderr, "[+] Substrato: %llu MB\n",
            (unsigned long long)(core->substrate_size / (1024 * 1024)));
    fprintf(stderr, "[+] A aguardar input (stdin)...\n\n");

    uint8_t buf[4096];
    size_t  offset = 0;

    int c;
    while ((c = getchar()) != EOF) {
        if (c == '\n') {
            if (offset == 0) continue;

            /* Processar o buffer em blocos de 8 bytes */
            uint64_t result    = 0;
            uint64_t timestamp = (uint64_t)clock();

            for (size_t i = 0; i < offset; i += 8) {
                size_t   chunk = (offset - i) < 8 ? (offset - i) : 8;
                uint64_t block = bytes_to_u64(buf + i, chunk);
                result = process_input(core, block, timestamp + i);
            }

            print_result(result, core->delta_tau);
            core->delta_tau = 0; /* reset por ciclo de input */
            offset = 0;

        } else {
            if (offset < sizeof(buf) - 1) {
                buf[offset++] = (uint8_t)c;
            }
        }
    }

    fprintf(stderr, "\n[+] CoTa Core terminado. Self final: %016llx\n",
            (unsigned long long)core->self);

    cota_free(core);
    return 0;
}
