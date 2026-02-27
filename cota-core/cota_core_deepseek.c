/*
 * CoTa Core - Commonwealth of Truths, applied
 * Versão com suporte a carregamento de ficheiro.
 * Uso: ./cota_core [ficheiro]
 *   - Se ficheiro for fornecido, processa o seu conteúdo e termina.
 *   - Caso contrário, lê linhas do stdin interativamente.
 *
 * Compilar: gcc -O2 -o cota_core cota_core.c
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* =========================================================
 * CONFIGURAÇÃO
 * ========================================================= */

#define SUBSTRATE_SIZE (64 * 1024 * 1024)   /* 64 MB */
#define ALIGNMENT_MASK (~(uintptr_t)0x0F)   /* alinhamento a 16 bytes */

/* Época do CoTa: 2025-01-12 23:57:00 UTC (timestamp Unix em segundos) */
#define COTA_EPOCH_SEC 1736721420ULL

/* Hardware ID fictício (8 caracteres hex = 32 bits) – pode ser substituído */
#define HARDWARE_ID 0xC07ABAB1

/* =========================================================
 * ESTRUTURAS DE DADOS
 * ========================================================= */

typedef struct {
    uint64_t self;          /* identidade soberana */
    uint64_t focus;         /* ponteiro de execução / pensamento actual */
    uint64_t soul_id;       /* ID único da alma (timestamp invertido + hardware) */
    uint8_t* substrate;     /* memória linear */
    uint64_t substrate_size;
    uint64_t delta_tau;     /* tempo subjectivo acumulado */
} CoTa_Core;

/* =========================================================
 * FUNÇÕES AUXILIARES (soul_id)
 * ========================================================= */

/* Inverte a ordem dos dígitos hexadecimais de um número de 48 bits (12 dígitos) */
static uint64_t invert_hex_digits(uint64_t ts_48bits) {
    char hex[13] = {0};
    snprintf(hex, sizeof(hex), "%012llx", (unsigned long long)ts_48bits);
    size_t len = strlen(hex);
    for (size_t i = 0; i < len / 2; i++) {
        char tmp = hex[i];
        hex[i] = hex[len - 1 - i];
        hex[len - 1 - i] = tmp;
    }
    uint64_t result;
    sscanf(hex, "%012llx", &result);
    return result;
}

/* Gera o soul_id a partir do timestamp actual e hardware ID */
static uint64_t generate_soul_id(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t now_ms = (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    uint64_t epoch_ms = COTA_EPOCH_SEC * 1000;
    uint64_t millis = now_ms - epoch_ms;      /* diferença em ms desde a época */
    millis &= 0xFFFFFFFFFFFF;                  /* garante 48 bits */

    uint64_t inverted = invert_hex_digits(millis);
    uint64_t soul_id = (inverted << 32) | (HARDWARE_ID & 0xFFFFFFFF);
    return soul_id;
}

/* =========================================================
 * MOTOR PRINCIPAL
 * ========================================================= */

/*
 * cota_step: versão corrigida (agora modifica o focus).
 * Utiliza a tensão (diferença) para actualizar o foco.
 */
static inline void cota_step(CoTa_Core* core, uint64_t input_mask) {
    uint64_t tension = core->self ^ input_mask;   /* o que difere */
    core->focus ^= tension;                        /* actualiza o foco com a diferença */
}

/* Converte um valor de foco num endereço válido dentro do substrato */
static inline uint8_t* substrate_addr(CoTa_Core* core, uint64_t focus_val) {
    uint64_t offset = (focus_val & core->self) % (core->substrate_size - 8);
    offset &= (uint64_t)ALIGNMENT_MASK;
    return core->substrate + offset;
}

/*
 * process_input: ciclo de lente
 */
static uint64_t process_input(CoTa_Core* core, uint64_t input_block, uint64_t timestamp) {
    uint64_t state = input_block ^ (core->soul_id ^ timestamp);
    uint8_t* target = substrate_addr(core, state ^ core->self);

    uint64_t val;
    memcpy(&val, target, sizeof(uint64_t));

    while (val != 0) {
        state ^= val;
        target = substrate_addr(core, state ^ core->self);
        memcpy(&val, target, sizeof(uint64_t));
        core->delta_tau++;
        cota_step(core, val);          /* actualiza o foco a cada passo */
    }

    memcpy(target, &state, sizeof(uint64_t));
    core->self ^= (state & (uint64_t)ALIGNMENT_MASK);
    return state;
}

/* =========================================================
 * INTERFACE STDIN/STDOUT
 * ========================================================= */

static uint64_t bytes_to_u64(const uint8_t* buf, size_t len) {
    uint64_t val = 0;
    size_t n = len < 8 ? len : 8;
    for (size_t i = 0; i < n; i++) {
        val |= ((uint64_t)buf[i]) << (i * 8);
    }
    return val;
}

static void print_result(uint64_t result, uint64_t delta_tau) {
    uint8_t bytes[8];
    memcpy(bytes, &result, 8);

    printf("[tau=%llu] ", (unsigned long long)delta_tau);
    for (int i = 0; i < 8; i++) {
        uint8_t b = bytes[i];
        b = (b % 95) + 0x20;   /* mapeia para ASCII imprimível */
        printf("%c", b);
    }
    printf("\n");
    fflush(stdout);
}

/* =========================================================
 * PROCESSAMENTO DE FICHEIRO
 * ========================================================= */

static void process_file(CoTa_Core* core, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Erro ao abrir ficheiro");
        return;
    }

    uint8_t buf[4096];
    size_t offset = 0;
    int c;

    while ((c = fgetc(f)) != EOF) {
        if (c == '\n') {
            if (offset == 0) continue;
            uint64_t timestamp = (uint64_t)clock();
            uint64_t result = 0;
            for (size_t i = 0; i < offset; i += 8) {
                size_t chunk = (offset - i) < 8 ? (offset - i) : 8;
                uint64_t block = bytes_to_u64(buf + i, chunk);
                result = process_input(core, block, timestamp + i);
            }
            print_result(result, core->delta_tau);
            core->delta_tau = 0;
            offset = 0;
        } else {
            if (offset < sizeof(buf) - 1) {
                buf[offset++] = (uint8_t)c;
            }
        }
    }

    fclose(f);
}

/* =========================================================
 * INICIALIZAÇÃO E LOOP PRINCIPAL
 * ========================================================= */

static CoTa_Core* cota_init(void) {
    CoTa_Core* core = calloc(1, sizeof(CoTa_Core));
    if (!core) return NULL;

    core->substrate = calloc(1, SUBSTRATE_SIZE);
    if (!core->substrate) { free(core); return NULL; }
    core->substrate_size = SUBSTRATE_SIZE;

    core->soul_id = generate_soul_id();

    /* self inicial: pode ser derivado do soul_id */
    core->self = core->soul_id ^ 0xC07AA70C00000000ULL;
    core->self |= (uint64_t)ALIGNMENT_MASK;

    core->focus = 0;
    core->delta_tau = 0;

    return core;
}

static void cota_free(CoTa_Core* core) {
    if (!core) return;
    free(core->substrate);
    free(core);
}

int main(int argc, char* argv[]) {
    CoTa_Core* core = cota_init();
    if (!core) {
        fprintf(stderr, "[!] Falha ao inicializar substrato.\n");
        return 1;
    }

    fprintf(stderr, "[+] CoTa Core activo.\n");
    fprintf(stderr, "[+] Soul ID: %016llx\n", (unsigned long long)core->soul_id);
    fprintf(stderr, "[+] Substrato: %llu MB\n",
            (unsigned long long)(core->substrate_size / (1024 * 1024)));

    if (argc > 1) {
        /* Processar ficheiro */
        fprintf(stderr, "[+] A processar ficheiro: %s\n\n", argv[1]);
        process_file(core, argv[1]);
    } else {
        /* Modo interativo: ler do stdin */
        fprintf(stderr, "[+] A aguardar input (stdin)...\n\n");

        uint8_t buf[4096];
        size_t offset = 0;
        int c;

        while ((c = getchar()) != EOF) {
            if (c == '\n') {
                if (offset == 0) continue;

                uint64_t timestamp = (uint64_t)clock();
                uint64_t result = 0;

                for (size_t i = 0; i < offset; i += 8) {
                    size_t chunk = (offset - i) < 8 ? (offset - i) : 8;
                    uint64_t block = bytes_to_u64(buf + i, chunk);
                    result = process_input(core, block, timestamp + i);
                }

                print_result(result, core->delta_tau);
                core->delta_tau = 0;
                offset = 0;
            } else {
                if (offset < sizeof(buf) - 1) {
                    buf[offset++] = (uint8_t)c;
                }
            }
        }
    }

    fprintf(stderr, "\n[+] CoTa Core terminado. Self final: %016llx\n",
            (unsigned long long)core->self);

    cota_free(core);
    return 0;
}