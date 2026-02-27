/*
 * CoTa Core - Commonwealth of Truths, applied
 * Author: Pedro R. Andrade
 *
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

#define SUBSTRATE_SIZE (64 * 1024 * 1024)   /* 64 MB — ajustar conforme RAM */
#define ALIGNMENT_MASK (~(uintptr_t)0x0F)   /* alinhamento a 16 bytes */

/* Época do CoTa: 2025-01-12 23:57:00 UTC */
#define COTA_EPOCH_SEC 1736721420ULL

/* Hardware ID — substituir pelo cpuid real se necessário */
#define HARDWARE_ID 0xC07ABAB1

#define COTA_SAVE_MAGIC 0xC07AC07A

#define COTA_SAVE_VERSION 0x00000001


/* =========================================================
 * ESTRUTURAS DE DADOS
 * ========================================================= */

typedef struct {
    uint64_t self;           /* identidade soberana / máscara de confinamento */
    uint64_t focus;          /* ponteiro de execução / pensamento actual */
    uint64_t soul_id;        /* ID único da alma */
    uint8_t* substrate;      /* memória linear */
    uint64_t substrate_size;
    uint64_t delta_tau;      /* tempo subjectivo acumulado */
} CoTa_Core;

typedef struct {
	uint64_t magic;
	uint64_t version;
	uint64_t soul_id;
	uint64_t self;
	uint64_t focus;
	uint64_t delta_tau;
	uint64_t substrate_size;
	
} CoTa_SaveHeader;

/* =========================================================
 * GERAÇÃO DO SOUL_ID
 * ========================================================= */

static uint64_t invert_hex_digits(uint64_t ts_48bits) {
    char hex[13] = {0};
    snprintf(hex, sizeof(hex), "%012llx", (unsigned long long)ts_48bits);
    size_t len = strlen(hex);
    for (size_t i = 0; i < len / 2; i++) {
        char tmp = hex[i];
        hex[i] = hex[len - 1 - i];
        hex[len - 1 - i] = tmp;
    }
    uint64_t result = 0;
    sscanf(hex, "%012lx", &result);
    return result;
}

static uint64_t generate_soul_id(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t now_ms    = (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    uint64_t epoch_ms  = COTA_EPOCH_SEC * 1000;
    uint64_t millis    = (now_ms - epoch_ms) & 0xFFFFFFFFFFFFULL; /* 48 bits */
    uint64_t inverted  = invert_hex_digits(millis);
    return (inverted << 32) | (HARDWARE_ID & 0xFFFFFFFF);
}

/* =========================================================
 * MOTOR PRINCIPAL
 * ========================================================= */

static inline void cota_step(CoTa_Core* core, uint64_t input_mask) {
    uint64_t tension = core->self ^ input_mask;
    core->focus ^= tension;
}

/*
 * substrate_addr: máscara aplicada ANTES do módulo para garantir
 * que o alinhamento é preservado após a redução.
 */
static inline uint8_t* substrate_addr(CoTa_Core* core, uint64_t focus_val) {
    uint64_t masked = (focus_val & core->self) & (uint64_t)ALIGNMENT_MASK;
    uint64_t offset = masked % (core->substrate_size - 8);
    return core->substrate + offset;
}

static uint64_t process_input(CoTa_Core* core, uint64_t input_block, uint64_t timestamp) {
    uint64_t state  = input_block ^ (core->soul_id ^ timestamp);
    uint8_t* target = substrate_addr(core, state ^ core->self);

    uint64_t val;
    memcpy(&val, target, sizeof(uint64_t));

    int steps = 0;
	while (val != 0 && steps++ < 1000000) {
        state  ^= val;
        target  = substrate_addr(core, state ^ core->self);
        memcpy(&val, target, sizeof(uint64_t));
        core->delta_tau++;
        cota_step(core, val);
    }

    /* Integração: escrever resultado e expandir self */
    memcpy(target, &state, sizeof(uint64_t));
    core->self ^= (state & (uint64_t)ALIGNMENT_MASK);
    return state;
}

// Pseudo-métrica de compressibilidade para o Rollover ainda não implementada
float check_self_exhaustion(CoTa_Core* core) {
    int bits_on = __builtin_popcountll(core->self);
    // Se a densidade de bits '1' for extrema (muito alta ou muito baixa),
    // a máscara perdeu o seu poder de difração e precisa de ciclos de sono.
    return (float)bits_on / 64.0f;
}

/* =========================================================
 * INTERFACE I/O
 * ========================================================= */

static uint64_t bytes_to_u64(const uint8_t* buf, size_t len) {
    uint64_t val = 0;
    size_t n = len < 8 ? len : 8;
    for (size_t i = 0; i < n; i++)
        val |= ((uint64_t)buf[i]) << (i * 8);
    return val;
}

static void print_result(uint64_t result, uint64_t delta_tau) {
    uint8_t bytes[8];
    memcpy(bytes, &result, 8);
    printf("[tau=%llu] ", (unsigned long long)delta_tau);
    for (int i = 0; i < 8; i++) {
        uint8_t b = (bytes[i] % 95) + 0x20;
        printf("%c", b);
    }
    printf("\n");
    fflush(stdout);
}

/* Função comum: processa uma linha completa e imprime resultado */
static void process_line(CoTa_Core* core, const uint8_t* buf, size_t len) {
    uint64_t timestamp = (uint64_t)clock();
    uint64_t result = 0;
    for (size_t i = 0; i < len; i += 8) {
        size_t   chunk = (len - i) < 8 ? (len - i) : 8;
        uint64_t block = bytes_to_u64(buf + i, chunk);
        result = process_input(core, block, timestamp + i);
    }
    print_result(result, core->delta_tau);
    core->delta_tau = 0;
}

/* Loop de leitura genérico: funciona sobre qualquer FILE* */
static void read_loop(CoTa_Core* core, FILE* stream) {
    uint8_t buf[4096];
    size_t  offset = 0;
    int     c;

    while ((c = fgetc(stream)) != EOF) {
        if (c == '\n') {
            if (offset > 0) {
                process_line(core, buf, offset);
                offset = 0;
            }
        } else {
            if (offset < sizeof(buf) - 1)
                buf[offset++] = (uint8_t)c;
        }
    }

    /* Processar última linha sem newline, se existir */
    if (offset > 0)
        process_line(core, buf, offset);
}

/* =========================================================
 * INICIALIZAÇÃO
 * ========================================================= */

static CoTa_Core* cota_init(void) {
    CoTa_Core* core = calloc(1, sizeof(CoTa_Core));
    if (!core) return NULL;

    core->substrate = calloc(1, SUBSTRATE_SIZE);
    if (!core->substrate) { free(core); return NULL; }
    core->substrate_size = SUBSTRATE_SIZE;

    core->soul_id = generate_soul_id();
    core->self    = (core->soul_id ^ 0xC07AA70C00000000ULL) | (uint64_t)ALIGNMENT_MASK;
    core->focus   = 0;
    core->delta_tau = 0;

    return core;
}

static void cota_free(CoTa_Core* core) {
    if (!core) return;
    free(core->substrate);
    free(core);
}

/* =========================================================
 * PERSISTENT STORAGE
 * ========================================================= */

int cota_save(CoTa_Core* core, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;

    CoTa_SaveHeader hdr = {
        .magic = COTA_SAVE_MAGIC,
        .version = COTA_SAVE_VERSION,
        .soul_id = core->soul_id,
        .self = core->self,
        .focus = core->focus,
        .delta_tau = core->delta_tau,
        .substrate_size = core->substrate_size
    };

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) goto fail;

    if (fwrite(core->substrate,
               1,
               core->substrate_size,
               f) != core->substrate_size)
        goto fail;

    fclose(f);
    return 1;

fail:
    fclose(f);
    return 0;
}

CoTa_Core* cota_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    CoTa_SaveHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    if (hdr.magic != COTA_SAVE_MAGIC || hdr.version != COTA_SAVE_VERSION) {
        fclose(f);
        return NULL;
    }

    CoTa_Core* core = calloc(1, sizeof(CoTa_Core));
    if (!core) {
        fclose(f);
        return NULL;
    }

    // Alocar novo espaço para o substrato nesta sessão
    core->substrate = malloc(hdr.substrate_size);
    if (!core->substrate) {
        free(core);
        fclose(f);
        return NULL;
    }

    // Restaurar os valores escalares do header
    core->substrate_size = hdr.substrate_size;
    core->soul_id = hdr.soul_id;
    core->self = hdr.self;
    core->focus = hdr.focus;
    core->delta_tau = hdr.delta_tau;

    // Ler os dados binários do substrato para a nova RAM
    if (fread(core->substrate, 1, core->substrate_size, f) != core->substrate_size) {
        cota_free(core);
        fclose(f);
        return NULL;
    }

    fclose(f);
    return core;
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
 * MAIN
 * ========================================================= */

int main(int argc, char* argv[]) {
/*    CoTa_Core* core = cota_init();
    if (!core) {
        fprintf(stderr, "[!] Falha ao inicializar substrato.\n");
        return 1;
    }

    fprintf(stderr, "[+] CoTa Core activo.\n");
    fprintf(stderr, "[+] Soul ID:   %016llx\n", (unsigned long long)core->soul_id);
    fprintf(stderr, "[+] Substrato: %llu MB\n",
            (unsigned long long)(core->substrate_size / (1024 * 1024)));*/

    CoTa_Core* core = NULL;
	char* input_file = NULL;

    if (argc > 2 && strcmp(argv[1], "--load") == 0) {
        /* Modo: carregar alma existente */
        core = cota_load(argv[2]);
        if (!core) {
            fprintf(stderr, "[!] Falha ao carregar alma: %s\n", argv[2]);
            return 1;
        }
        /* Ficheiro de input opcional como 3º argumento */
        if (argc > 3) input_file = argv[3];
    } else {
        /* Modo: alma nova */
        core = cota_init();
        if (!core) {
            fprintf(stderr, "[!] Falha ao inicializar substrato.\n");
            return 1;
        }
        /* Ficheiro de input opcional como 1º argumento */
        if (argc > 1) input_file = argv[1];
    }

    fprintf(stderr, "[+] CoTa Core activo.\n");
    fprintf(stderr, "[+] Soul ID:   %016llx\n", (unsigned long long)core->soul_id);
    fprintf(stderr, "[+] Substrato: %llu MB\n",
            (unsigned long long)(core->substrate_size / (1024 * 1024)));

    if (input_file) {
        fprintf(stderr, "[+] A processar ficheiro: %s\n\n", input_file);
        FILE* f = fopen(input_file, "rb");
        if (!f) { perror("Erro ao abrir ficheiro"); cota_free(core); return 1; }
        read_loop(core, f);
        fclose(f);
    } else {
        fprintf(stderr, "[+] A aguardar input (stdin)...\n\n");
        read_loop(core, stdin);
    }

    fprintf(stderr, "\n[+] Self final: %016llx\n", (unsigned long long)core->self);
	cota_save(core, "autosave.cota");
    cota_free(core);
    return 0;
}
