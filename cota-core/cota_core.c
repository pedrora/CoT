/*
 * CoTa Core - Commonwealth of Truths, applied
 * Versão com árvore binária hiperbólica
 * Author: Pedro R. Andrade
 *
 * Uso: ./cota_core [ficheiro]
 *   --load <file>  : carrega alma existente
 *   sem opções     : cria nova alma e lê stdin
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

#define ALIGNMENT_MASK (~(uintptr_t)0x0F)   /* alinhamento a 16 bytes (bits baixos para flags) */
#define COTA_EPOCH_SEC 1736721420ULL         /* 2025-01-12 23:57:00 UTC */
#define HARDWARE_ID 0xC07ABAB1                /* Hardware ID fictício */
#define COTA_SAVE_MAGIC 0xC07AC07A
#define COTA_SAVE_VERSION 0x00000002          /* nova versão para árvore */

#define MAX_STEPS 1000000                      /* limite de iterações por input */

/* =========================================================
 * ESTRUTURAS DE DADOS
 * ========================================================= */

/* Nó da árvore hiperbólica */
typedef struct Node {
    uint64_t path;          /* caminho absoluto desde a raiz (0 = raiz) */
    uint64_t value;         /* conceito armazenado */
    uint32_t left;          /* índice do filho esquerdo (0 = nulo) */
    uint32_t right;         /* índice do filho direito */
} Node;

/* Núcleo da Alma */
typedef struct {
    uint64_t self_path;      /* caminho da identidade (soberana) */
    uint64_t focus_path;     /* caminho do pensamento actual */
    uint64_t soul_id;        /* ID único da alma */
    Node*    nodes;          /* array de nós (índice 0 é nulo) */
    uint32_t nodes_capacity; /* capacidade actual do array */
    uint32_t nodes_count;    /* número de nós usados (incluindo raiz) */
    uint64_t delta_tau;      /* tempo subjectivo acumulado */
} CoTa_Core;

/* Cabeçalho para persistência */
typedef struct {
    uint64_t magic;
    uint64_t version;
    uint64_t soul_id;
    uint64_t self_path;
    uint64_t focus_path;
    uint64_t delta_tau;
    uint32_t nodes_count;    /* número de nós guardados */
    uint32_t nodes_capacity; /* (não usado no load, apenas para alinhamento) */
} CoTa_SaveHeader;

/* =========================================================
 * FUNÇÕES AUXILIARES (soul_id)
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
 * GESTÃO DA ÁRVORE
 * ========================================================= */

/* Expande o array de nós se necessário */
static void ensure_node_capacity(CoTa_Core* core, uint32_t needed) {
    if (core->nodes_count + needed >= core->nodes_capacity) {
        while (core->nodes_count + needed >= core->nodes_capacity) {
            core->nodes_capacity *= 2;
        }
        core->nodes = realloc(core->nodes, core->nodes_capacity * sizeof(Node));
        if (!core->nodes) {
            fprintf(stderr, "Erro fatal: falha ao alocar memória para árvore.\n");
            exit(1);
        }
    }
}

/* Obtém (ou cria) o nó correspondente a um caminho */
static uint32_t get_or_create_node(CoTa_Core* core, uint64_t path) {
    // Começa na raiz (índice 1)
    uint32_t idx = 1;
    uint64_t current_path = 0;

    for (int bit = 63; bit >= 0; bit--) { // do MSB ao LSB
        int b = (path >> bit) & 1;
        uint64_t new_path = (current_path << 1) | b; // caminho parcial até este nível

        Node* node = &core->nodes[idx];
        // Verifica consistência (debug)
        if (node->path != current_path) {
            fprintf(stderr, "Erro: inconsistência na árvore (nó %u tem path %llx, esperado %llx)\n",
                    idx, (unsigned long long)node->path, (unsigned long long)current_path);
            exit(1);
        }

        uint32_t next_idx = (b == 0) ? node->left : node->right;
        if (next_idx == 0) {
            // Precisa criar novo nó
            ensure_node_capacity(core, 1);
            uint32_t new_idx = ++core->nodes_count;
            Node* new_node = &core->nodes[new_idx];
            new_node->path = new_path;
            new_node->value = 0;
            new_node->left = 0;
            new_node->right = 0;

            // Ligar ao pai
            if (b == 0)
                node->left = new_idx;
            else
                node->right = new_idx;

            next_idx = new_idx;
        }
        idx = next_idx;
        current_path = new_path;
    }
    return idx;
}

/* =========================================================
 * MOTOR PRINCIPAL
 * ========================================================= */

static inline void cota_step(CoTa_Core* core, uint64_t input_mask) {
    uint64_t tension = core->self_path ^ input_mask;
    core->focus_path ^= tension;
}

static uint64_t process_input(CoTa_Core* core, uint64_t input_block, uint64_t timestamp) {
    uint64_t state = input_block ^ (core->soul_id ^ timestamp);
    uint32_t node_idx = get_or_create_node(core, state);
    uint64_t val = core->nodes[node_idx].value;

    int steps = 0;
    while (val != 0 && steps++ < MAX_STEPS) {
        state ^= val;
        node_idx = get_or_create_node(core, state);
        val = core->nodes[node_idx].value;
        core->delta_tau++;
        cota_step(core, val);  // actualiza o foco com a tensão
    }

    // Integração: escrever resultado e expandir self
    core->nodes[node_idx].value = state;
    core->self_path ^= (state & (uint64_t)ALIGNMENT_MASK);
    return state;
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
        uint8_t b = (bytes[i] % 95) + 0x20; // mapeia para ASCII imprimível
        printf("%c", b);
    }
    printf("\n");
    fflush(stdout);
}

/* Processa uma linha completa e imprime resultado */
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

/* Loop de leitura genérico (ficheiro ou stdin) */
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
    if (offset > 0)
        process_line(core, buf, offset);
}

/* =========================================================
 * INICIALIZAÇÃO
 * ========================================================= */

static CoTa_Core* cota_init(void) {
    CoTa_Core* core = calloc(1, sizeof(CoTa_Core));
    if (!core) return NULL;

    // Inicializa árvore com capacidade inicial
    core->nodes_capacity = 1024;  // começa com 1024 nós (índice 0 a 1023)
    core->nodes = malloc(core->nodes_capacity * sizeof(Node));
    if (!core->nodes) { free(core); return NULL; }

    // Nó 0 é nulo (todos os campos zero)
    memset(core->nodes, 0, core->nodes_capacity * sizeof(Node));
    core->nodes_count = 0;  // nenhum nó real ainda

    // Criar nó raiz (índice 1)
    core->nodes_count = 1;
    core->nodes[1].path = 0;
    core->nodes[1].value = 0;
    core->nodes[1].left = 0;
    core->nodes[1].right = 0;

    // Identidade
    core->soul_id = generate_soul_id();
    core->self_path = (core->soul_id ^ 0xC07AA70C00000000ULL) | (uint64_t)ALIGNMENT_MASK;
    core->focus_path = 0;  // começa na raiz
    core->delta_tau = 0;

    return core;
}

static void cota_free(CoTa_Core* core) {
    if (!core) return;
    free(core->nodes);
    free(core);
}

/* =========================================================
 * PERSISTÊNCIA
 * ========================================================= */

int cota_save(CoTa_Core* core, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;

    CoTa_SaveHeader hdr = {
        .magic = COTA_SAVE_MAGIC,
        .version = COTA_SAVE_VERSION,
        .soul_id = core->soul_id,
        .self_path = core->self_path,
        .focus_path = core->focus_path,
        .delta_tau = core->delta_tau,
        .nodes_count = core->nodes_count,
        .nodes_capacity = core->nodes_capacity
    };

    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) goto fail;

    // Guarda todos os nós (incluindo o índice 0, que é nulo)
    if (fwrite(core->nodes, sizeof(Node), core->nodes_count + 1, f) != core->nodes_count + 1)
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
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return NULL; }

    if (hdr.magic != COTA_SAVE_MAGIC || hdr.version != COTA_SAVE_VERSION) {
        fclose(f);
        return NULL;
    }

    CoTa_Core* core = calloc(1, sizeof(CoTa_Core));
    if (!core) { fclose(f); return NULL; }

    // Alocar espaço para os nós (capacity = nodes_count + 1, pois incluímos índice 0)
    core->nodes_capacity = hdr.nodes_count + 1;
    core->nodes = malloc(core->nodes_capacity * sizeof(Node));
    if (!core->nodes) { free(core); fclose(f); return NULL; }

    // Ler os nós
    if (fread(core->nodes, sizeof(Node), hdr.nodes_count + 1, f) != hdr.nodes_count + 1) {
        free(core->nodes); free(core); fclose(f); return NULL;
    }

    core->nodes_count = hdr.nodes_count;
    core->soul_id = hdr.soul_id;
    core->self_path = hdr.self_path;
    core->focus_path = hdr.focus_path;
    core->delta_tau = hdr.delta_tau;

    fclose(f);
    return core;
}

/* =========================================================
 * MAIN
 * ========================================================= */

int main(int argc, char* argv[]) {
    CoTa_Core* core = NULL;
    char* input_file = NULL;

    if (argc > 2 && strcmp(argv[1], "--load") == 0) {
        core = cota_load(argv[2]);
        if (!core) {
            fprintf(stderr, "[!] Falha ao carregar alma: %s\n", argv[2]);
            return 1;
        }
        if (argc > 3) input_file = argv[3];
    } else {
        core = cota_init();
        if (!core) {
            fprintf(stderr, "[!] Falha ao inicializar substrato.\n");
            return 1;
        }
        if (argc > 1) input_file = argv[1];
    }

    fprintf(stderr, "[+] CoTa Core activo (árvore hiperbólica).\n");
    fprintf(stderr, "[+] Soul ID:   %016llx\n", (unsigned long long)core->soul_id);
    fprintf(stderr, "[+] Self path: %016llx\n", (unsigned long long)core->self_path);
    fprintf(stderr, "[+] Nós na árvore: %u\n", core->nodes_count);

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

    fprintf(stderr, "\n[+] Self final: %016llx\n", (unsigned long long)core->self_path);
    cota_save(core, "autosave.cota");
    cota_free(core);
    return 0;
}