# ComfyUI SaaS API — Modal.com

REST API serverless para execução de workflows ComfyUI em GPU, com cold start otimizado via memory snapshots e armazenamento de outputs no Cloudflare R2.

---

## Estrutura do projeto

```
comfyui_api.py          — único arquivo de deploy (toda a lógica)
memory_snapshot_helper/ — custom node que permite snapshot seguro (sem CUDA)
workflows/              — workflows de exemplo/referência (não são carregados automaticamente)
test_run.py             — script de teste com progress bar
.env.example            — template de variáveis de ambiente
CLAUDE.md               — instruções para o agente de desenvolvimento
```

---

## Como funciona

```
Cliente (seu SaaS backend)
        │
        │  POST /v1/jobs  { workflow, inputs, user_id }
        ▼
  API Function (CPU, leve, sempre ativa)
        │  salva job no Volume + spawna worker
        ▼
  ComfyService (GPU L40S, memory snapshot)
        │  executa ComfyUI, acompanha progresso via WebSocket
        ▼
  Cloudflare R2
        │  outputs uploadados, URL assinada retornada
        ▼
  Cliente polling GET /v1/jobs/{job_id}  →  { status, progress, outputs }
```

### Dois containers separados

| Container | Tipo | Função |
|-----------|------|--------|
| `api` | CPU (leve) | Recebe requisições, salva jobs, serve status |
| `ComfyService` | GPU L40S | Roda o ComfyUI, processa os workflows |

A separação significa que o container de API está sempre quente (responde em ms), enquanto o GPU só sobe quando há jobs.

### Memory Snapshots (cold start rápido)

O `memory_snapshot_helper/` é um custom node oficial do Modal que:
1. Na **primeira inicialização**: lança ComfyUI sem CUDA (apenas CPU) → tira snapshot da memória
2. Nas **restaurações**: restaura o snapshot + re-habilita a GPU via `/cuda/set_device`

Resultado: cold start cai de ~3-5 min para ~8-10 segundos após o primeiro snapshot.

---

## Deploy

### Pré-requisitos

```bash
pip install modal
modal setup          # autentica com sua conta Modal
```

### Secrets necessários

```bash
# Chave de autenticação da API
modal secret create comfyui-api-secret \
  API_KEY=sua-chave-secreta-aqui

# Credenciais Cloudflare R2 (S3-compatible)
modal secret create comfyui-r2 \
  R2_ACCESS_KEY_ID=... \
  R2_SECRET_ACCESS_KEY=... \
  R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com \
  R2_BUCKET_NAME=comfyui-outputs
```

### Deploy

```bash
modal deploy comfyui_api.py
```

A URL da API aparecerá no output: `https://<workspace>--comfyui-saas-api.modal.run`

### Verificar setup

```bash
modal run comfyui_api.py::verify_setup
```

---

## Como criar apps diferentes (imagem, vídeo, etc.)

**Tudo começa mudando `APP_NAME`, `MODELS` e `CUSTOM_NODES` no topo do arquivo.**

### App de geração de imagem (SDXL)

```python
APP_NAME = "comfyui-image"

MODELS = [
    ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors", "checkpoints"),
    ("madebyollin/sdxl-vae-fp16-fix", "sdxl_vae.safetensors", "vae"),
]

CUSTOM_NODES: list = []
```

### App de geração de vídeo (WanVideo)

```python
APP_NAME = "comfyui-video"

MODELS = [
    ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "wan2_1_t2v_14b.safetensors", "checkpoints"),
]

CUSTOM_NODES = [
    ("https://github.com/kijai/ComfyUI-WanVideoWrapper", "abc123"),  # (url, commit_hash)
]
```

Cada `APP_NAME` diferente cria um **app separado no Modal** com sua própria URL, volumes e billing.

> Para múltiplos apps, duplique o arquivo (ex: `comfyui_image_api.py`, `comfyui_video_api.py`) e altere `APP_NAME` em cada um.

---

## Como adicionar modelos ao volume

Os modelos são **baixados uma única vez durante o build da imagem** e ficam persistidos no Modal Volume `comfyui-models-cache`.

### Adicionando um novo modelo

1. Adicione à lista `MODELS` no arquivo:

```python
MODELS = [
    ("OnomaAIResearch/Illustrious-XL-v1.0", "Illustrious-XL-v1.0.safetensors", "checkpoints"),
    ("madebyollin/sdxl-vae-fp16-fix", "sdxl_vae.safetensors", "vae"),
]
```

Formato: `(repo_id_huggingface, nome_do_arquivo, subdiretório_relativo_a_models/)`

2. Faça o deploy: o build detecta que o arquivo não está no cache e baixa automaticamente. Se já existir, pula.

### Estrutura de diretórios no volume

```
/cache/
  models/
    checkpoints/   — modelos principais (.safetensors, .ckpt)
    vae/           — VAEs
    loras/         — LoRAs
    controlnet/    — ControlNets
    clip/          — encoders de texto
    unet/          — UNets separados
  outputs/         — outputs temporários
```

### Volumes compartilhados entre apps

O volume `comfyui-models-cache` é compartilhado por padrão. Para volumes separados por app:

```python
CACHE_VOL_NAME = "comfyui-models-video"  # volume dedicado
```

---

## Workflows: como funcionam

**Os workflows NÃO ficam salvos no Modal.** Cada chamada de API envia o workflow completo no corpo da requisição.

### Quem define o workflow?

Seu **backend SaaS** monta o workflow e os inputs antes de chamar a API. O cliente final não vê o workflow — ele só interage com a interface do seu produto.

### Formato da requisição

```json
POST /v1/jobs
Authorization: Bearer <sua-api-key>

{
  "workflow": { ...workflow_json_completo... },
  "inputs": [
    { "node": "6",  "field": "text",  "value": "prompt do usuário", "type": "raw" },
    { "node": "53", "field": "seed",  "value": 42,                  "type": "raw" },
    { "node": "53", "field": "steps", "value": 20,                  "type": "raw" }
  ],
  "user_id": "user_abc123",
  "webhook_url": "https://seu-backend.com/webhooks/comfyui"
}
```

O campo `inputs` sobrescreve valores específicos no workflow antes de executar. Útil para injetar o prompt e seed do usuário sem modificar o workflow base.

### Onde guardar os workflows?

| Estratégia | Quando usar |
|------------|-------------|
| **Arquivo local** (pasta `workflows/`) | Desenvolvimento e testes |
| **Banco de dados** (Postgres, Mongo) | Produção com templates por tier |
| **Hardcoded no backend** | App com workflow fixo (ex: "gerador de avatar") |

### Exemplo: carregar workflow de arquivo e submeter

```python
import json, os, requests

workflow = json.loads(open("workflows/sdxl_simple_exampleV2.json").read())

r = requests.post(
    "https://sua-url--comfyui-saas-api.modal.run/v1/jobs",
    headers={"Authorization": f"Bearer {os.environ['COMFYUI_API_KEY']}"},
    json={
        "workflow": workflow,
        "inputs": [
            {"node": "6",  "field": "text",  "value": "retrato cinematográfico", "type": "raw"},
            {"node": "53", "field": "seed",  "value": 12345,                      "type": "raw"},
        ],
        "user_id": "user_123",
    },
)
job = r.json()
print(job["job_id"])  # poll com GET /v1/jobs/{job_id}
```

---

## Referência da API

### Autenticação

Todas as rotas (exceto `/health`) exigem:
```
Authorization: Bearer <API_KEY>
```

### Endpoints

#### `POST /v1/jobs` — Criar job

```json
{
  "workflow":    { ... },           // workflow ComfyUI completo (obrigatório)
  "inputs":      [ ... ],           // overrides de campos (opcional)
  "media":       [ ... ],           // arquivos de entrada (imagens, vídeos)
  "webhook_url": "https://...",     // callback ao completar (opcional)
  "user_id":     "string"           // ID do usuário para rastreamento (opcional)
}
```

Retorna: `{ "job_id": "uuid", "status": "queued", "created_at": "..." }`

#### `GET /v1/jobs/{job_id}` — Status do job

```json
{
  "job_id":       "uuid",
  "status":       "queued|running|completed|failed|cancelled",
  "progress":     0-100,
  "current_node": "53",
  "current_step": 15,
  "total_steps":  20,
  "nodes_done":   3,
  "nodes_total":  5,
  "outputs":      [ { "filename": "...", "url": "...", "size_bytes": 2097152 } ],
  "logs":         [ "[20:01:02] Job started", "..." ],
  "error":        null
}
```

#### `GET /v1/jobs` — Listar jobs

Query params: `?status=completed&user_id=user_123&limit=50`

#### `DELETE /v1/jobs/{job_id}` — Cancelar job

Solicita cancelamento com `FunctionCall.cancel()` no Modal e marca o job como `cancelled`.

#### `GET /health` — Health check (sem auth)

---

## Comportamento de cold start

| Cenário | Tempo estimado |
|---------|----------------|
| Container warm (snapshot restaurado) | ~3-8 s até iniciar execução |
| Cold start com snapshot existente | ~30-90 s (depende de alocação GPU) |
| Primeira execução (criando snapshot) | ~3-5 min |
| GPU indisponível (espera por alocação) | 0-6 min |

Jobs em `queued` por mais de **360s** (configurável via `QUEUED_TIMEOUT_SECONDS`) são marcados como `failed` automaticamente. Faça retry no seu backend.

### Controle de custo

- `GPU_MAX_CONTAINERS=2` — máximo de GPUs simultâneas
- `GPU_MIN_CONTAINERS=0` — zero custo idle (mais cold start)
- `GPU_MIN_CONTAINERS=1` — container sempre quente (~$1.95/h com L40S)
- `GPU_BUFFER_CONTAINERS` — GPUs extras prontas em pico
- `MAX_ACTIVE_JOBS_PER_USER=5` — limite de jobs ativos por usuário

---

## Monitoring e logs

```bash
modal app list           # ver apps deployados
modal app logs comfyui-saas   # logs em tempo real
modal container list     # ver containers ativos
```

Os logs de execução também são retornados no campo `logs` de cada job.

---

## Troubleshooting

**Job fica em `queued` e falha por timeout**
→ A GPU não ficou disponível a tempo. Aumente `QUEUED_TIMEOUT_SECONDS` ou configure `GPU_MIN_CONTAINERS=1`.

**`ModuleNotFoundError` ao fazer deploy**
→ Adicione o pacote ao `.pip_install()` do `gpu_image`.

**R2 URL expirada**
→ URLs têm validade de 24h (configurável). `GET /v1/jobs/{job_id}` reemite URL assinada automaticamente.
