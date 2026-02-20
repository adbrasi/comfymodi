# Guias de uso — ComfyUI SaaS API

Guias práticos para as tarefas mais comuns. Todos os exemplos assumem que você tem o `comfyui_api.py` deployado.

---

## 1. Adicionar um novo workflow

### Como funciona

Os workflows **não ficam salvos no Modal**. Cada request envia o workflow completo. Você define o workflow no seu backend e injeta os valores do usuário via `inputs`.

### Passo a passo

**1. Exporte o workflow do ComfyUI como API JSON**

No ComfyUI Desktop/local, clique no menu → **Save (API Format)**.
O arquivo gerado (`workflow_api.json`) é o formato que a API aceita.
Não use o formato padrão (sem API) — ele não tem os node IDs corretos.

**2. Identifique os nodes que precisam de inputs dinâmicos**

Abra o JSON e localize os campos que mudam por usuário. Por exemplo:
```json
{
  "6": {
    "class_type": "CLIPTextEncode",
    "inputs": { "text": "prompt aqui" }
  },
  "53": {
    "class_type": "KSampler",
    "inputs": { "seed": 12345, "steps": 20 }
  }
}
```
Os IDs de node são as chaves (`"6"`, `"53"`).

**3. No seu backend, carregue o workflow e submeta**

```python
import json, os, requests

workflow = json.loads(open("workflows/meu_workflow.json").read())

r = requests.post(
    f"{API_URL}/v1/jobs",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "workflow": workflow,
        "inputs": [
            {"node": "6",  "field": "text",  "value": prompt_do_usuario, "type": "raw"},
            {"node": "53", "field": "seed",  "value": seed_aleatorio,     "type": "raw"},
            {"node": "53", "field": "steps", "value": 20,                 "type": "raw"},
        ],
        "user_id": str(discord_user_id),
        "webhook_url": "https://seu-backend.com/webhook/comfyui",
    },
)
job = r.json()
job_id = job["job_id"]
```

**4. Poll até completar**

```python
import time

while True:
    r = requests.get(
        f"{API_URL}/v1/jobs/{job_id}",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    data = r.json()
    status = data["status"]
    progress = data["progress"]

    if status == "completed":
        for output in data["outputs"]:
            print(f"URL: {output['url']}")
        break
    elif status in ("failed", "cancelled"):
        print(f"Erro: {data['error']}")
        break

    time.sleep(3)
```

### Onde guardar os workflows no seu projeto

```
workflows/
  sdxl_portrait.json       — geração de retrato com SDXL
  sdxl_landscape.json      — paisagem
  wan_video_t2v.json       — WanVideo text-to-video
```

Carregue com `json.loads(Path("workflows/sdxl_portrait.json").read_text())`.

---

## 2. Adicionar um novo modelo

Os modelos ficam no Volume `comfyui-models-cache` e são baixados durante o build da imagem.

### Passo a passo

**1. Abra `comfyui_api.py` e adicione à lista `MODELS`**

```python
MODELS = [
    # Modelo atual
    ("OnomaAIResearch/Illustrious-XL-v1.0", "Illustrious-XL-v1.0.safetensors", "checkpoints"),

    # Adicione aqui:
    ("madebyollin/sdxl-vae-fp16-fix", "sdxl_vae.safetensors", "vae"),
    ("stabilityai/stable-diffusion-xl-refiner-1.0", "sd_xl_refiner_1.0.safetensors", "checkpoints"),
]
```

Formato: `(repo_id_huggingface, nome_do_arquivo, subdiretório)`

**2. Faça o deploy**

```bash
modal deploy comfyui_api.py
```

O build detecta se o arquivo já existe no volume pelo nome e tamanho. Se já existir, pula. Se for novo, baixa.

**3. Verifique**

```bash
modal run comfyui_api.py::verify_setup
```

Mostra todos os modelos presentes no volume e o tamanho total.

### Subdiretórios suportados pelo ComfyUI

| Subdir | O que guardar |
|--------|---------------|
| `checkpoints` | Modelos principais (SDXL, SD1.5, Flux, etc.) |
| `vae` | VAEs separados |
| `loras` | LoRA adapters |
| `controlnet` | ControlNet models |
| `clip` | Text encoders (CLIP, T5) |
| `unet` | UNets separados (Flux dev/schnell) |
| `upscale_models` | Upscalers (RealESRGAN, etc.) |

### Modelos de outros repositórios (não Hugging Face)

Se o modelo não está no HF Hub, você pode fazer upload manual para o volume:

```bash
# Sobe um arquivo local para o volume
modal volume put comfyui-models-cache /caminho/local/modelo.safetensors /models/checkpoints/modelo.safetensors
```

---

## 3. Adicionar custom nodes

Os custom nodes são instalados durante o build da imagem via `git clone`.

### Passo a passo

**1. Abra `comfyui_api.py` e adicione à lista `CUSTOM_NODES`**

```python
# Sem pin de commit (menos reprodutível):
CUSTOM_NODES = [
    "https://github.com/kijai/ComfyUI-WanVideoWrapper",
]

# Com pin de commit (recomendado para produção):
CUSTOM_NODES = [
    ("https://github.com/kijai/ComfyUI-WanVideoWrapper", "a3b4c5d"),  # hash do commit
]
```

Para pegar o commit hash:
```bash
git ls-remote https://github.com/kijai/ComfyUI-WanVideoWrapper HEAD
```

**2. Faça o deploy**

```bash
modal deploy comfyui_api.py
```

O build clona o repositório e instala os `requirements.txt` do node automaticamente.

**3. Se o node precisar de modelos extras**

Adicione os modelos à lista `MODELS`. Por exemplo, para WanVideo:

```python
MODELS = [
    ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "wan2_1_t2v_14b.safetensors", "checkpoints"),
    # Adicione VAE, CLIP, etc. conforme a documentação do node
]
```

### Troubleshooting de custom nodes

**Node não aparece no ComfyUI**
→ Verifique os logs do `modal deploy`. Se o clone falhou, o deploy exibe o erro.

**`ImportError` ou `ModuleNotFoundError` no worker**
→ O node precisa de um pacote Python extra. Adicione ao `gpu_image`:
```python
gpu_image = (
    ...
    .pip_install("algum-pacote==1.2.3")  # adicione antes do run_function
    .run_function(install_custom_nodes)
    ...
)
```

---

## 4. Criar múltiplos apps (imagem, vídeo, etc.)

Cada `APP_NAME` gera um app separado no Modal com sua própria URL, volumes e billing independentes.

### Estratégia recomendada: um arquivo por app

**`comfyui_image_api.py`** — geração de imagem:
```python
APP_NAME = "comfyui-image"
CACHE_VOL_NAME = "comfyui-image-cache"   # volume próprio

MODELS = [
    ("OnomaAIResearch/Illustrious-XL-v1.0", "Illustrious-XL-v1.0.safetensors", "checkpoints"),
    ("madebyollin/sdxl-vae-fp16-fix", "sdxl_vae.safetensors", "vae"),
]
CUSTOM_NODES: list = []
```

**`comfyui_video_api.py`** — geração de vídeo:
```python
APP_NAME = "comfyui-video"
CACHE_VOL_NAME = "comfyui-video-cache"   # volume isolado (modelos grandes)

MODELS = [
    ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "wan2_1_t2v_14b.safetensors", "checkpoints"),
]
CUSTOM_NODES = [
    ("https://github.com/kijai/ComfyUI-WanVideoWrapper", "abc123def"),
]

# Vídeo precisa de mais memória — considere GPU maior
GPU_CONFIG = "a100"
```

### Deploy de múltiplos apps

```bash
# Copia o arquivo base
cp comfyui_api.py comfyui_video_api.py

# Edita APP_NAME, MODELS, CUSTOM_NODES no novo arquivo
# ... edição ...

# Deploya cada um separadamente
modal deploy comfyui_api.py        # URL: workspace--comfyui-saas-api
modal deploy comfyui_video_api.py  # URL: workspace--comfyui-video-api
```

### URLs dos apps

```bash
modal app list
# comfyui-saas    deployed  →  https://workspace--comfyui-saas-api.modal.run
# comfyui-video   deployed  →  https://workspace--comfyui-video-api.modal.run
```

No seu backend, direcione cada tipo de request para a URL correta:
```python
COMFYUI_IMAGE_URL = "https://workspace--comfyui-saas-api.modal.run"
COMFYUI_VIDEO_URL = "https://workspace--comfyui-video-api.modal.run"

url = COMFYUI_VIDEO_URL if is_video_request else COMFYUI_IMAGE_URL
```

### Volumes compartilhados vs. separados

| Situação | Recomendação |
|----------|-------------|
| Apps usam modelos diferentes (ex: SDXL vs Wan2.1) | `CACHE_VOL_NAME` separado por app |
| Apps compartilham modelos base (ex: SDXL base + refinador) | Mesmo `CACHE_VOL_NAME` |
| Um app enterprise por cliente | `APP_NAME` único + `CACHE_VOL_NAME` único por cliente |

---

## 5. Escalar para pico de usuários

### Configurações no topo de `comfyui_api.py`

```python
# Modo econômico (padrão) — cold start ao acordar
GPU_MAX_CONTAINERS = 2
GPU_MIN_CONTAINERS = 0
GPU_BUFFER_CONTAINERS = 0

# Modo pico — sempre 1 GPU quente + expande rápido
GPU_MAX_CONTAINERS = 5
GPU_MIN_CONTAINERS = 1     # ~$1.95/h sempre ligado com L40S
GPU_BUFFER_CONTAINERS = 1  # pré-aquece +1 GPU em antecipação

# Limite por usuário
MAX_ACTIVE_JOBS_PER_USER = 5   # máx jobs ativos simultaneamente
QUEUED_TIMEOUT_SECONDS = 360   # 6 min na fila antes de falhar
```

### Estratégia horária (avançado)

Para um bot Discord, o pico é geralmente à noite e fins de semana. Você pode ter dois deploys com configs diferentes e alternar via CI/CD, ou ajustar manualmente:

```bash
# Manhã (baixa demanda) — zero custo idle
GPU_MIN_CONTAINERS=0 modal deploy comfyui_api.py

# Noite (pico) — container sempre quente
GPU_MIN_CONTAINERS=1 GPU_BUFFER_CONTAINERS=1 modal deploy comfyui_api.py
```

---

## 6. Configurar webhooks

Webhooks permitem que o Modal notifique seu backend quando um job termina — útil para Discord (responder na thread sem polling).

### No job create

```python
r = requests.post(
    f"{API_URL}/v1/jobs",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "workflow": workflow,
        "inputs": [...],
        "user_id": str(discord_user_id),
        "webhook_url": "https://seu-backend.com/webhook/comfyui",
    },
)
```

### No seu backend (endpoint receptor)

```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/webhook/comfyui")
async def comfyui_webhook(request: Request):
    data = await request.json()
    event = data["event"]      # "job.completed" ou "job.failed"
    job_id = data["job_id"]

    if event == "job.completed":
        # busca outputs e responde no Discord
        status = requests.get(f"{API_URL}/v1/jobs/{job_id}", headers=headers).json()
        image_url = status["outputs"][0]["url"]
        await discord_channel.send(image_url)

    return {"ok": True}
```

### Retry automático

O worker tenta enviar o webhook 3 vezes (backoff exponencial: 1s, 2s, 4s) antes de desistir.

---

## 7. Enviar imagens como input (img2img, ControlNet)

Para workflows que precisam de imagem de entrada, use o campo `media`:

```python
import base64

# Opção 1: base64 (para arquivos locais ou já em memória)
with open("foto_usuario.jpg", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

r = requests.post(
    f"{API_URL}/v1/jobs",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "workflow": workflow,
        "media": [
            {"name": "input.jpg", "data": encoded}
        ],
        "inputs": [
            # No workflow, referencia o arquivo pelo nome
            {"node": "10", "field": "image", "value": "input.jpg", "type": "raw"},
        ],
        ...
    },
)

# Opção 2: URL pública (ex: avatar do Discord)
r = requests.post(
    f"{API_URL}/v1/jobs",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "workflow": workflow,
        "media": [
            {"name": "avatar.png", "url": "https://cdn.discordapp.com/avatars/..."}
        ],
        "inputs": [
            {"node": "10", "field": "image", "value": "avatar.png", "type": "raw"},
        ],
        ...
    },
)
```

O worker salva o arquivo em `ComfyUI/input/{job_id}/avatar.png` e o workflow acessa normalmente.

---

## 8. Referência rápida de variáveis de ambiente

Defina via `modal secret` ou env vars no deploy:

| Variável | Default | Descrição |
|----------|---------|-----------|
| `API_KEY` | — | Chave Bearer (obrigatório, via secret) |
| `GPU_MAX_CONTAINERS` | `2` | Máximo de GPUs simultâneas |
| `GPU_MIN_CONTAINERS` | `0` | GPUs sempre ligadas (custo fixo) |
| `GPU_BUFFER_CONTAINERS` | `0` | GPUs pré-aquecidas em antecipação |
| `GPU_SCALEDOWN_WINDOW_SECONDS` | `60` | Tempo ocioso antes de desligar GPU |
| `GPU_CONFIG` | `l40s,a100,a10g` | GPUs em ordem de preferência |
| `API_MAX_CONTAINERS` | `1` | Containers da API (CPU, leve) |
| `MAX_ACTIVE_JOBS_PER_USER` | `5` | Limite de jobs ativos por usuário |
| `QUEUED_TIMEOUT_SECONDS` | `360` | Tempo máximo na fila antes de falhar |
| `R2_URL_TTL_SECONDS` | `86400` | Validade das URLs de output (24h) |
