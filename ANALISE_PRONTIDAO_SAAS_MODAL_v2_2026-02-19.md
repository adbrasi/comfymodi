# Analise de prontidao SaaS (v2 - revisado)

Data: 19/02/2026  
Base de comparacao: codigo atual + `README.md` + exemplos oficiais Modal + docs Modal (Context7).

## Atualizacao final (19/02/2026 23:09 UTC)
Esta secao substitui qualquer ponto conflitante abaixo.

### O que foi validado de forma real agora
1. Benchmark real em producao (`cold`, `warm`, `concorrencia=3`, `cancelamento`):
- arquivo: `tests_outputs/benchmark_latency_warm_cold_conc_cancel_20260219_200228.json`
- `cold`: `queue_wait=44.94s`, `run=39.02s`, `total=83.96s`
- `warm`: `queue_wait=4.82s`, `run=38.28s`, `total=43.11s`
- `cancel`: job chegou em `running`, `DELETE` retornou `200`, estado final `cancelled`
- provas de output: `tests_outputs/benchmark_cold_2b1bcb7a.png`, `tests_outputs/benchmark_warm_f02ae642.png`

2. Burst de cota por usuario (`8` submits paralelos):
- arquivo: `tests_outputs/quota_burst_check_20260219_200402.json`
- resultado: `accepted=6`, `429=2`, `other=0`
- leitura correta: limite atual controla **ativos simultaneos**, nao “maximo absoluto por janela curta”.

3. Validacao final pos-hardening de auth:
- output: `tests_outputs/post_auth_hardening_e7be18b3.png`
- tempos: `queue_wait=42.52s`, `run_time=30.08s`, `total=72.6s`

### Ajustes novos aplicados no codigo (prontos)
1. API key em comparacao constante:
- `comfyui_api.py`: `hmac.compare_digest(...)` em `verify_api_key`.

2. Formato estrito de `X-User-ID`:
- `comfyui_api.py`: regex `^[A-Za-z0-9._:@-]{1,128}$`.

3. Proxy auth nativo Modal opcional:
- `comfyui_api.py`: `@modal.asgi_app(requires_proxy_auth=API_REQUIRE_PROXY_AUTH)`.
- habilitar com `API_REQUIRE_PROXY_AUTH=1`.

4. README alinhado com producao real:
- formato de `X-User-ID`, proxy auth opcional, knobs de custo/latencia e tempos cold/warm realistas.

### Comparacao objetiva com os exemplos oficiais citados
1. `06_gpu_and_ml/comfyui/comfyapp.py`
- exemplo usa `@app.cls(..., scaledown_window=300, gpu=\"L40S\")` + health guard com `stop_fetching_inputs`.
- seu codigo: mesma estrategia de health guard/remoção de container ruim e API headless por classe GPU.

2. `06_gpu_and_ml/comfyui/memory_snapshot/memory_snapshot_example.py`
- exemplo usa `enable_memory_snapshot=True` + `@modal.enter(snap=True)` e `@modal.enter(snap=False)` com `/cuda/set_device`.
- seu codigo: alinhado exatamente nesse padrao.

3. `06_gpu_and_ml/image-to-video/image_to_video.py`
- exemplo usa volume para pesos e `scaledown_window` para balancear custo/latencia.
- seu codigo: usa volume para modelos e agora defaults economicos (`min=0`, `buffer=0`) com override por env.

4. `06_gpu_and_ml/gpu_fallbacks.py`
- exemplo mostra fallback de GPU por lista.
- seu codigo: usa lista configuravel por `GPU_CONFIG` (`l40s,a100,a10g`), sem `any` por previsibilidade.

5. `03_scaling_out/cls_with_options.py`
- exemplo reforca “perfil de recurso por contexto”.
- no seu caso, isso foi materializado por knobs de env (sem adicionar complexidade estrutural).

6. `07_web_endpoints/basic_web.py`
- exemplo recomenda `requires_proxy_auth=True` para endpoints sensiveis.
- seu codigo: agora suporta essa camada extra por env (`API_REQUIRE_PROXY_AUTH=1`).

### Veredito de prontidao (honesto)
- **Pronto para producao controlada via backend proprio**: **SIM**.
- **Para 2k+ DAU com baixa surpresa de latencia**: **SIM, com tune de deploy**:
  - em horario de pico: `GPU_MIN_CONTAINERS=1` e `GPU_BUFFER_CONTAINERS=1`
  - fora de pico: `GPU_MIN_CONTAINERS=0`, `GPU_BUFFER_CONTAINERS=0`
  - manter `MAX_ACTIVE_JOBS_PER_USER` + rate limit no gateway (Cloudflare/API Gateway)
  - se endpoint exposto publicamente: `API_REQUIRE_PROXY_AUTH=1`

## Veredito rapido
- **Pronto para producao controlada (2k+ DAU) via backend proprio**: **SIM, com pre-condicoes operacionais**.
- **Pronto para abertura publica sem gateway/controles externos**: **NAO**.

## O que foi corrigido agora (codigo)
1. **SSRF endurecido (DNS + localhost + redirect + limite de tamanho)**
- No codigo: `comfyui_api.py:877`, `comfyui_api.py:916`, `comfyui_api.py:944`, `comfyui_api.py:456`.
- O que mudou:
- bloqueio explicito de `localhost`.
- resolucao DNS e bloqueio de IP restrito apos resolucao.
- controle de redirect validando URL a cada salto.
- limite de 50MB em download externo.

2. **Cancelamento forte com API da Modal**
- No codigo: `comfyui_api.py:847`, `comfyui_api.py:858`, `comfyui_api.py:1124`.
- O que mudou:
- persiste `function_call_id` no job ao `spawn`.
- `DELETE /v1/jobs/{id}` chama `modal.FunctionCall.from_id(call_id).cancel()`.
- worker continua com cancelamento cooperativo lendo status do volume.

3. **Ownership e isolamento por tenant obrigatorios**
- No codigo: `comfyui_api.py:254`, `comfyui_api.py:792`, `comfyui_api.py:809`, `comfyui_api.py:1008`, `comfyui_api.py:1062`.
- O que mudou:
- `user_id` virou obrigatorio no payload.
- header `X-User-ID` obrigatorio.
- `GET /v1/jobs/{id}` e `DELETE` validam ownership.
- `GET /v1/jobs` retorna apenas jobs do caller.

4. **Quota de capacidade para conter custo idiota**
- No codigo: `comfyui_api.py:54`, `comfyui_api.py:55`, `comfyui_api.py:985`.
- O que mudou:
- limite global de jobs ativos (`MAX_ACTIVE_JOBS_GLOBAL`).
- limite de jobs ativos por usuario (`MAX_ACTIVE_JOBS_PER_USER`).

5. **Timeout de fila saiu de GET/list (sem side effect em leitura)**
- No codigo: `comfyui_api.py:56`, `comfyui_api.py:1189`.
- O que mudou:
- expiracao de queued foi movida para tarefa agendada (`fail_stale_queued_jobs`).

6. **Reemissao de URL assinada no GET de job**
- No codigo: `comfyui_api.py:57`, `comfyui_api.py:1013`, `comfyui_api.py:1056`.
- O que mudou:
- outputs com `r2_key` recebem URL nova no `GET /v1/jobs/{id}`.

7. **Pinning da stack torch para build reproducivel**
- No codigo: `comfyui_api.py:171`.
- O que mudou:
- `torch==2.8.0`, `torchvision==0.23.0`, `torchaudio==2.8.0`.

8. **Escala/custo com knobs explicitos**
- No codigo: `comfyui_api.py:48`, `comfyui_api.py:341`.
- O que mudou:
- `max/min/buffer/scaledown` parametrizados por env var.

## Comparativo direto com exemplos oficiais
1. **`06_gpu_and_ml/comfyui/comfyapp.py`**
- No exemplo, eles fazem servidor ComfyUI em `@app.cls` e health guard com `modal.experimental.stop_fetching_inputs`.
- No seu codigo, isso esta alinhado (`comfyui_api.py:387`, `comfyui_api.py:370`) e com API de producao por cima.

2. **`06_gpu_and_ml/comfyui/memory_snapshot/memory_snapshot_example.py`**
- No exemplo, eles fazem `@modal.enter(snap=True)` + `@modal.enter(snap=False)` e reativam CUDA com `/cuda/set_device`.
- No seu codigo, esta alinhado (`comfyui_api.py:353`, `comfyui_api.py:336`).

3. **`06_gpu_and_ml/image-to-video/image_to_video.py`**
- No exemplo, eles pinam dependencias e usam revisao fixa de modelo.
- No seu codigo, o pinning da stack torch foi alinhado (`comfyui_api.py:171`), mas ainda faltam revisoes SHA nos modelos Hugging Face listados em `MODELS`.

4. **`09_job_queues/web_job_queue_wrapper.py` + `doc_ocr_webapp.py` + `08_advanced/poll_delayed_result.py`**
- Nos exemplos, o padrao e `spawn` + polling com `FunctionCall.from_id(...).get(timeout=0)`.
- No seu codigo, isso foi levado para um fluxo de job persistido e agora com cancelamento explicito via `FunctionCall.cancel()` (`comfyui_api.py:1127`).

5. **`07_web_endpoints/basic_web.py`**
- No exemplo, endpoint sensivel pode usar `requires_proxy_auth=True`.
- No seu codigo, auth principal e Bearer + `X-User-ID`; proxy auth do Modal ainda e opcional e recomendado para camada extra.

6. **`03_scaling_out/cls_with_options.py` + `dynamic_batching.py`**
- Nos exemplos, foco em ajuste dinamico de escala.
- No seu codigo, voce manteve `max_inputs=1` (coerente para ComfyUI pesado) e adicionou knobs de capacidade/latencia por env var.

## Cancelamento: como os exemplos fazem
- Nos exemplos de fila citados acima, **nao existe endpoint HTTP de cancelamento**.
- O padrao oficial mostrado e submit async + polling.
- Para cancelamento, a base confiavel vem da API do SDK Modal:
- validado localmente no seu `.venv`: `modal 1.3.3`, `FunctionCall.cancel(self, terminate_containers: bool = False)`.
- Seu codigo agora usa esse caminho oficialmente suportado.

## Estado atual de prontidao para 2k+ DAU
## Ja esta bom
1. Separacao API (CPU) + worker GPU com autoscaling.
2. Snapshot de memoria para reduzir cold start.
3. Jobs assincronos com progresso e polling.
4. Outputs em R2 com URL assinada renovavel.
5. Cancelamento funcional de job via API Modal.
6. Isolamento basico de tenant no proprio backend.

## Ainda precisa para producao publica robusta
1. **Rate limit distribuido por segundo/minuto no gateway** (Cloudflare/API Gateway/Nginx), nao apenas quota de jobs ativos.
2. **Observabilidade de producao**: metricas (fila, tempo medio, erro, cancel), alertas e dashboards.
3. **Teste de carga real** com perfil do seu SaaS (burst e pico) para calibrar `MAX_ACTIVE_*`, `GPU_MAX_CONTAINERS`, `QUEUED_TIMEOUT_SECONDS`.
4. **Revisao SHA de modelos** (quando aplicavel) para previsibilidade total de deploy.

## APIs prontas para seu site (agora)
1. `POST /v1/jobs` (obrigatorio `Authorization` + `X-User-ID`, e `user_id` no body igual ao header)
2. `GET /v1/jobs/{job_id}`
3. `GET /v1/jobs`
4. `DELETE /v1/jobs/{job_id}`
5. `GET /health`

## README atualizado
- Contrato novo de tenant/header: `README.md:259`.
- Timeout de fila por tarefa agendada: `README.md:335`.
- Reemissao de URL assinada: `README.md:385`.
- Controle de custo por quota/buffer: `README.md:350`.

## Validacao executada nesta revisao
1. `python -m py_compile comfyui_api.py test_api.py test_run.py` -> OK.
2. Smoke de SSRF: `localhost` e `127.0.0.1` bloqueados; `https://example.com` permitido -> OK.
3. Smoke de fetch externo seguro: download de payload pequeno publico (`httpbin /bytes/16`) com validacao -> OK.
4. SDK Modal no venv: `modal 1.3.3` com `FunctionCall.cancel(self, terminate_containers: bool = False)` -> OK.
5. E2E HTTP real no endpoint Modal:
- `GET /health` -> 200
- `POST /v1/jobs` com chave invalida -> 401
- `POST /v1/jobs` valido -> 200 (job criado)
- ownership check em `GET /v1/jobs/{id}` com outro `X-User-ID` -> 404
- `DELETE /v1/jobs/{id}` -> 200
- polling final -> `cancelled`
6. Execucao direta das funcoes de manutencao:
- `modal run comfyui_api.py::fail_stale_queued_jobs` -> OK
- `modal run comfyui_api.py::cleanup_old_jobs` -> OK

## Observacao sobre .env
- Para Modal em producao, `.env` e `.env.example` ajudam localmente, mas o runtime usa `modal secret`.
- Segredos necessarios: `comfyui-api-secret` (API_KEY) e `comfyui-r2` (R2_*).

## Referencias usadas
- https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/comfyui
- https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/comfyui/comfyapp.py
- https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/comfyui/memory_snapshot/memory_snapshot_example.py
- https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/image-to-video
- https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/image-to-video/image_to_video.py
- https://github.com/modal-labs/modal-examples/tree/main/09_job_queues
- https://github.com/modal-labs/modal-examples/blob/main/09_job_queues/web_job_queue_wrapper.py
- https://github.com/modal-labs/modal-examples/blob/main/09_job_queues/doc_ocr_jobs.py
- https://github.com/modal-labs/modal-examples/blob/main/09_job_queues/doc_ocr_webapp.py
- https://github.com/modal-labs/modal-examples/blob/main/08_advanced/poll_delayed_result.py
- https://github.com/modal-labs/modal-examples/tree/main/07_web_endpoints
- https://github.com/modal-labs/modal-examples/blob/main/07_web_endpoints/basic_web.py
- https://github.com/modal-labs/modal-examples/tree/main/03_scaling_out
- Context7 (`/websites/modal`): guides `job-queue`, `webhook-timeouts`, `volumes`, `scale`, examples `basic_web`/`doc_ocr_webapp`.

## Validacao real de inferencia (2 prompts)
Data: 19/02/2026

Base usada:
- endpoint deploy estavel: `https://cezarsaint--comfyui-saas-api.modal.run`
- workflow: `workflows/sdxl_simple_exampleV2.json` (usa `ChenkinNoob-XL-V0.2.safetensors`)
- saida local: `tests_outputs/`
- R2: validado por `r2_key` e download via URL assinada

Execucoes:
1. `job_id=2b0046a1-a56f-4a3e-9d16-a850345187be`
- status final: `completed`
- tempo fim-a-fim: `19.2s`
- outputs: `1`
- `r2_key`: `outputs/2b0046a1-a56f-4a3e-9d16-a850345187be/ComfyUI_00006_.png`
- arquivo local: `tests_outputs/run1_2b0046a1_ComfyUI_00006_.png`

2. `job_id=50a2cf69-ffce-4a9c-9b90-584e547aa547`
- status final: `completed`
- tempo fim-a-fim: `24.4s`
- outputs: `1`
- `r2_key`: `outputs/50a2cf69-ffce-4a9c-9b90-584e547aa547/ComfyUI_00007_.png`
- arquivo local: `tests_outputs/run2_50a2cf69_ComfyUI_00007_.png`

Comparacao:
- run1: `19.2s`
- run2: `24.4s`
- delta: `+5.2s` no run2

## Confirmacao de modelo no app/volume
- `modal run comfyui_api.py::verify_setup` retornou:
  - `Models: 1 files, 6.6 GB total`
  - `Setup OK!`

## Nuances de producao (a partir dos exemplos oficiais citados)
1. `gpu_fallbacks.py`:
- no exemplo eles usam `gpu=[\"h100\", \"a100\", \"any\"]` para reduzir risco de fila por SKU indisponivel.
- no seu codigo foi aplicado fallback por env (`GPU_CONFIG`).

2. `gpu_snapshot.py`:
- no exemplo eles destacam que snapshot GPU e para app deployado e exige testes cuidadosos em producao.
- seu fluxo com snapshot continua valido, mas o ganho depende de deploy estavel e aquecimento adequado.

3. `long-training.py`:
- no exemplo eles reforcam retries e reentrada com estado persistido para workloads longos.
- no seu caso de inferencia, o paralelo pratico e tratar falha de fila (`queued timeout`) com retry no backend chamador.

4. `gpu_packing.py`:
- no exemplo eles aumentam throughput por GPU com concorrencia e pool de modelos.
- para ComfyUI pesado, packing agressivo tende a bater VRAM; aqui faz mais sentido controlar `max_inputs=1`, usar fallback de GPU e warm pool.
