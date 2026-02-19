# Analise de prontidao SaaS (v2)

Data: 19/02/2026  
Escopo: comparar seu estado atual com referencias oficiais da Modal em `modal-examples/06_gpu_and_ml` (principalmente `comfyui`, `memory_snapshot`, `gpu_packing`, `gpu_fallbacks`) e avaliar se ja da para plugar no seu site.

## Veredito rapido
- **Pronto para beta fechado via backend seu**: **SIM** (com controles).
- **Pronto para producao publica (escala + multitenancy forte)**: **AINDA NAO**.

## Achados (ordenados por severidade)

## P0 - Critico
1. **SSRF ainda contornavel via `localhost`**
- Evidencia: `_validate_external_url` permite hostnames nao-IP (`comfyui_api.py:855-866`), e teste local confirmou `ALLOW http://localhost:8188`.
- Risco: atacante pode forcar requests internas no container (inclusive serviços locais).
- Nota de referencia: este ponto e inferencia de seguranca da implementacao atual; os exemplos oficiais focam em arquitetura e nao cobrem hardening SSRF em profundidade.
- Impacto SaaS: alto (seguranca de infra).

2. **Cancelamento nao interrompe explicitamente o prompt do ComfyUI**
- Evidencia: cancel seta status no volume (`comfyui_api.py:967-971`), worker detecta e faz `return` (`comfyui_api.py:520-523`), mas nao chama `/interrupt` nem `/queue/cancel` no ComfyUI local.
- Risco: job pode continuar rodando internamente por algum tempo, consumindo GPU/custo.
- Impacto SaaS: alto (custo + previsibilidade).

## P1 - Alto
3. **Isolamento multi-tenant ainda fraco (user_id e filtro sao opcionais)**
- Evidencia: `user_id` opcional (`comfyui_api.py:237`), listagem global existe e filtra apenas se caller passar `user_id` (`comfyui_api.py:915-947`).
- Risco: com chave vazada/uso interno incorreto, metadados de todos os clientes ficam expostos.

4. **Timeout de fila agressivo (2 min) com side effect em rotas de leitura**
- Evidencia: `QUEUED_TIMEOUT_SECONDS = 120` (`comfyui_api.py:821`) e transicao para failed ocorre dentro de GET/list (`comfyui_api.py:870-883`, `comfyui_api.py:895`, `comfyui_api.py:932`).
- Risco: falso-fail em picos de alocacao GPU; semantica de leitura alterando estado.

5. **Sem rate limit / quota por cliente**
- Evidencia: nao ha limiter nos endpoints FastAPI.
- Risco: burst/abuso derruba UX e aumenta custo.

6. **Build ainda parcialmente nao deterministico (drift de dependencias)**
- Evidencia: `torch`, `torchvision`, `torchaudio` sem versao fixa em `comfyui_api.py:152-157`.
- Comparativo: exemplos de `02_building_containers` e `image-to-video` priorizam pinning explicito de versoes para reproduzibilidade.
- Risco: comportamento muda entre deploys sem alteracao de codigo.

7. **Persistencia de payload grande no volume de jobs**
- Evidencia: request aceita ate 10 midias de 50MB e salva JSON com `media` no volume (`comfyui_api.py:776-813`), com fallback base64 para output (`comfyui_api.py:657-663`).
- Risco: I/O pesado e latencia sob carga.

## P2 - Medio
8. **README promete comportamento que o codigo nao implementa**
- Evidencia: README diz que `GET /v1/jobs/{job_id}` regenera URL expirada (`README.md:375`), mas o codigo apenas retorna o que esta salvo em `outputs` (`comfyui_api.py:909`).

9. **Testes automatizados nao estao fechados para CI**
- Evidencia: `python -m py_compile` OK; imports OK; mas suite `pytest` nao esta preparada no venv atual (nao ha `pytest` instalado no `.venv`).

## Comparativo ampliado (pastas solicitadas)
1. **`06_gpu_and_ml/comfyui`**
- No exemplo `comfyui/comfyapp.py`, eles fazem API simples com `@modal.fastapi_endpoint` e estado minimo de execucao; voce esta fazendo API completa com estado persistido em Volume (`comfyui_api.py:796-813`, `comfyui_api.py:915-947`).
- No exemplo `comfyui/comfyapp.py`, eles fazem health guard com `modal.experimental.stop_fetching_inputs`; voce tambem esta fazendo isso em `comfyui_api.py:370` (alinhado).

2. **`06_gpu_and_ml/comfyui/memory_snapshot`**
- No exemplo `memory_snapshot_example.py`, eles fazem snapshot em `snap=True` e restauram CUDA em `snap=False`; voce esta fazendo o mesmo em `comfyui_api.py:336-359`.
- No exemplo `memory_snapshot_example.py`, eles alertam que custom node pode quebrar snapshot; voce esta fazendo com helper local (`memory_snapshot_helper`), entao precisa manter teste de cold start a cada alteracao de node/model.

3. **`06_gpu_and_ml/image-to-video`**
- No exemplo `image_to_video.py`, eles fazem pinning estrito de versoes e revision SHA de modelo; voce esta fazendo pinning parcial e deixando `torch/torchvision/torchaudio` sem versao fixa (`comfyui_api.py:152-157`).
- No exemplo `image_to_video.py`, eles fazem retorno de arquivo/volume sem base64 gigante; voce esta fazendo URL assinada em R2 (bom) mas com fallback base64 (`comfyui_api.py:657-663`).

4. **`02_building_containers`**
- No exemplo `install_cuda.py`, eles fazem imagem explicitando stack CUDA quando necessario; voce esta fazendo build GPU funcional, mas com menor controle de reproducao por dependencia sem pinning total.
- No exemplo `02_building_containers`, eles fazem foco em previsibilidade de imagem; voce esta fazendo imagem robusta, mas ainda com risco de drift entre deploys por versoes abertas.

5. **`09_job_queues`**
- No exemplo `web_job_queue_wrapper.py`, eles fazem submit async com `spawn` e polling por `request_id`; voce esta fazendo `spawn` com job persistido em JSON (`comfyui_api.py:816`), que funciona, mas depende de consistencia de volume.
- No exemplo `doc_ocr_jobs.py`, eles fazem retries e padrao de fila escalavel para tarefas longas; voce esta fazendo timeout de fila de 120s (`comfyui_api.py:821`), que pode ser curto para picos de GPU.

## Cancelamento nos exemplos oficiais
- Nos exemplos que voce pediu (`09_job_queues/web_job_queue_wrapper.py` e `09_job_queues/doc_ocr_webapp.py`), o padrao oficial e **spawn + polling de status/resultado**; nao existe endpoint de cancelamento explicito nesses exemplos.
- Em `08_advanced/poll_delayed_result.py` (outro exemplo oficial do repo), o padrao continua sendo **spawn + poll com timeout=0** para saber se terminou, sem cancelamento HTTP exposto.
- Na documentacao oficial consultada via Context7 (`guide/webhook-timeouts` e `guide/trigger-deployed-functions`), o fluxo recomendado mostrado e o mesmo: **aceitar job, retornar `call_id`, e fazer polling**.
- A mesma base de docs via Context7 tambem indica existencia de `modal.FunctionCall.cancel` (changelog 1.1.3 cita esse metodo), mas nao encontrei nos exemplos acima um fluxo completo de endpoint de cancelamento usando esse metodo.
- Comparacao direta: voce esta indo alem dos exemplos ao implementar `DELETE /v1/jobs/{job_id}` e sinalizacao cooperativa de cancelamento (`comfyui_api.py:950-973` e `comfyui_api.py:515-523`), o que e positivo para SaaS. O gap restante e tornar esse cancelamento mais forte no ComfyUI (interrupt/cancel da fila local).

6. **`07_web_endpoints`**
- No exemplo `basic_web.py`, eles fazem opcao de `requires_proxy_auth=True` para endpoints sensiveis; voce esta fazendo auth bearer propria (`comfyui_api.py:761-768`), mas sem segunda camada de proxy auth do Modal.
- No exemplo `07_web_endpoints`, eles fazem padrao claro de API publica simples; voce esta fazendo API de producao com mais risco operacional, entao precisa rate-limit e quotas no gateway.

7. **`03_scaling_out`**
- No exemplo `cls_with_options.py`, eles fazem ajuste dinamico de recursos em runtime com `with_options`; voce esta fazendo configuracao fixa (`max_containers=10`, `max_inputs=1` em `comfyui_api.py:324-327`).
- No exemplo `dynamic_batching.py`, eles fazem batching para throughput; voce esta fazendo processamento serial por container (adequado para ComfyUI pesado), mas pode combinar com `min_containers`/`buffer_containers` para reduzir fila/cold start.

## Pontos muito bons (alinhados ao “santo graal” da Modal)
1. **Memory snapshot com helper dedicado**
- `@modal.enter(snap=True/snap=False)` em `comfyui_api.py:336-359`, usando `memory_snapshot_helper`.

2. **Health check com retirada do pool**
- `modal.experimental.stop_fetching_inputs()` em `comfyui_api.py:370`.

3. **Separacao API leve (CPU) e worker GPU**
- arquitetura correta para escalar e proteger latencia.

4. **Outputs em R2 com URL assinada**
- upload + presign em `comfyui_api.py:646-653`.

5. **Auth Bearer presente em todas as rotas de negocio**
- `verify_api_key` + `Depends` nos endpoints.

## APIs para seu site: o que ja pode liberar
## Pode liberar agora (via backend seu)
- `POST /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `DELETE /v1/jobs/{job_id}`
- `GET /health`

Condicao: manter chamadas **somente pelo seu backend** (nunca direto do front), com controle de usuario/plano no seu sistema.

## Nao liberar diretamente para cliente final ainda
- `GET /v1/jobs` sem gate adicional por tenant/escopo.

## Checklist minimo antes de “producao real”
1. Corrigir SSRF: bloquear `localhost`, resolver DNS e bloquear IP privado apos resolucao, bloquear redirect para rede privada.
2. Implementar cancelamento forte no worker: chamar `/interrupt` + `/queue/cancel` quando status virar `cancelled`.
3. Tornar `user_id` obrigatorio (ou claim do token) e validar ownership em `GET /v1/jobs/{id}`.
4. Adicionar rate limit/quota por cliente (API gateway ou middleware).
5. Ajustar timeout de fila (2 min tende a ser curto em picos).
6. Pinning completo de dependencias da imagem (torch stack incluso).
7. Opcional forte: endpoint para reemitir URL assinada de outputs expirados.

## Como resolver o checklist (baseado em exemplos + docs)
1. **SSRF**
- O que os exemplos/docs mostram: os exemplos oficiais focam no padrao de API e fila, nao trazem middleware pronto de SSRF.
- O que esta comprovado no seu codigo: `_validate_external_url` bloqueia IP privado literal, mas aceita hostname nao resolvido (`comfyui_api.py:855-866`).
- O que fazer de forma objetiva:
  - bloquear `localhost` explicitamente;
  - resolver DNS do hostname e aplicar o mesmo bloqueio de IP privado/loopback/link-local nos IPs resolvidos;
  - bloquear redirect que leve para rede privada.

2. **Cancelamento forte**
- O que os exemplos oficiais fazem:
  - `09_job_queues/web_job_queue_wrapper.py`: `spawn` + polling por `request_id`;
  - `09_job_queues/doc_ocr_webapp.py`: `spawn` + polling;
  - `08_advanced/poll_delayed_result.py`: `spawn` + polling `get(timeout=0)`.
- Ou seja: os exemplos **nao** implementam endpoint de cancelamento HTTP.
- O que os docs/sdks confirmam:
  - `FunctionCall.from_id(...)` e polling sao padrao nos exemplos/docs;
  - no SDK local validado aqui, `FunctionCall.cancel(self, terminate_containers: bool = False)` existe e a docstring informa cancelamento do call.
- O que fazer de forma objetiva:
  - salvar `function_call_id` quando fizer `spawn`;
  - no `DELETE /v1/jobs/{job_id}`, carregar esse ID e chamar `modal.FunctionCall.from_id(id).cancel()`;
  - manter sua parada cooperativa atual no worker (`comfyui_api.py:515-523`) como fallback.

3. **`user_id` obrigatorio + ownership**
- O que os exemplos fazem: nao implementam multi-tenant completo (isso fica para a aplicacao).
- O que fazer de forma objetiva:
  - tornar `user_id` obrigatorio no payload de criacao;
  - em `GET /v1/jobs/{id}`, negar acesso quando `job.user_id != caller.user_id`;
  - em `GET /v1/jobs`, exigir filtro por `user_id` do caller.

4. **Rate limit / quota**
- O que os exemplos/docs mostram:
  - `07_web_endpoints/basic_web.py` mostra `requires_proxy_auth=True` como camada de protecao de endpoint;
  - docs de web endpoints explicam proxy auth e autenticacao no proprio app.
- O que fazer de forma objetiva:
  - para endpoint backend-to-backend, considerar `requires_proxy_auth=True` + seu bearer interno;
  - aplicar quota/rate limit no gateway (por tenant/chave).

5. **Timeout de fila**
- O que os exemplos de fila fazem: `spawn` + polling de status, sem mutacao de estado em rota de leitura.
- O que esta comprovado no seu codigo: GET/list pode alterar estado para failed via `_check_queued_timeout` (`comfyui_api.py:870-883`, `comfyui_api.py:895`, `comfyui_api.py:932`).
- O que fazer de forma objetiva:
  - mover expiracao de fila para job supervisor/cron (fora do GET);
  - revisar SLA de timeout (120s costuma ser curto em disputa de GPU).

6. **Reemitir URL assinada**
- O que esta comprovado no seu codigo: voce salva `r2_key` por output (`comfyui_api.py:653`) e ja tem `generate_r2_url(...)` (`comfyui_api.py:292-309`).
- O que fazer de forma objetiva:
  - no GET de job `completed`, regenerar URL assinada para cada output com `r2_key`;
  - opcionalmente persistir a nova URL no job.

7. **Pinning de dependencias**
- O que os exemplos mostram:
  - `06_gpu_and_ml/image-to-video/image_to_video.py` usa versoes pinadas e revision SHA de modelo;
  - `02_building_containers` reforca reproducibilidade de imagem.
- O que esta comprovado no seu codigo: `torch/torchvision/torchaudio` sem versao fixa (`comfyui_api.py:152-157`).
- O que fazer de forma objetiva:
  - fixar versoes da stack torch/cuda e libs criticas no `gpu_image`;
  - registrar revisao dos modelos (quando aplicavel) para deploy deterministico.

## Conclusao
Seu projeto avancou muito e ja esta numa base solida para integrar com seu site em **beta fechado**. Para considerar “bom o suficiente” em **producao publica com 2 mil ativos/dia**, faltam principalmente os 2 P0 (SSRF localhost + cancelamento forte) e isolamento tenant mais rigido.

## Referencias
- https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml
- https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/comfyui
- https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/image-to-video
- https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/comfyui/comfyapp.py
- https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/comfyui/memory_snapshot/memory_snapshot_example.py
- https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/gpu_packing.py
- https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/gpu_fallbacks.py
- https://github.com/modal-labs/modal-examples/tree/main/02_building_containers
- https://github.com/modal-labs/modal-examples/tree/main/09_job_queues
- https://github.com/modal-labs/modal-examples/tree/main/07_web_endpoints
- https://github.com/modal-labs/modal-examples/tree/main/03_scaling_out
