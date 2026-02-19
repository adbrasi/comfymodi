# Checklist de Correcoes Obrigatorias (19/02/2026)

Objetivo: listar **somente** o que ainda precisa corrigir para reduzir risco real em producao (2k+ DAU), com base em testes executados e no estado atual do codigo.

## Critico (corrigir primeiro)
1. Endpoint esta parado por padrao operacional
- Evidencia: `modal app list` mostrou `comfyui-saas` em `stopped`; `/health` retorna `app ... is stopped`.
- Impacto: indisponibilidade total.
- Correcao:
  - definir runbook claro de subida/queda e quem pode rodar `modal app stop`.
  - se for ambiente produtivo, manter app deployado e observabilidade de uptime.

2. Limite por usuario nao e estrito sob burst curto
- Evidencia: `tests_outputs/quota_burst_check_20260219_200402.json` -> `accepted=6`, `rejected_429=2` com limite configurado `MAX_ACTIVE_JOBS_PER_USER=5`.
- Impacto: previsibilidade de custo/capacidade pior em pico.
- Correcao:
  - manter limite atual de ativos (ja existe) **e** adicionar rate limit no gateway (ex.: req/min por tenant).
  - se quiser limite estrito no backend, implementar “janela temporal” separada do controle de ativos.

3. `.env.example` desatualizado em relacao ao codigo atual (risco de custo alto involuntario)
- Evidencia:
  - `comfyui_api.py:50` => `GPU_MAX_CONTAINERS=2`
  - `comfyui_api.py:51` => `GPU_MIN_CONTAINERS=0`
  - `comfyui_api.py:52` => `GPU_BUFFER_CONTAINERS=0`
  - `comfyui_api.py:53` => `GPU_SCALEDOWN_WINDOW_SECONDS=60`
  - `comfyui_api.py:74` => `GPU_CONFIG` default sem `any`
  - `.env.example:12`..`.env.example:17` ainda traz valores agressivos (`20/1/1`, `any`, `API_MAX_CONTAINERS=10`)
- Impacto: deploy novo pode subir custo/latencia inesperados por configuracao incoerente.
- Correcao: alinhar `.env.example` aos defaults atuais e aos perfis (pico/off-peak).

## Alto
4. Documentacao interna ficou inconsistente com o codigo em pontos tecnicos
- Evidencia: `ANALISE_PRONTIDAO_SAAS_MODAL_v2_2026-02-19.md:120` afirma torch pinado (`torch==2.8.0` etc), mas codigo atual esta sem pin em `comfyui_api.py:192`.
- Impacto: decisao operacional baseada em informacao incorreta.
- Correcao:
  - ou pinar torch/torchvision/torchaudio no codigo,
  - ou corrigir documento para refletir o estado real.

5. Reprodutibilidade de build/modelo ainda fragil
- Evidencia:
  - PyTorch sem pin exato: `comfyui_api.py:192`
  - modelos HF sem `revision` fixa em `MODELS`/download.
  - custom nodes (quando usados) via `git clone --depth 1` sem commit fixo: `comfyui_api.py:170`.
- Impacto: comportamento pode mudar entre deploys sem mudanca de codigo.
- Correcao: pinar versoes e revisoes SHA (modelo + nodes + libs criticas).

6. Protecao de borda ainda opcional (e geralmente desligada)
- Evidencia: `comfyui_api.py:63`, `comfyui_api.py:1385` (`API_REQUIRE_PROXY_AUTH` default `0`).
- Impacto: endpoint mais exposto a abuso quando publicado.
- Correcao: em producao aberta, ligar `API_REQUIRE_PROXY_AUTH=1` e manter API key/JWT no gateway.

## Medio
7. Timeout de fila pode ser curto em pico real
- Evidencia: `QUEUED_TIMEOUT_SECONDS=240` em `comfyui_api.py:58`; cold real medido ~84s, e bursts mostraram esperas grandes.
- Impacto: falso negativo em pico (job falha por timeout de fila).
- Correcao: ajustar para faixa mais segura no perfil de carga real e aplicar retry no backend.

8. Persistencia de jobs em arquivos pode virar gargalo no crescimento
- Evidencia:
  - `list_jobs` varre e parseia arquivos JSON: `comfyui_api.py:1283`
  - muitos commits no volume ao longo do fluxo.
- Impacto: latencia e lock contention com aumento de volume.
- Correcao:
  - curto prazo: manter cleanup e limites.
  - medio prazo: migrar metadados de job para banco (status/index por tenant).

9. Webhook sem assinatura de integridade
- Evidencia: `_send_webhook` envia JSON simples sem assinatura/HMAC: `comfyui_api.py:832`.
- Impacto: backend receptor nao consegue validar origem de forma robusta.
- Correcao: adicionar assinatura HMAC e timestamp no header do webhook.

## Ja correto (nao mexer agora)
1. Auth basica + ownership:
- `verify_api_key` com compare_digest: `comfyui_api.py:863`
- `X-User-ID` valido por regex: `comfyui_api.py:871`
- ownership em `GET/DELETE`: `comfyui_api.py:1219`, `comfyui_api.py:1325`

2. Cancelamento:
- endpoint chama `FunctionCall.cancel()`: `comfyui_api.py:1359`
- testes recentes confirmaram `running -> cancelled` em benchmark.

3. SSRF/redirect/size limit:
- `_validate_external_url` e `_safe_fetch_external`: `comfyui_api.py:971`, `comfyui_api.py:1038`.

## Referencias de benchmark usadas nesta analise
- `tests_outputs/benchmark_latency_warm_cold_conc_cancel_20260219_200228.json`
- `tests_outputs/quota_burst_check_20260219_200402.json`
- `tests_outputs/post_auth_hardening_e7be18b3.png`

