# Revisao de prontidao SaaS (Modal + ComfyUI)

Data: 2026-02-19
Base comparativa: `modal-labs/modal-examples`, pasta `06_gpu_and_ml` (com foco em `comfyui`, `memory_snapshot`, `gpu_packing`, `gpu_fallbacks`)

## Resultado rapido
- Seu código atual está **muito melhor** e alinhado com o exemplo oficial em pontos importantes (snapshot helper, health check, stop_fetching_inputs, output por URL assinada).
- Para **SaaS com 2 mil usuarios ativos/dia**, eu classifico como **quase pronto para beta fechado**, mas **ainda não pronto para produção pública** sem corrigir os P0 abaixo.

## Achados críticos (P0)
1. **Segredo sensível hardcoded no repositório**
- Evidência: `test_run.py:12`
- Impacto: vazamento direto de credencial de API; qualquer pessoa com acesso ao repo pode usar sua API.
- Ação: remover imediatamente, rotacionar chave, usar apenas secret/env em runtime.

2. **Cancelamento continua inconsistente com execução real do worker**
- Evidência: cancel apenas marca JSON como failed em `comfyui_api.py:872`, `comfyui_api.py:875`; worker não verifica flag de cancelamento durante loop e pode concluir depois (`comfyui_api.py:498` até `comfyui_api.py:655`).
- Impacto: usuário cancela, mas job pode continuar consumindo GPU e até virar `completed` depois.
- Ação: implementar cancelamento cooperativo no worker (checagem periódica no arquivo/estado + interrupção do prompt no ComfyUI do próprio container worker).

3. **SSRF ainda aberto para `media.url` e `webhook_url`**
- Evidência: download direto de URL em `comfyui_api.py:425` e webhook outbound em `comfyui_api.py:703`.
- Impacto: abuso de rede, acesso a endpoints internos/metadata, risco de exfiltração.
- Ação: bloquear IPs privados/localhost/link-local, allowlist de domínios, limitar redirects e tamanho de resposta, assinatura HMAC de webhook.

4. **Modelo de autenticação é global (sem isolamento por tenant)**
- Evidência: um único `API_KEY` em `comfyui_api.py:67`, validação simples em `comfyui_api.py:733`; listagem global em `comfyui_api.py:828`.
- Impacto: com múltiplos clientes/planos, qualquer cliente autenticado vê metadados de jobs de todos.
- Ação: introduzir `tenant_id/user_id` obrigatório em job + escopo por token; remover listagem global sem filtro de tenant.

## Achados altos (P1)
1. **Timeout de fila pode gerar falso-fail em pico**
- Evidência: `QUEUED_TIMEOUT_SECONDS = 300` em `comfyui_api.py:781`, mutação de estado no GET em `comfyui_api.py:809`.
- Impacto: em horários de fila longa, jobs legítimos podem ser marcados como failed.
- Ação: timeout dinâmico por fila/plano, ou fila explícita em broker (Redis/SQS) com worker pull.

2. **Concorrência de escrita no mesmo arquivo de job**
- Evidência: API e worker escrevem no mesmo JSON (`comfyui_api.py:773`, `comfyui_api.py:683`, `comfyui_api.py:876`).
- Impacto: condição de corrida em cancelamento/atualizações, especialmente sob carga.
- Ação: migrar estado de job para banco (Postgres/Redis) e deixar Volume para artefatos/cache.

3. **`max_containers=10` com `max_inputs=1` pode ficar caro e com fila em picos**
- Evidência: `comfyui_api.py:318`, `comfyui_api.py:321`.
- Comparação com examples: `gpu_packing.py` sugere aproveitar concorrência quando viável; docs recomendam calibrar `min_containers` e `buffer_containers`.
- Ação: testar `target_inputs`, `min_containers` e `buffer_containers`; medir throughput por workflow.

4. **Fallback para base64 ainda pode explodir payload**
- Evidência: fallback em `comfyui_api.py:629`-`comfyui_api.py:635`.
- Impacto: respostas grandes, custo de egress e latência.
- Ação: em produção, retornar erro transitório de storage ou retry de upload, evitando fallback base64 para arquivos grandes.

## Achados médios (P2)
1. **Inconsistência de nome de variável de segredo em docs/env example**
- Evidência: código espera `API_KEY` (`comfyui_api.py:67`, `comfyui_api.py:737`), mas `.env.example` usa `API_SECRET_KEY` (`.env.example:9`).
- Impacto: configuração errada em novos ambientes.

2. **Suite de testes não está operacional como teste automatizado**
- Evidência:
  - `pytest -q` (global) falha em import de `requests`.
  - `python -m pytest -q` no venv falha porque `pytest` não está instalado no venv.
- Impacto: sem validação automática confiável no CI.

3. **Sem rate limit no FastAPI**
- Evidência: não há middleware/limiter configurado em `comfyui_api.py`.
- Impacto: abuso de endpoint e variação de latência em pico.

## O que está muito bom (alinhado ao “santo graal”)
1. **Memory snapshot com helper dedicado**
- Evidência: `comfyui_api.py:330` e `comfyui_api.py:340`, helper em `memory_snapshot_helper/prestartup_script.py`.

2. **Health check com retirada do pool quando necessário**
- Evidência: `modal.experimental.stop_fetching_inputs()` em `comfyui_api.py:364`.
- Isso segue exatamente a direção do exemplo oficial de ComfyUI.

3. **Output em object storage + URL assinada**
- Evidência: upload R2 em `comfyui_api.py:618` e presigned URL em `comfyui_api.py:619`.
- Esse é um salto grande de arquitetura para SaaS.

4. **Separação API leve e worker GPU**
- Evidência: API ASGI separada em `comfyui_api.py:891`-`comfyui_api.py:900` e classe GPU em `comfyui_api.py:311`.

## GO / NO-GO para 2 mil DAU
- **NO-GO para produção pública agora** por causa dos P0 (principalmente segredo hardcoded, cancelamento, SSRF e isolamento por tenant).
- **GO para beta fechado controlado** após corrigir os P0.

## Prioridade de execução (ordem sugerida)
1. Remover segredo hardcoded + rotacionar chave de API.
2. Implementar cancelamento real no worker (cooperativo + interrupt do prompt do próprio worker).
3. Fechar SSRF (validação rígida de URL inbound/outbound).
4. Introduzir tenant-aware auth e escopo por usuário em `/v1/jobs` e `/v1/jobs/{id}`.
5. Ajustar fila/timeout e observabilidade para pico.
6. Consertar testes/CI e alinhar `.env.example` com `API_KEY`.

## Referências utilizadas
- https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml
- https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/comfyui
- https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/comfyui/comfyapp.py
- https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/comfyui/memory_snapshot/memory_snapshot_example.py
- https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/gpu_packing.py
- https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/gpu_fallbacks.py
- https://modal.com/docs/guide/scale
- https://modal.com/docs/guide/concurrent-inputs
- https://modal.com/docs/guide/volumes
