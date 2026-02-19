# Validacao Geral das Ferramentas e Stress (2026-02-19)

## Escopo executado
1. Sanidade de API (`/health`).
2. Seguranca e contrato de entrada:
- API key invalida.
- header `X-User-ID` ausente.
- `workflow` vazio.
- mismatch `user_id` vs `X-User-ID`.
- bloqueio SSRF (`localhost`).
3. Fluxo E2E real:
- criar job, processar, baixar imagem gerada.
- ownership de job entre tenants.
- reemissao de URL assinada (R2) em `GET` repetido.
4. Cancelamento de job.
5. Stress de burst de criacao de jobs (8 simultaneos).
6. Stress de camada HTTP (sem GPU): bursts em `/health` e `/v1/jobs`.
7. Funcoes operacionais Modal:
- `verify_setup`
- `fail_stale_queued_jobs`
- `cleanup_old_jobs`

## Artefatos gerados
- JSON completo da validacao: `tests_outputs/full_general_validation_2026-02-19.json`
- Imagem E2E desta bateria: `test_outputs/fullcheck_e2e_4de4348c_0.png`
- Imagens Illustrious ja geradas/copiadas:
  - `test_outputs/illustrious_run1_74d918bf_0.png`
  - `test_outputs/illustrious_run2_ef2966b7_0.png`
  - `test_outputs/illustrious_run3_cold_after_idle_8355dd4f_0.png`
  - `test_outputs/illustrious_conc1_5354075b_0.png`
  - `test_outputs/illustrious_conc2_7bb9609d_0.png`
  - `test_outputs/illustrious_conc3_aa9f5a70_0.png`

## Resultado consolidado
- Total de testes: `19`
- Passaram: `17`
- Falharam: `2`

## O que passou
- Auth e validacoes de entrada funcionando.
- SSRF (localhost) bloqueado.
- E2E com output real e download ok.
- Isolamento multi-tenant por ownership ok (`404` para outro usuario).
- `GET /v1/jobs/{id}` reemite URL assinada (url muda entre chamadas).
- Burst API HTTP sem GPU:
  - `health_burst`: 120/120 OK, p50 585ms, p95 3739.5ms, max 4028.5ms
  - `list_burst`: 80/80 OK, p50 673.2ms, p95 2515.9ms, max 6167.8ms
- `verify_setup` confirmou ambiente:
  - `Models: 2 files, 13.1 GB total`
  - custom node `memory_snapshot_helper` presente.

## Falhas criticas encontradas

### 1) Cancelamento nao foi consistente no primeiro sinal
Evidencia:
- teste `cancel_reaches_terminal` falhou por timeout de polling.
- job ficou `running` com progresso `0` por varios minutos (`cd444bbd-7fca-415c-87be-27a3e81caf63`).
- novo `DELETE /v1/jobs/{id}` posterior mudou para `cancelled` imediatamente.

Leitura tecnica:
- ha condicao de corrida entre o `DELETE` e o worker no inicio de `run_job`.
- o worker pode sobrescrever estado para `running` usando snapshot antigo do arquivo.

Referencia de codigo:
- `comfyui_api.py:819` (create)
- `comfyui_api.py:1122` (cancel)
- `comfyui_api.py:430-520` (inicio de `run_job` e marcacao de `running`)

Risco em producao:
- cancelamento aparenta aceito na API, mas job pode continuar consumindo GPU.

### 2) Limite ativo por usuario nao segurou burst simultaneo
Evidencia:
- teste `stress_burst_enforces_user_limit` falhou.
- burst de 8 creates simultaneos para o mesmo user:
  - aceitos: `8`
  - rejeitados `429`: `0`
- esperado com `MAX_ACTIVE_JOBS_PER_USER=5`: aceitar no maximo 5 e rejeitar excedente.

Leitura tecnica:
- `_enforce_active_job_limits()` faz leitura + decisao sem lock transacional.
- requests concorrentes observam o mesmo estado antes das escritas dos demais.

Referencia de codigo:
- `comfyui_api.py:1002` (`_enforce_active_job_limits`)
- `comfyui_api.py:819` (fluxo `create_job` sem trava atomica)

Risco em producao:
- bursts reais podem ultrapassar limite por tenant e estourar custo/capacidade.

## Stress GPU (burst aceito)
Mesmo com falha de rate-limit, os 8 jobs aceitos terminaram:
- completed: `8`
- failed: `0`
- cancelled: `0`

Tempos (jobs completados):
- queue wait: min 1.68s, p50 24.74s, max 30.91s
- run time: min 13.17s, p50 21.07s, max 38.53s
- total: min 18.89s, p50 43.77s, max 68.98s

## Conclusao objetiva
Nao esta "sem surpresa" ainda. Existem 2 pontos de comportamento critico sob concorrencia/cancelamento que precisam ajuste antes de considerar robusto para carga diaria de SaaS:
1. Cancelamento com janela de corrida no start do worker.
2. Limite por usuario sem garantia atomica sob burst.

## Acao recomendada (prioridade alta)
1. Tornar cancelamento idempotente/forte no worker (recarregar estado antes de setar `running` e em checkpoints curtos).
2. Implementar controle atomico de quota (lock distribuido/contador transacional) para evitar race em `_enforce_active_job_limits`.
3. Reexecutar a mesma bateria apos patch; criterio de aceite:
- `cancel_reaches_terminal` = PASS
- `stress_burst_enforces_user_limit` = PASS (aceitos <= 5; 429 >= 3 no burst x8)
