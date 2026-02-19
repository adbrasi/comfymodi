# Validacao Illustrious XL em API Modal (2026-02-19)

## Escopo
Teste E2E da API `comfyui-saas` com checkpoint `Illustrious-XL-v1.0.safetensors`, cobrindo:
1. Cold start.
2. Warm start.
3. Cold apos idle (scaledown).
4. Concorrencia com 3 requests simultaneos.
5. Evidencia de cache de modelo (sem re-download por request).

## Ambiente usado
- Endpoint: `https://cezarsaint--comfyui-saas-api.modal.run`
- Deploy de teste:
  - `GPU_MIN_CONTAINERS=0`
  - `GPU_BUFFER_CONTAINERS=0`
  - `GPU_SCALEDOWN_WINDOW_SECONDS=60`
  - `GPU_CONFIG=l40s,a100,a10g,any`
- Workflow base: `workflows/sdxl_simple_exampleV2.json`
- Override aplicado em todos os jobs:
  - `node=4`, `field=ckpt_name`, `value=Illustrious-XL-v1.0.safetensors`

## Evidencia de modelo no volume Modal
Inspecao remota do volume `comfyui-models-cache`:

- `checkpoints_count=2`
- `ChenkinNoob-XL-V0.2.safetensors` -> `7105349958 bytes`, `mtime=2026-02-19T19:40:44.717137Z`
- `Illustrious-XL-v1.0.safetensors` -> `6938040736 bytes`, `mtime=2026-02-19T21:31:21.920116Z`

Conclusao: o modelo Illustrious esta fisicamente no volume do app.

## Resultado sequencial (cold/warm/cold-after-idle)

1. `illustrious_run1` (cold esperado)
- Job: `74d918bf-bb99-4d50-becf-4df800b6d9f2`
- `queue_wait_s`: `35.72`
- `run_time_s`: `48.71`
- `total_time_s`: `84.43`
- Output: `tests_outputs/illustrious_run1_74d918bf_0.png`

2. `illustrious_run2` (warm esperado)
- Job: `ef2966b7-13a6-46cd-a0da-60dde34f4ed6`
- `queue_wait_s`: `2.25`
- `run_time_s`: `17.43`
- `total_time_s`: `19.68`
- Output: `tests_outputs/illustrious_run2_ef2966b7_0.png`

3. `illustrious_run3_cold_after_idle` (apos esperar 75s)
- Job: `8355dd4f-3581-490a-88c9-4f4849d50b87`
- `queue_wait_s`: `47.28`
- `run_time_s`: `121.51`
- `total_time_s`: `168.79`
- Output: `tests_outputs/illustrious_run3_cold_after_idle_8355dd4f_0.png`

Leitura tecnica:
- Warm start foi muito mais rapido que cold.
- Com `min_containers=0`, cold start ainda existe e aparece na fila.

## Resultado de concorrencia (3 requests simultaneos)

1. `illustrious_conc3`
- Job: `aa9f5a70-e100-4f12-8f7e-ecd78161c676`
- `queue_wait_s`: `15.21`
- `run_time_s`: `38.78`
- `total_time_s`: `54.00`
- Output: `tests_outputs/illustrious_conc3_aa9f5a70_0.png`

2. `illustrious_conc1`
- Job: `5354075b-759d-43ce-a905-440f81cb14cc`
- `queue_wait_s`: `33.92`
- `run_time_s`: `45.90`
- `total_time_s`: `79.82`
- Output: `tests_outputs/illustrious_conc1_5354075b_0.png`

3. `illustrious_conc2`
- Job: `7bb9609d-d331-4f6e-810c-cc1e49afbf16`
- `queue_wait_s`: `2.99`
- `run_time_s`: `113.97`
- `total_time_s`: `116.95`
- Output: `tests_outputs/illustrious_conc2_7bb9609d_0.png`

Leitura tecnica:
- Concorrencia funcionou com 3 jobs simultaneos.
- Latencia variou bastante entre jobs, coerente com fallback de GPU e cold/warm mistos.

## O modelo esta sendo baixado toda vez?
Resposta objetiva: **nao**.

Motivos verificaveis:
1. No codigo, o download ocorre em `download_models()` durante build da imagem (`run_function`), nao durante `run_job` da API.
2. Todos os jobs com Illustrious executaram com `override_ckpt_logged=true` e completaram normalmente.
3. O arquivo `Illustrious-XL-v1.0.safetensors` no volume tem `mtime=21:31:21Z`, enquanto os jobs rodaram depois (`~21:36` a `~21:42`), sem evidencias de reescrita por request.
4. Se houvesse re-download de ~6.9 GB por request, os tempos warm observados (`~19.68s total`) nao seriam possiveis.

## Arquivos de evidencia gerados
- Metricas completas: `tests_outputs/illustrious_e2e_metrics.json`
- Imagens geradas:
  - `tests_outputs/illustrious_run1_74d918bf_0.png`
  - `tests_outputs/illustrious_run2_ef2966b7_0.png`
  - `tests_outputs/illustrious_run3_cold_after_idle_8355dd4f_0.png`
  - `tests_outputs/illustrious_conc1_5354075b_0.png`
  - `tests_outputs/illustrious_conc2_7bb9609d_0.png`
  - `tests_outputs/illustrious_conc3_aa9f5a70_0.png`

## Conclusao de prontidao para SaaS
Funciona E2E com o modelo Illustrious, outputs em R2, e concorrencia basica validada.

Ponto critico para experiencia de produto:
- Com `GPU_MIN_CONTAINERS=0`, cold start ainda gera espera significativa.

Ajuste direto para producao:
1. Usar `GPU_MIN_CONTAINERS>=1` para reduzir cold start extremo.
2. Manter fallback (`GPU_CONFIG`) para disponibilidade, mas medir custo/latencia por GPU real.
3. Ajustar `buffer_containers` conforme pico para reduzir fila em bursts.
