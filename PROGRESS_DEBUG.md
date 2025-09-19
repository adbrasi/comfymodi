# API Progress Debug Notes

## Background

The deployment exposes a FastAPI wrapper for ComfyUI on Modal (`modal_comfyui_with_sageattention.py`). Each job is stored as `/jobs/<job_id>.json`, which is then surfaced via `GET /v1/jobs/{job_id}`. The CLI helper `test_wan22_api.py` polls that endpoint to print progress.

## Observed Behaviour

- ComfyUI logs (from `ComfyService.*`) clearly show real-time progress updates (e.g., `📊 Progress: 47% (Step 20/24)`), so the WebSocket listener receives the events.
- The REST status endpoint often returns `progress: 0` until the job completes, resulting in no updates in `test_wan22_api.py`.
- Even after redeployments, the CLI prints only the completion summary.

## Changes Attempted

1. **Frequent Persistence**
   - Updated `process_job` to write `progress` and `progress_details` to the job file on every `progress` event.
   - Removed deferred commit batching logic.

2. **Volume Reloads**
   - Eliminated status-endpoint caching so `/v1/jobs/{job_id}` reloads the volume on every request.
   - Ensured all writes happen under `job_volume_guard` for thread safety.

3. **test_wan22_api.py Enhancements**
   - Added configurable poll interval and timeout.
   - Printed sampler step and node whenever values change.

Despite these changes, the client still reads only the final state. This suggests the job JSON may not hit disk fast enough or the API is served from different containers with stale mounts.

## Likely Root Cause

In Modal, the ASGI app (`api`) runs in a separate function/volume mount from the `ComfyService` class. Although both attach the `job-storage` volume, it is possible the API container is not seeing new commits immediately (volume propagation delay). Direct filesystem reloads show stale data until the job finishes, supporting this theory.

## Next Diagnostic Steps

- Query status from within a running `ComfyService` container after a commit to confirm the file contains the updated progress.
- Use Modal’s `volume.reload()` with `volume.refresh()` (if available) right after commits to force synchronization.
- Introduce a simple debug endpoint that reads the job file within the worker process.

## Reference: ComfyUI Progress Monitoring

ComfyUI exposes a WebSocket at `/ws` that pushes messages during execution:

- `progress` or `progress_state`: includes `value` and `max` for sampler iterations.
- `executing`: indicates which node is currently running; a `null` node with the matching `prompt_id` means completion.
- `execution_cached`: signals cached node usage.
- `executed`: node finished successfully.

To monitor progress manually:

```python
import json
import uuid
import websocket
import urllib.request

client_id = str(uuid.uuid4())
ws = websocket.WebSocket()
ws.connect(f"ws://localhost:8188/ws?clientId={client_id}")

prompt_id = json.loads(urllib.request.urlopen(
    urllib.request.Request(
        "http://localhost:8188/prompt",
        json.dumps({"prompt": prompt_json, "client_id": client_id}).encode("utf-8")
    )
).read())["prompt_id"]

while True:
    message = json.loads(ws.recv())
    if message["type"] == "progress":
        data = message["data"]
        pct = data["value"] / data["max"] * 100
        print(f"Step {data['value']} / {data['max']} ({pct:.2f}%)")
    if message["type"] == "executing" and message["data"]["prompt_id"] == prompt_id:
        if message["data"]["node"] is None:
            print("Execution finished!")
            break
```

After completion, fetch artifacts via:

- `GET /history/{prompt_id}` for outputs metadata.
- `GET /view?...` for individual files.

## Modal Notes

- Each `@app.cls` worker (`ComfyService`) handles job execution. It commits job metadata to `/jobs`.
- The ASGI app (`api`) runs under `@modal.asgi_app()`, potentially across different containers than the worker.
- To force more immediate consistency, consider `job_volume.reload()` on each poll and adding a small delay (`time.sleep(0.5)`) after commits, though this increases latency.

---

**Summary**: We have progress events flowing through ComfyUI, but surface-level polling still shows stale data due to volume synchronization. Further investigation should concentrate on Modal’s volume propagation between the worker and API containers or redesigning progress reporting (e.g., storing status in Modal’s `dict`/Redis or returning updates via a WebSocket proxy).


comfyui progress portuguese guide:
bora resolver isso 👇

# O jeito “oficial” de acompanhar progresso no ComfyUI

1. **Abra um WebSocket** para `ws://<host>:8188/ws?clientId=<UUID>`.
2. **Envie o workflow** (JSON no formato “Save (API Format)”) via `POST /prompt` **usando o *mesmo* `client_id`**. A resposta traz `prompt_id`.
3. **Escute as mensagens do WS**:

   * `progress` → traz `data.value` e `data.max` (percentual = `value/max`).
   * `executing` → informa qual nó está rodando; quando vier `node: null` **com seu `prompt_id`**, a execução terminou.
   * Você também verá `status`, `execution_start`, `execution_cached` etc.
4. **Ao terminar**, busque as imagens por `GET /history/{prompt_id}` (ou receba frames binários se usar nó “SaveImageWebsocket”). ([docs.comfy.org][1])

> Observação: a mensagem se chama **`progress`** nas versões atuais (com `value` e `max`). Algumas fontes antigas falam em `progress_state`. Trate as duas por compatibilidade. ([GitHub][2])

## Exemplo — browser/Node (WebSocket nativo)

```js
// URL do seu ComfyUI
const SERVER = "127.0.0.1:8188";

// gere um UUID por request e use em /prompt e no WS
const clientId = crypto.randomUUID();
const ws = new WebSocket(`ws://${SERVER}/ws?clientId=${clientId}`);

let done = false;

ws.onmessage = (evt) => {
  if (typeof evt.data === "string") {
    const msg = JSON.parse(evt.data);

    if (msg.type === "progress" || msg.type === "progress_state") {
      const { value, max } = msg.data;
      const pct = max ? (value / max) * 100 : 0;
      console.log(`Progresso: ${pct.toFixed(1)}%`);
      // aqui você pode atualizar uma barra de progresso na UI
    }

    if (msg.type === "executing") {
      const { prompt_id, node } = msg.data;
      if (node) console.log("Executando nó:", node);
      if (node === null && prompt_id === window.__promptId) {
        console.log("Execução concluída");
        done = true;
        ws.close();
        // agora faça GET /history/{prompt_id} para obter as imagens
      }
    }
  } else {
    // evt.data é binário: se usar SaveImageWebsocket, isso pode ser imagem
  }
};

// depois de abrir o WS, envie o prompt
async function enqueue(promptJson) {
  const res = await fetch(`http://${SERVER}/prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: promptJson, client_id: clientId }),
  });
  const { prompt_id } = await res.json();
  window.__promptId = prompt_id;
  console.log("prompt_id:", prompt_id);
}
```

Esse fluxo (WS em `/ws` + POST em `/prompt` + GET em `/history`) é exatamente o que a UI faz, e é o caminho recomendado para feedback em tempo real. ([Hugging Face][3])

## Exemplo — Python (websocket-client)

```python
import uuid, json, urllib.request, urllib.parse
import websocket  # pip install websocket-client

SERVER = "127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())

def queue_prompt(prompt):
    body = json.dumps({"prompt": prompt, "client_id": CLIENT_ID}).encode("utf-8")
    req = urllib.request.Request(f"http://{SERVER}/prompt", data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{SERVER}/history/{prompt_id}") as r:
        return json.loads(r.read())

ws = websocket.WebSocket()
ws.connect(f"ws://{SERVER}/ws?clientId={CLIENT_ID}")

# ... monte seu 'prompt' (JSON exportado em "Save (API Format)")
resp = queue_prompt(prompt)
prompt_id = resp["prompt_id"]

finished = False
while not finished:
    out = ws.recv()
    if isinstance(out, str):
        msg = json.loads(out)
        if msg["type"] in ("progress", "progress_state"):
            value, maxv = msg["data"].get("value", 0), msg["data"].get("max", 0)
            pct = (value / maxv * 100) if maxv else 0
            print(f"Progresso: {pct:.1f}%")
        elif msg["type"] == "executing":
            data = msg["data"]
            if data["node"] is None and data["prompt_id"] == prompt_id:
                finished = True
ws.close()

hist = get_history(prompt_id)[prompt_id]
print("Nós com saída:", list(hist["outputs"].keys()))
```

Esse exemplo segue o **`websockets_api_example.py`** do projeto, adicionando leitura de mensagens `progress`. ([Hugging Face][3])

---

# Não posso usar WebSocket. E agora?

* **Polling simples**: faça `GET /history/{prompt_id}` em loop até aparecerem `outputs` com `images`. É funcional, mas **sem barra de progresso** (só “pronto / não pronto”). ([Hugging Face][3])
* **Fila**: `GET /queue` e `GET /prompt` (GET) mostram estado da fila/execução atual — úteis para posição/estado, mas novamente sem percentual fino. ([docs.comfy.org][1])

# RunPod (worker serverless do ComfyUI)

Em **Serverless** você normalmente **não consegue abrir WS do cliente para o ComfyUI** rodando dentro do *worker*. A alternativa é o próprio handler enviar *progress* para a API da RunPod:

1. No `rp_handler.py`, durante a execução, chame:

```python
import runpod
# ...
runpod.serverless.progress_update(job, {"step": cur_step, "max": total_steps})
```

2. No seu cliente, **interrogue `/status`** do job periodicamente para ler esses updates e mostrar a barra de progresso. ([Runpod Docs][4])

Se quiser **progresso do ComfyUI** (passo-a-passo do sampler) dentro da RunPod, faça um *fork* do worker, conecte-se localmente ao WS do ComfyUI (`ws://127.0.0.1:8188/ws?...`) a partir do `rp_handler.py`, repasse cada mensagem `progress` para `progress_update`, e retorne as imagens no `output` ao final. (O repositório base do worker descreve o formato de input/output; você pode estendê-lo). ([GitHub][5])

---

# Dicas que evitam dor de cabeça

* **Use o mesmo `client_id`** no POST `/prompt` e na URL do WS — isso garante que você receba os eventos certos do seu job. ([Hugging Face][3])
* **Combine sinais** para um progresso mais “suave”: mostre `%` de `progress` e, entre picos, use `executing` + (`executed_nodes`/`total_nodes`) quando esses campos vierem (alguns clients já tipam assim). ([app.unpkg.com][6])
* **Previews em tempo real**: com o nó **SaveImageWebsocket**, frames binários chegam direto pelo WS (sem salvar em disco). Útil para mostrar thumbs durante a amostragem. ([https://cnb.cool][7])
* **Fallback robusto**: se o WS falhar por rede, faça *polling* do `/history/{prompt_id}` como contingência. (Há relatos de instabilidade em cargas pesadas; trate *timeouts*.) ([GitHub][8])
* **Exportar workflow pro API**: na UI, habilite “Dev mode options” e use **“Save (API Format)”** para gerar o JSON certo. ([GitHub][9])

Se quiser, eu adapto esses snippets ao seu stack (Node/React, Python FastAPI, etc.) e ao seu workflow atual.

[1]: https://docs.comfy.org/development/comfyui-server/comms_routes?utm_source=chatgpt.com "Routes"
[2]: https://github.com/comfyanonymous/ComfyUI/issues/5118?utm_source=chatgpt.com "Progress indicator endpoint · Issue #5118"
[3]: https://huggingface.co/spideyrim/ComfyUI/blob/main/script_examples/websockets_api_example.py "script_examples/websockets_api_example.py · spideyrim/ComfyUI at main"
[4]: https://docs.runpod.io/serverless/workers/handler-functions?utm_source=chatgpt.com "Handler functions"
[5]: https://github.com/runpod-workers/worker-comfyui?utm_source=chatgpt.com "runpod-workers/worker-comfyui"
[6]: https://app.unpkg.com/%40node-ai/comfyui%400.1.20/files/dist/types/types/comfy.websocket.type.d.ts?utm_source=chatgpt.com "node-ai/comfyui"
[7]: https://cnb.cool/zdmwhy/Wan2.1_Vace/-/blob/ccf92a7e77f2a762d37622d2de4febb9b8c3125d/ComfyUI/script_examples/websockets_api_example_ws_images.py?utm_source=chatgpt.com "ComfyUI/script_examples/websockets_api_example_ws_images.py ..."
[8]: https://github.com/comfyanonymous/ComfyUI/issues/3128?utm_source=chatgpt.com "Websocket is very unstable and often stuck without ..."
[9]: https://github.com/flov/comfy-deploy-runpod-worker?utm_source=chatgpt.com "flov/comfy-deploy-runpod-worker: ComfyUI as a serverless ..."
