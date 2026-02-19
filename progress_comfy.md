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

