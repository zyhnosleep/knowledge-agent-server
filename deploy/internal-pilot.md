# Internal Pilot Deployment

This deployment keeps the pilot API private on `127.0.0.1:8001` and exposes
only a named Cloudflare Tunnel hostname over HTTPS.

## Required operator values

- A Cloudflare-managed hostname, replacing `research.example.com`.
- A named Tunnel UUID and its credentials JSON.
- Feishu enterprise application ID and secret.
- The Feishu tenant key when access must be restricted to one tenant.

The Feishu redirect URI must exactly match:

```text
https://research.example.com/api/auth/callback
```

## Runtime files

Create `runtime/auth.env` from `.env.server.example`, use the deployed database
password and hostname, then set the authentication values. Protect the file:

```sh
chmod 600 runtime/auth.env
```

Copy `deploy/cloudflared/config.yml.example` to
`runtime/cloudflared/config.yml`, replace the hostname and Tunnel UUID, and put
the Tunnel credentials JSON in the same directory. Protect both files with
mode `600`.

Install the user services:

```sh
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llm-wiki-pilot.service llm-wiki-tunnel.service
```

The two GPU runtimes are separate user services. GPU 0 hosts the 14B fast
model and the embedding model on port 11435; GPU 1 hosts the 27B deep model
on port 11436. Both use Flash Attention, q8_0 KV cache, one generation at a
time, and permanent model residency:

```sh
systemctl --user enable --now \
  llm-wiki-ollama-fast.service llm-wiki-ollama-deep.service
OLLAMA_HOST=127.0.0.1:11435 ollama run qwen3:14b "warmup"
curl -fsS http://127.0.0.1:11435/api/embed \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding:8b","input":["warmup"]}' >/dev/null
OLLAMA_HOST=127.0.0.1:11436 ollama run qwen3.6:27b "warmup"
curl -fsS http://127.0.0.1:8001/api/health
```

The health response must report `api_status: ok` and all three model entries
as `ready`. A missing, unreachable, or non-resident model deliberately makes
the overall status `degraded` without marking the API process itself down.

For services to start before an interactive login, an administrator must run
`loginctl enable-linger zhangyh` once.

## Acceptance checks

```sh
curl -fsS http://127.0.0.1:8001/api/auth/status
curl -fsS https://research.example.com/api/auth/status
systemctl --user --no-pager status llm-wiki-pilot.service llm-wiki-tunnel.service \
  llm-wiki-ollama-fast.service llm-wiki-ollama-deep.service
```

The public auth status must report both `enabled` and `configured` as `true`.
Then complete login with two Feishu users and verify that projects/documents
are shared while Agent sessions, attachments, and traces remain private.
