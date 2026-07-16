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

For services to start before an interactive login, an administrator must run
`loginctl enable-linger zhangyh` once.

## Acceptance checks

```sh
curl -fsS http://127.0.0.1:8001/api/auth/status
curl -fsS https://research.example.com/api/auth/status
systemctl --user --no-pager status llm-wiki-pilot.service llm-wiki-tunnel.service
```

The public auth status must report both `enabled` and `configured` as `true`.
Then complete login with two Feishu users and verify that projects/documents
are shared while Agent sessions, attachments, and traces remain private.
