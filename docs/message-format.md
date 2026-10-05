# Request message format (draft v0)

Draft for the network and simulation teams to agree on. Fields can change until both teams sign off.

```json
{
  "from": "robot-a@house-1",
  "to": "robot-b@house-2",
  "action": "carry_box",
  "request": "Can you help me carry this box to the porch?",
  "reason": "delivery arrived",
  "time": 1759680000,
  "nonce": "8f3c1a9e2b7d4f60",
  "sig": "<base64 Ed25519 signature>"
}
```

| Field | Meaning |
|---|---|
| `from` | Sender robot id, `robot@house` |
| `to` | Receiver robot id. Rejected if it is not the receiver, so a message cannot be bounced to another robot |
| `action` | Short machine-readable command, for tests |
| `request` | The same request in plain language, so other brands and the decide step can read the intent |
| `reason` | Why the sender is asking |
| `time` | Unix seconds when sent |
| `nonce` | Random one-time value, at least 16 hex characters |
| `sig` | Ed25519 signature over every other field |

## Signing
- Sign the canonical JSON of every field except `sig`: keys sorted, no spaces, UTF-8 bytes.
  Python: `json.dumps(msg, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")`
- `ensure_ascii=False` matters: Korean text in `request` or `reason` must sign the same in every language.
- Replies use the same format and are signed too, with `from` and `to` swapped.

## Checks on receive, in order
1. `to` is me
2. `from` is in my house's key registry and its status is `active`
3. Signature verifies with that key
4. `time` is within ±60 s of my clock
5. `nonce` not seen before. Keep each nonce until its own `time` + 60 s has passed (not 60 s after it arrived, which leaves a gap when clocks differ)

## Result passed to the decide step
```json
{
  "key_id": "robot-a@house-1",
  "registry_status": "active",
  "introduced_by_owner": true,
  "verified_at": 1759680001
}
```

## Open questions
- How keys get into each house's registry. Proposal: each house keeps its own registry, filled when owners pair in person (for example, scanning a QR code).
- Who runs the MQTT broker outside of testing.
