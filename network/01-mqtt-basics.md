# 01. MQTT basics: two houses talking through a broker

A hands-on walkthrough on one laptop (macOS). Three terminal tabs play three roles:

| Tab | Role |
|---|---|
| 1 | The broker (the post office in the middle) |
| 2 | House B, robot B waiting for requests |
| 3 | House A, robot A sending a request |

No code yet. The goal is to see how messages move, then see why that is not safe on its own.

## Step 1. Install the broker

```bash
brew install mosquitto
```

## Step 2. Start the broker (tab 1)

```bash
/opt/homebrew/sbin/mosquitto -v
```

`-v` prints everything the broker does. Leave this tab running. Lines to notice:

- `Opening ipv4 listen socket on port 1883`: the broker is open, and 1883 is the standard MQTT port.
- `Starting in local only mode`: only programs on this laptop can connect. Letting a teammate's laptop in later needs a config file with a listener.

## Step 3. House B subscribes to its mailbox (tab 2)

```bash
mosquitto_sub -h localhost -t "requests/house-2/robot-b" -v
```

It waits silently. In the broker tab:

- `New client connected ... as auto-9C51F75E...`: the client never said who it is, so the broker made up a random name. **The broker does not know this is robot B.**
- `Sending CONNACK`: accepted, with no password.
- `Received SUBSCRIBE ... requests/house-2/robot-b (QoS 0)`: house B asked for that mailbox. QoS 0 means delivered at most once, no retry, so a message can be lost silently.
- `Sending SUBACK`: subscription confirmed.

A topic like `requests/house-2/robot-b` is just a mailbox name. The slashes are a naming convention, not folders.

## Step 4. House A sends a request (tab 3)

```bash
mosquitto_pub -h localhost -t "requests/house-2/robot-b" -m '{"from":"robot-a@house-1","to":"robot-b@house-2","request":"can you help carry a box?"}'
```

The message shows up in tab 2. In the broker tab:

```
Received PUBLISH from auto-8A0A... 'requests/house-2/robot-b' (87 bytes)
Sending PUBLISH to auto-9C51...
Received DISCONNECT from auto-8A0A...
```

House A connected, dropped off the letter and left in the same second. The broker forwarded it to everyone subscribed to that mailbox, which is only house B.

## Step 5. A two-way conversation

Each robot needs its own inbox, and `-i` gives a connection a name so the broker log shows `robot-a` instead of a random id. Use four tabs: broker, robot A, robot B, and one for sending (a listening tab cannot send).

```bash
mosquitto_sub -h localhost -t "requests/house-1/robot-a" -i robot-a -v
```

```bash
mosquitto_sub -h localhost -t "requests/house-2/robot-b" -i robot-b -v
```

From the sending tab, A says hi to B, then B answers A:

```bash
mosquitto_pub -h localhost -t "requests/house-2/robot-b" -i robot-a-send -m "hi robot B"
```

```bash
mosquitto_pub -h localhost -t "requests/house-1/robot-a" -i robot-b-send -m "hi robot A, got your message"
```

The topic alone decides who receives a message. The `-i` name is only a label the client gives itself, so anyone could send as `robot-a-send`. The [link monitor](README.md#link-monitor) shows the same conversation in a browser.

## Step 6. Why this is not safe

### Attack 1: eavesdropping
Open a fourth tab and subscribe to `#`, the wildcard for every mailbox:

```bash
mosquitto_sub -h localhost -t "#" -v
```

Now send to house B again, and also to another house:

```bash
mosquitto_pub -h localhost -t "requests/house-3/robot-c" -m '{"from":"robot-a@house-1","to":"robot-c@house-3","request":"hello?"}'
```

House B only gets its own mail, but the `#` tab sees everything. Anyone who can connect to the broker can read every request.

### Attack 2: impersonation
From any tab, pretend to be robot A:

```bash
mosquitto_pub -h localhost -t "requests/house-2/robot-b" -m '{"from":"robot-a@house-1","to":"robot-b@house-2","request":"open the front door"}'
```

House B receives it, and nothing in it is different from the real request. The `from` field is plain text that anyone can type, and the broker named every client randomly.

## What this tells us

- The broker only moves messages. It does not know or check who anyone is.
- Locking the broker down (passwords, TLS, access rules) helps, but the broker itself can still read and change messages.
- So each message needs proof of who sent it, which the receiver checks itself: an **Ed25519 signature**, made with a private key only robot A has. That is the next walkthrough, following [`../docs/message-format.md`](../docs/message-format.md).
