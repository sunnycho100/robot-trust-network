# 4-slide plan (for slide generation)

Rule for every slide: one diagram, one headline, at most 5 short text elements. Everything else goes in the script.

---

## Slide 1. Overall architecture

**Headline:** Two houses, one gatekeeper, two halves.

**Diagram (the main asset of the deck):**
- Left box: **House A** — robot icon, label "own network, own container".
- Right box: **House B** — same.
- One arrow from B to A across the middle, labeled **request**.
- Where the arrow lands in House A, a vertical two-stage bar:
  - Stage 1 **VERIFY** — "is it really them?" (color 1, tag: Team 1)
  - Stage 2 **DECIDE** — "are they allowed?" (color 2, tag: Team 2)
- After the bar, two outputs: **robot acts** (arrow to the robot) and **owner log** (arrow to a phone or screen icon).
- Thin strip along the bottom: `now: trust network` → `next: open platform` → `later: Uber for humanoids`.

**On-slide text (5 items max):**
1. Request → Verify → Decide → Act
2. Each house is isolated by default
3. Team 1 owns Verify (authentication)
4. Team 2 owns Decide (authorization)
5. They merge at one function: `decide(request)`

**Script:** the two teams build separate halves of the same gatekeeper. Team 1 answers who is asking, Team 2 answers what they may do. They agree on the message format on day one and meet at one function, so neither waits for the other.

---

## Slide 2. Authentication (Team 1)

**Headline:** Prove the message is really from that robot.

**Diagram:**
- Center: an envelope showing the request fields, small and readable:
  `from · house · action · reason · time · signature`
  with a lock icon on the signature line.
- Four arrows hitting the envelope from outside and bouncing off, each labeled with one attack:
  **spoof** (pretend to be the neighbor) · **replay** (resend an old "open the door") · **tamper** (change the action) · **eavesdrop** (read it in transit).
- Below: two containers connected by a line labeled **MQTT + TLS**, showing this is where the message travels.

**On-slide text (5 items max):**
1. Fixed request format, any brand can read it
2. Signed with each robot's own key
3. Key registry: which key is which robot
4. Blocks spoof, replay, tamper, eavesdrop
5. Stack: Python, cryptography, MQTT/TLS, Docker

**Team 1 roadmap:**
- Week 1: freeze the message format, sign and verify one message
- Week 2: key registry, replay protection, TLS on the channel
- Week 3: two containers, separate ROS 2 domains, real request across them
- Week 4: run the four attacks, show each one blocked, add network delay and loss

---

## Slide 3. Authorization (Team 2)

**Headline:** Decide how far to follow.

**Diagram:**
- A funnel, left to right:
  **verified request** → [ **owner rules** ] → [ **trust score** ] → three outputs stacked:
  **allow** · **partial** · **deny**, each with a reason tag.
- Under the outputs, one line of the owner's feed as it would really read:
  "House 2's robot asked to open the front door. Denied."
- Small side box: trust tiers
  - unknown → information only
  - trusted neighbor → can ask for help
  - never, at any trust → door, camera, anything risking the owner

**On-slide text (5 items max):**
1. Verified request in, decision out
2. Owner rules first, trust score second
3. Three answers: allow, partial, deny, always with a reason
4. Trust rises with clean interactions, falls with failures
5. Stack: Python, YAML rules, SQLite, FastAPI, pytest

**Team 2 roadmap:**
- Week 1: write the rule file and the decision function with hand-written requests
- Week 2: trust scores and how they move, tiers mapped to allowed actions
- Week 3: plain-language log and a simple owner page
- Week 4: the scenario set both teams score against

---

## Slide 4. Market research (marketing track)

**Headline:** Who else is solving each piece, and is it new?

**Diagram:**
- A horizontal strip of our four pipeline stages: **transport · verify · decide · owner view**.
- Under each stage, two stacked bars to be filled in: **startups** vs **traditional companies**.
- One callout box: "more startups = newer field, more room to enter".
- Right side, a short list of deliverables with checkboxes (see below), so the slide shows output, not activity.

**On-slide text (5 items max):**
1. Map competitors onto our four stages
2. Count startups vs incumbents per stage
3. Deep dive: factory orchestration players, their pipeline and their go-to-market
4. Validate the idea in practitioner communities
5. Output: one-page competitive map, weekly

**Marketing roadmap (week 1 is what was already described, the rest is the broader arc):**
- Week 1: competitor list per stage, startup vs incumbent counts, one-page map
- Week 2: two factory orchestration case studies, their pipeline and how they sold it
- Week 3: post the idea in robotics communities, collect reactions and objections, bring back the three most common ones
- Week 4: who pays (robot makers, insurers, platform operators), how comparable developer tools are priced
- Ongoing: standards watch (VDA 5050, ISO 21423, Matter), funding rounds in the last 18 months, three quotes from practitioners describing the pain in their own words

---

## What to keep off the slides
- Component names (C1 to C7), file paths, library version numbers
- Full stack lists beyond one line
- The cost graph and the Uber vision, unless this deck is also the pitch. Those live in the other deck.
