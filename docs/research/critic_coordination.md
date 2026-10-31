# Critic: fleet coordination, 2026-10-31

**Verdict:** A small reactive layer is a sensible first step, but it is collision avoidance between robots heading to goals, not "coordinating on a job". The queued Colab run will likely deadlock, because the scene ranks robots by `--robot` order while the tests use `priority()`. Fix that first, then add obstacle-aware paths, one job dependency and a baseline run.

## 1. Is this the right next step?

A reviewer expects four things: a named method, static obstacles respected, a job with ordering (not just a goal swap), and a comparison against no coordination.

- **Current layer**: cheap and already wired. Call it "prioritized reactive yielding", not ORCA.
- **ORCA (Python-RVO2)**: a credible name for about 2 h of work, but it assumes holonomic robots (H1 probably is not, see 2), deadlocks in symmetric cases and needs obstacle polygons.
- **Report pick (A* on a grid plus reservation table)**: the right medium-term layer. The grid is what buys credibility in a reconstructed factory. Legged tracking error breaks timed reservations, so use cell locks with "wait until free" instead of timestamps.
- **Nav2 or Open-RMF**: what factory customers recognize (the RMF traffic schedule). ROS 2 in the container plus one adapter per robot type is not a one-day job, so it is next week's item.
- **Isaac**: the 6.2 multi-robot Nav2 tutorial uses wheeled Carters. CBS is overkill for four robots.

**Best trade for tomorrow:** keep this layer for local safety. Add grid A* from a top-down map of the scene, or confirm the corridor is clear and say so. Add one scripted dependency: robot B waits until A clears a station. Add the baseline run with avoidance off. Add the LLM allocator last, since it adds story, not safety.

## 2. Failure modes and cheapest fixes

- **Priority mismatch (verified, unicycle sim).** The scene passes `poses[:i]` as higher priority, which is list order. With that order on the queued layout, h1_2 (goal 4.6, 0) never arrives in 180 s. It sits behind the parked robots at x about 2.7. With `priority()` it arrives at 26 s. Fix: call `priority()` in the scene, and add a test using the scene's ranking.
- **Pushed off after arrival (verified).** Even with `priority()`, go2_1 arrives at 8.9 s, gets pushed aside by the passing H1, and ends 1.4 m from its goal at 180 s, held at the 1.2 m boundary (blocked and repelled). The latched `arrived` hides this. Fix: report the final distance to goal, and use a smaller clearance once a robot is parked.
- **Chatter (verified).** vx changes sign about 1770 times in 60 s at the clearance boundary. Fix: hysteresis (enter at 1.2 m, leave at 1.5 m), speed scaled by gap, and a command rate limit of about 1 m/s².
- **Out-of-range commands (unverified).** Isaac Lab's H1 velocity tasks train with lin_vel_y 0 and lin_vel_x 0 or more (from memory, check each `physx_env.yaml`). Repulsion sends H1 backward and sideways. Fix: clip to each yaml's ranges, so H1 yields by stopping and turning. Clamping H1 that way in the sim kept a 1.04 m minimum gap on the layout.
- **Repulsion overrides the block (from code).** The `away` branch ignores `blocked`. With a higher robot behind, it can drive forward into a lower one. Fix: zero positive vx while blocked.
- **Center distance, not surface gap.** Spot is about 1.1 m long (unverified), and the tests' 0.5 m threshold is contact for any pair. Fix: per-type radius.
- **Static obstacles ignored.** Whether the x -4 to 4.6 corridor in full_warehouse is clear is unverified. Check a top-down render.
- **Goals closer than the clearance** cause endless repulsion. Assert pairwise goal spacing above 1.5 m.
- **Small commands near the goal.** At 0.3 m vx is 0.18 m/s, and the policies may not track that (unverified). Fix: floor the speed at about 0.25 m/s.
- **Falls are not detected.** Log base height. Latency is fine at 50 Hz: a 0.3 s tracking lag only cut the sim's minimum gap from 1.19 m to 1.04 m.

## 3. Metrics to report

Report per robot: arrival time, final distance to goal (not latched), stall seconds (speed under 0.05 m/s while away from goal), falls and command sign flips. For the fleet: makespan, minimum surface gap, and near misses (gap under 0.3 m). Give the same numbers for the baseline run with avoidance off, and over 5 jittered starts, reporting the worst case.

## 4. "One shared policy driving several robots"

Not met. Four types means four policies. Each robot has its own `RobotPolicyRunner` stepped separately (from code). Whether weights load once per type is unverified. The honest claim today: "each robot runs its stock policy unchanged, and one fleet layer commands them all." To strengthen it, run a single-type fleet (four Go2 or four H1) through one batched forward pass `[N, obs]`, as `h1_standalone --num-robots` does, and show one policy file hash. Do not imply cross-embodiment.
