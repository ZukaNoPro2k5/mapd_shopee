# Greedy BFS — case studies and report notes

> Purpose: slide/report material for why the Greedy BFS baseline is defensible in an online MAPD-style delivery problem. This is not runtime code.

## 1. Flatland Challenge / railway MAPF

**Reference:** Li et al., *Scalable Rail Planning and Replanning: Winning the 2020 Flatland Challenge*, ICAPS 2021, DOI `10.1609/icaps.v31i1.15994`.

**Problem shape:** trains move on a constrained grid/graph, dense traffic causes bottlenecks, and the system must produce real-time actions under uncertainty.

**Useful lesson for Graph Shopee:** the winning Flatland stack was not “pure RL”. It combined practical MAPF/OR components such as prioritized planning, safe-interval planning, large-neighborhood repair, and replanning. For our Greedy BFS, the directly portable idea is **priority-ordered one-step planning**: low-id/urgent/high-value shippers get a path first; later shippers yield or wait when conflicts appear.

**How it maps to our solver:**

- BFS gives the local shortest path on the grid.
- One-step conflict resolution approximates prioritized planning cheaply.
- Target stickiness prevents constant replanning churn when new orders appear.

## 2. Lifelong MAPD / warehouse robots

**Reference:** Ma, Li, Kumar, Koenig, *Lifelong Multi-Agent Path Finding for Online Pickup and Delivery Tasks*, AAMAS 2017.

**Problem shape:** agents repeatedly receive pickup-delivery tasks in an online stream, each task requires pickup then delivery, and paths must avoid collisions. The paper explicitly motivates this with warehouse robot domains.

**Useful lesson for Graph Shopee:** when real-time computation matters, decoupled task assignment plus individual path planning is often preferred over one global optimization over the whole future. The paper’s Token Passing family keeps shared assignments/paths and lets agents commit to tasks while new tasks arrive.

**How it maps to our solver:**

- We reserve picked target order IDs and pickup cells inside one timestep.
- We keep a lightweight per-shipper target memory to avoid everyone reselecting every step.
- We do not plan using future/hidden generation parameters.

## 3. Online MAPF / plan stability

**Reference:** Švancara et al., *Online Multi-Agent Pathfinding*, AAAI 2019, DOI `10.1609/aaai.v33i01.33017732`.

**Problem shape:** new agents/tasks appear while existing agents are already executing plans. The paper notes that optimal online MAPF is generally impossible under some variants, so practical solvers balance quality, runtime, and number of plan changes.

**Useful lesson for Graph Shopee:** the solver should not chase tiny score improvements every step. Replanning is useful, but excessive plan changes make agents oscillate and waste time around bottlenecks.

**How it maps to our solver:**

- A small stickiness bonus favors the current target unless a clearly better order appears.
- Urgent carried orders and full bags override pickup greed.
- Standing/yielding is allowed when a conflict would waste both shippers.

## 4. Dynamic pickup-and-delivery / insertion heuristics

**References:**

- Berbeglia et al., *Dynamic pickup and delivery problems*, European Journal of Operational Research, 2010.
- Luo & Schonfeld, *Online Rejected-Reinsertion Heuristics for Dynamic Multivehicle Dial-a-Ride Problem*, Transportation Research Record, 2011.

**Problem shape:** requests are revealed over time; routes are adjusted online instead of solved once statically. Immediate insertion and rolling-horizon insertion are common practical strategies.

**Useful lesson for Graph Shopee:** when a shipper already carries orders, a new pickup should be judged by marginal route damage, not raw reward alone.

**How it maps to our solver:**

- Pickup score now subtracts an insertion-detour estimate when the shipper already carries cargo.
- Same-destination delivery remains strongly rewarded because `cargo_op=2` can deliver multiple carried orders at one cell.
- Visible pickup/destination clustering uses only currently observed orders, so it stays legal under the v6 no-surge/no-hotspot rule.

## Slide-ready takeaway

Greedy BFS is not just “take nearest order”. The defensible version is:

```text
Online observation
  -> score visible tasks by reward, priority, deadline, distance, clustering, insertion detour
  -> reserve order/pickup targets to avoid duplicate pursuit
  -> BFS one step toward selected target
  -> resolve conflicts by simple priority/yield rules
  -> repeat every timestep without reading hidden future parameters
```

This gives a baseline that is explainable, offline-safe, and fast enough for Kaggle Phase 2.
