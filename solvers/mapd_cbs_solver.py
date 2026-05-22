"""Self-contained MAPD solver with rolling-horizon CBS-style conflict repair."""

from __future__ import annotations

import time
from collections import deque
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, valid_next_pos
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]
Path = List[Position]
Target = Tuple[str, int, Position, float]
PickupCommitment = Tuple[Position, int, int]
INF = 10**9
NEG_INF = -10**18
MOVES: Tuple[Move, ...] = ("S", "U", "D", "L", "R")
MOVE_ONLY: Tuple[Move, ...] = ("U", "D", "L", "R")

WINDOW = 10
MIN_WINDOW = 7
MAX_WINDOW = 14
MAX_REPLAN_ROUNDS = 5
MIN_PICKUP_SCORE = 0.0
DISTANCE_PENALTY = 0.02   # Closer to true move cost (0.01 * w)
LATE_PENALTY = 0.01       # Tiny penalty just for tie-breaking; rely on env's beta modifier
PRIORITY_BONUS = 5.0
SAME_DESTINATION_BONUS = 3.0
URGENCY_BONUS = 0.1

# Stickiness / clustering / detour / endgame
STICKY_TARGET_BONUS = 4.0
PICKUP_CLUSTER_RADIUS = 2
PICKUP_CLUSTER_BONUS = 0.8
VISIBLE_DESTINATION_BONUS = 1.0
INSERTION_DETOUR_PENALTY = 0.10
MIN_INSERTION_DELTA = -1.0
SLOT_PRESSURE_WEIGHT = 0.0
PICKUP_WAIT_PENALTY = 0.03
ENDGAME_WINDOW = 12
ROUTE_DELTA_WEIGHT = 0.0
COMMITMENT_STALE_TICKS = 3
COMMITMENT_DELIVERY_MARGIN = 12.0
ENABLE_PICKUP_COMMITMENT = False
MAX_EXACT_ROUTE_DESTINATIONS = 4
MAX_EXACT_ASSIGNMENT_SHIPPERS = 5
KBEST_ASSIGNMENTS = 4
KBEST_PICKUPS_PER_SHIPPER = 3
KBEST_TICK_BUDGET_SEC = 0.025
KBEST_CONFLICT_PENALTY = 0.25
KBEST_FIRST_STEP_CONFLICT_PENALTY = 1.5
KBEST_NO_PROGRESS_PENALTY = 1.0
KBEST_SWITCH_MARGIN = 8.0


class MAPDCBSSolver(Solver):
    """
    Online MAPD-CBS-lite solver.

    It uses deadline-aware task selection, plans short paths in a rolling horizon,
    detects vertex/swap conflicts, then repairs lower-priority paths with CBS-style
    vertex/edge constraints. It is deliberately self-contained and stdlib-only.
    """

    method_name = "MAPD-CBS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._blocked_counts: Dict[int, int] = {}
        self._recent_conflicts = 0
        self._target_by_shipper: Dict[int, Tuple[str, int]] = {}
        self._assignment_candidates: List[
            Tuple[float, Dict[int, Optional[Target]]]
        ] = []
        self._kbest_eval_ticks = 0
        self._kbest_changed_ticks = 0
        self._kbest_heuristic_score_total = 0.0
        self._kbest_path_score_total = 0.0
        self._last_kbest_evaluated = 1
        self._last_selected_targets: Dict[int, Optional[Target]] = {}
        self._last_chosen_conflicts = 0
        self._pickup_commitments: Dict[int, PickupCommitment] = {}
        self._insertion_pickup_evaluations = 0
        self._insertion_rejects = 0
        self._insertion_rejected_orders: Set[int] = set()
        self._slot_last_slot_evaluations = 0
        self._slot_high_load_evaluations = 0
        self._slot_pressure_orders: Set[int] = set()
        self._commitment_kept_ticks = 0
        self._commitment_drop_reasons: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Grid/path helpers
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if move == "S" or nxt != pos:
                yield move, nxt

    def _move_between(self, start: Position, nxt: Position) -> Move:
        if nxt == start:
            return "S"
        for move in MOVE_ONLY:
            if valid_next_pos(start, move, self.grid) == nxt:
                return move
        return "S"

    def _distance(self, p1: Position, p2: Position) -> int:
        if p1 == p2:
            return 0
        pair = (p1, p2)
        if pair in self._distance_cache:
            return self._distance_cache[pair]

        queue = deque([(p1, 0)])
        visited = {p1}

        while queue:
            curr, dist = queue.popleft()
            if curr == p2:
                self._distance_cache[pair] = dist
                self._distance_cache[(p2, p1)] = dist
                return dist

            for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (curr[0] + d[0], curr[1] + d[1])
                if 0 <= nxt[0] < len(self.grid) and 0 <= nxt[1] < len(self.grid[0]):
                    if self.grid[nxt[0]][nxt[1]] == 0 and nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, dist + 1))

        self._distance_cache[pair] = INF
        self._distance_cache[(p2, p1)] = INF
        return INF

    def _delivery_time_after_distance(self, t: int, distance: int) -> int:
        if distance >= INF:
            return INF
        return t + max(distance - 1, 0)

    def _pad_path(self, path: Path, horizon: int) -> Path:
        if not path:
            return []
        while len(path) < horizon + 1:
            path.append(path[-1])
        return path[: horizon + 1]

    # ------------------------------------------------------------------
    # Scoring/task selection
    # ------------------------------------------------------------------
    def _carried_orders(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]

    def _same_destination_count(
        self,
        shipper: Shipper,
        destination: Position,
        orders: Dict[int, Order],
    ) -> int:
        return sum(
            1
            for order in self._carried_orders(shipper, orders)
            if (order.ex, order.ey) == destination
        )

    # --- NEW: stickiness, clustering, detour, route scoring ---

    def _stickiness_bonus(self, shipper: Shipper, kind: str, order: Order) -> float:
        if self._target_by_shipper.get(shipper.id) == (kind, order.id):
            return STICKY_TARGET_BONUS
        return 0.0

    def _visible_pickup_cluster_bonus(self, order: Order, orders: Dict[int, Order]) -> float:
        pickup = (order.sx, order.sy)
        bonus = 0.0
        for other in orders.values():
            if other.id == order.id or other.picked or other.delivered:
                continue
            dist = abs(other.sx - pickup[0]) + abs(other.sy - pickup[1])
            if dist == 0:
                bonus += 1.5 * PICKUP_CLUSTER_BONUS
            elif dist <= PICKUP_CLUSTER_RADIUS:
                bonus += PICKUP_CLUSTER_BONUS / dist
        return bonus

    def _visible_destination_cluster_bonus(self, order: Order, orders: Dict[int, Order]) -> float:
        dest = (order.ex, order.ey)
        same = near = 0
        for other in orders.values():
            if other.id == order.id or other.picked or other.delivered:
                continue
            d = abs(other.ex - dest[0]) + abs(other.ey - dest[1])
            if d == 0:
                same += 1
            elif d <= 1:
                near += 1
        return VISIBLE_DESTINATION_BONUS * same + 0.5 * VISIBLE_DESTINATION_BONUS * near

    def _insertion_detour(
        self, shipper: Shipper, order: Order, orders: Dict[int, Order],
    ) -> int:
        carried = self._carried_orders(shipper, orders)
        if not carried:
            return 0
        current = shipper.position
        pickup = (order.sx, order.sy)
        new_dest = (order.ex, order.ey)
        to_pickup = self._distance(current, pickup)
        pickup_to_new = self._distance(pickup, new_dest)
        if to_pickup >= INF or pickup_to_new >= INF:
            return INF
        best_detour = INF
        for co in carried:
            cd = (co.ex, co.ey)
            direct = self._distance(current, cd)
            if direct >= INF:
                continue
            new_to_c = self._distance(new_dest, cd)
            if new_to_c < INF:
                best_detour = min(best_detour, max(0, to_pickup + pickup_to_new + new_to_c - direct))
            p_to_c = self._distance(pickup, cd)
            c_to_new = self._distance(cd, new_dest)
            if p_to_c < INF and c_to_new < INF:
                best_detour = min(best_detour, max(0, to_pickup + p_to_c + c_to_new - direct))
        return best_detour

    def _route_delivery_time(self, start_t: int, cum_dist: int, after_pickup: bool) -> int:
        if after_pickup:
            return start_t + max(cum_dist, 1)
        return start_t + max(cum_dist - 1, 0)

    def _fallback_destination_route(
        self,
        groups: Dict[Position, List[Order]],
        start: Position,
    ) -> Tuple[Position, ...]:
        remaining = set(groups)
        position = start
        route: List[Position] = []
        while remaining:
            destination = min(
                remaining,
                key=lambda dest: (
                    min(order.et for order in groups[dest]),
                    self._distance(position, dest),
                    -max(order.p for order in groups[dest]),
                ),
            )
            route.append(destination)
            remaining.remove(destination)
            position = destination
        return tuple(route)

    def _destination_routes(
        self,
        groups: Dict[Position, List[Order]],
        start: Position,
    ) -> Iterable[Tuple[Position, ...]]:
        destinations = tuple(groups)
        if len(destinations) <= MAX_EXACT_ROUTE_DESTINATIONS:
            return permutations(destinations)
        return (self._fallback_destination_route(groups, start),)

    def _route_score(
        self,
        route_orders: List[Order],
        start: Position,
        start_t: int,
        horizon: int,
        after_pickup: bool = False,
    ) -> float:
        if not route_orders:
            return 0.0
        groups: Dict[Position, List[Order]] = {}
        for o in route_orders:
            groups.setdefault((o.ex, o.ey), []).append(o)
        best = NEG_INF
        for perm in self._destination_routes(groups, start):
            pos = start
            cum = 0
            score = 0.0
            ok = True
            for dest in perm:
                leg = self._distance(pos, dest)
                if leg >= INF:
                    ok = False
                    break
                cum += leg
                dt = self._route_delivery_time(start_t, cum, after_pickup)
                for o in groups[dest]:
                    late = max(0, dt - o.et)
                    penalty = 40.0 if dt >= horizon else 0.0
                    score += delivery_reward(o, dt, horizon) + PRIORITY_BONUS * o.p - LATE_PENALTY * late - penalty
                pos = dest
            if not ok:
                continue
            score -= DISTANCE_PENALTY * cum
            best = max(best, score)
        return best

    def _best_delivery_route(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> Optional[Target]:
        carried = self._carried_orders(shipper, orders)
        if not carried:
            return None
        groups: Dict[Position, List[Order]] = {}
        for o in carried:
            groups.setdefault((o.ex, o.ey), []).append(o)
        best_order: Optional[Order] = None
        best_score = NEG_INF
        for perm in self._destination_routes(groups, shipper.position):
            route = [o for d in perm for o in groups[d]]
            score = self._route_score(route, shipper.position, t, horizon)
            if score <= NEG_INF / 2:
                continue
            first_group = groups[perm[0]]
            first = min(first_group, key=lambda o: (-o.p, o.et, o.id))
            score += self._stickiness_bonus(shipper, "delivery", first)
            if score > best_score:
                best_score = score
                best_order = first
        if best_order is None:
            return None
        return ("delivery", best_order.id, (best_order.ex, best_order.ey), best_score)

    def _endgame_started(self, t: int, horizon: int) -> bool:
        return horizon - t <= ENDGAME_WINDOW

    def _endgame_pickup_allowed(
        self,
        shipper: Shipper,
        order: Order,
        t: int,
        horizon: int,
    ) -> bool:
        if not self._endgame_started(t, horizon):
            return True
        pickup = (order.sx, order.sy)
        destination = (order.ex, order.ey)
        to_pickup = self._distance(shipper.position, pickup)
        trip = self._distance(pickup, destination)
        if to_pickup >= INF or trip >= INF:
            return False

        delivery_t = t + max(to_pickup - 1, 0) + trip
        if delivery_t >= horizon:
            return False
        if delivery_t <= order.et:
            return True

        late_reward = delivery_reward(order, delivery_t, horizon)
        pessimistic_move_cost = 0.03 * (to_pickup + trip)
        return late_reward > pessimistic_move_cost

    # --- Enhanced scoring ---

    def _delivery_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> float:
        destination = (order.ex, order.ey)
        distance = self._distance(shipper.position, destination)
        if distance >= INF:
            return NEG_INF
        delivery_t = self._delivery_time_after_distance(t, distance)
        reward = delivery_reward(order, delivery_t, horizon)
        lateness = max(0, delivery_t - order.et)
        slack = max(0, order.et - delivery_t)
        same_destination = self._same_destination_count(shipper, destination, orders)
        return (
            reward
            + PRIORITY_BONUS * order.p
            + self._stickiness_bonus(shipper, "delivery", order)
            + SAME_DESTINATION_BONUS * max(0, same_destination - 1)
            + URGENCY_BONUS * max(0, horizon - t - slack)
            - DISTANCE_PENALTY * distance
            - LATE_PENALTY * lateness
        )

    def _carried_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(
            orders[order_id].w
            for order_id in shipper.bag
            if order_id in orders
        )

    def _slot_pressure_penalty(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        delivery_t: int,
    ) -> float:
        slots_after = shipper.K_max - len(shipper.bag) - 1
        if slots_after >= 2:
            penalty = 0.0
        elif slots_after == 1:
            penalty = 2.0
        else:
            penalty = 8.0
            self._slot_last_slot_evaluations += 1

        load_ratio_after = (
            self._carried_weight(shipper, orders) + order.w
        ) / max(shipper.W_max, 1.0)
        if load_ratio_after > 0.8:
            penalty += min(3.0, 15.0 * (load_ratio_after - 0.8))
            self._slot_high_load_evaluations += 1

        if penalty <= 0.0:
            return 0.0

        carried_destinations = {
            (carried.ex, carried.ey)
            for carried in self._carried_orders(shipper, orders)
        }
        if (order.ex, order.ey) in carried_destinations:
            penalty *= 0.5
        if order.p == 3 and delivery_t <= order.et:
            penalty *= 0.5
        weighted_penalty = SLOT_PRESSURE_WEIGHT * penalty
        if weighted_penalty > 0.0:
            self._slot_pressure_orders.add(order.id)
        return weighted_penalty

    def _insertion_pickup_score(
        self,
        shipper: Shipper,
        order: Order,
        carried: List[Order],
        orders: Dict[int, Order],
        t: int,
        horizon: int,
        to_pickup: int,
        trip: int,
    ) -> float:
        self._insertion_pickup_evaluations += 1
        pickup = (order.sx, order.sy)
        destination = (order.ex, order.ey)
        pickup_t = t + max(to_pickup - 1, 0)
        current_route = self._route_score(
            carried,
            shipper.position,
            t,
            horizon,
        )
        after_pickup_route = self._route_score(
            carried + [order],
            pickup,
            pickup_t,
            horizon,
            after_pickup=True,
        )
        if (
            current_route <= NEG_INF / 2
            or after_pickup_route <= NEG_INF / 2
        ):
            self._insertion_rejects += 1
            self._insertion_rejected_orders.add(order.id)
            return NEG_INF

        insertion_delta = after_pickup_route - current_route
        if insertion_delta < MIN_INSERTION_DELTA:
            self._insertion_rejects += 1
            self._insertion_rejected_orders.add(order.id)
            return NEG_INF

        delivery_t = pickup_t + trip
        same_destination = self._same_destination_count(
            shipper,
            destination,
            orders,
        )
        return (
            insertion_delta
            + self._stickiness_bonus(shipper, "pickup", order)
            + SAME_DESTINATION_BONUS * same_destination
            + self._visible_pickup_cluster_bonus(order, orders)
            + self._visible_destination_cluster_bonus(order, orders)
            - DISTANCE_PENALTY * to_pickup
            - PICKUP_WAIT_PENALTY * max(0, t - order.appear_t)
            - self._slot_pressure_penalty(
                shipper,
                order,
                orders,
                delivery_t,
            )
        )

    def _pickup_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> float:
        if not shipper.can_carry(order, orders):
            return NEG_INF
        pickup = (order.sx, order.sy)
        destination = (order.ex, order.ey)
        to_pickup = self._distance(shipper.position, pickup)
        trip = self._distance(pickup, destination)
        if to_pickup >= INF or trip >= INF:
            return NEG_INF
        delivery_t = t + max(to_pickup - 1, 0) + trip
        reward = delivery_reward(order, delivery_t, horizon)
        lateness = max(0, delivery_t - order.et)
        same_destination = self._same_destination_count(shipper, destination, orders)

        detour = 0
        route_delta = 0.0
        carried = self._carried_orders(shipper, orders)
        if carried:
            detour = self._insertion_detour(shipper, order, orders)
            if detour >= INF:
                return NEG_INF
            if ROUTE_DELTA_WEIGHT:
                current_route = self._route_score(
                    carried,
                    shipper.position,
                    t,
                    horizon,
                )
                pickup_t = t + max(to_pickup - 1, 0)
                new_route = self._route_score(
                    carried + [order],
                    pickup,
                    pickup_t,
                    horizon,
                    after_pickup=True,
                )
                route_delta = new_route - current_route

        return (
            reward
            + ROUTE_DELTA_WEIGHT * route_delta
            + PRIORITY_BONUS * order.p
            + self._stickiness_bonus(shipper, "pickup", order)
            + SAME_DESTINATION_BONUS * same_destination
            + self._visible_pickup_cluster_bonus(order, orders)
            + self._visible_destination_cluster_bonus(order, orders)
            - DISTANCE_PENALTY * (to_pickup + trip)
            - INSERTION_DETOUR_PENALTY * detour
            - LATE_PENALTY * lateness
            - PICKUP_WAIT_PENALTY * max(0, t - order.appear_t)
        )

    def _has_delivery_at_position(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        position: Position,
    ) -> bool:
        return any(
            (order.ex, order.ey) == position
            for order in self._carried_orders(shipper, orders)
        )

    def _best_delivery_target(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> Optional[Target]:
        # Use route-aware TSP when carried orders > 1
        carried = self._carried_orders(shipper, orders)
        if len(carried) > 1:
            route_target = self._best_delivery_route(shipper, orders, t, horizon)
            if route_target is not None:
                return route_target
        # Fallback to single-order scoring
        best_delivery: Optional[Order] = None
        best_delivery_score = NEG_INF
        for order in carried:
            score = self._delivery_score(shipper, order, orders, t, horizon)
            if score > best_delivery_score:
                best_delivery = order
                best_delivery_score = score
        if best_delivery is not None:
            return (
                "delivery",
                best_delivery.id,
                (best_delivery.ex, best_delivery.ey),
                best_delivery_score,
            )
        return None

    def _needs_delivery_first(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> bool:
        carried = self._carried_orders(shipper, orders)
        if not carried:
            return False
        if len(shipper.bag) >= shipper.K_max:
            return True
        remaining = horizon - t
        for order in carried:
            destination = (order.ex, order.ey)
            distance = self._distance(shipper.position, destination)
            delivery_t = self._delivery_time_after_distance(t, distance)
            if delivery_t + 2 >= order.et:
                return True
            if remaining <= max(12, distance + 6):
                return True
        return False

    def _pickup_orders_by_cell(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
    ) -> Iterable[Order]:
        """
        Yield the order the environment would pick for each visible pickup cell.

        DeliveryEnv does not pick the order referenced by a solver target. It
        selects the best carryable order at the shipper's actual pickup cell.
        """
        orders_by_cell: Dict[Position, List[Order]] = {}
        for order in orders.values():
            if order.picked or order.delivered:
                continue
            orders_by_cell.setdefault((order.sx, order.sy), []).append(order)

        for cell_orders in orders_by_cell.values():
            carryable = [
                order
                for order in cell_orders
                if shipper.can_carry(order, orders)
            ]
            if carryable:
                yield min(carryable, key=lambda order: (-order.p, order.et, order.id))

    def _pickup_order_at_cell(
        self,
        shipper: Shipper,
        pickup: Position,
        orders: Dict[int, Order],
    ) -> Tuple[Optional[Order], Optional[str]]:
        cell_orders = [
            order
            for order in orders.values()
            if (
                not order.picked
                and not order.delivered
                and (order.sx, order.sy) == pickup
            )
        ]
        if not cell_orders:
            return None, "invalid_cell"
        carryable = [
            order
            for order in cell_orders
            if shipper.can_carry(order, orders)
        ]
        if not carryable:
            return None, "cannot_carry"
        return min(carryable, key=lambda order: (-order.p, order.et, order.id)), None

    def _pickup_cell_score(
        self,
        shipper: Shipper,
        expected_order: Order,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> float:
        """Score the real pickup plus visible follow-up work at the same cell."""
        pickup = (expected_order.sx, expected_order.sy)
        best_cell_score = self._pickup_score(
            shipper,
            expected_order,
            orders,
            t,
            horizon,
        )
        if best_cell_score <= NEG_INF / 2:
            return NEG_INF
        for order in orders.values():
            if (
                order.id == expected_order.id
                or order.picked
                or order.delivered
                or (order.sx, order.sy) != pickup
                or not shipper.can_carry(order, orders)
            ):
                continue
            best_cell_score = max(
                best_cell_score,
                self._pickup_score(shipper, order, orders, t, horizon),
            )
        return best_cell_score

    def _drop_pickup_commitment(self, shipper_id: int, reason: str) -> None:
        if shipper_id not in self._pickup_commitments:
            return
        self._pickup_commitments.pop(shipper_id, None)
        self._commitment_drop_reasons[reason] = (
            self._commitment_drop_reasons.get(reason, 0) + 1
        )

    def _committed_pickup_target(
        self,
        shipper: Shipper,
        best_delivery: Optional[Target],
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> Optional[Target]:
        commitment = self._pickup_commitments.get(shipper.id)
        if commitment is None:
            return None

        pickup, last_distance, stale_ticks = commitment
        order, invalid_reason = self._pickup_order_at_cell(shipper, pickup, orders)
        if order is None:
            self._drop_pickup_commitment(
                shipper.id,
                invalid_reason or "invalid_cell",
            )
            return None

        score = self._pickup_cell_score(shipper, order, orders, t, horizon)
        if score <= NEG_INF / 2:
            self._drop_pickup_commitment(shipper.id, "invalid_cell")
            return None
        if (
            best_delivery is not None
            and best_delivery[3] >= score + COMMITMENT_DELIVERY_MARGIN
        ):
            self._drop_pickup_commitment(shipper.id, "delivery_override")
            return None

        distance = self._distance(shipper.position, pickup)
        if distance >= INF:
            self._drop_pickup_commitment(shipper.id, "invalid_cell")
            return None
        stale_ticks = 0 if distance < last_distance else stale_ticks + 1
        if stale_ticks >= COMMITMENT_STALE_TICKS:
            self._drop_pickup_commitment(shipper.id, "stalled")
            return None

        self._pickup_commitments[shipper.id] = (pickup, distance, stale_ticks)
        self._commitment_kept_ticks += 1
        return ("pickup", order.id, pickup, score)

    def _pickup_reservations(
        self,
        targets: Dict[int, Optional[Target]],
    ) -> Tuple[Set[int], Set[Position]]:
        reserved_orders: Set[int] = set()
        reserved_pickups: Set[Position] = set()
        for target in targets.values():
            if target is None or target[0] != "pickup":
                continue
            reserved_orders.add(target[1])
            reserved_pickups.add(target[2])
        return reserved_orders, reserved_pickups

    def _assign_targets_global(
        self,
        shippers: List[Shipper],
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> Dict[int, Optional[Target]]:
        targets: Dict[int, Optional[Target]] = {shipper.id: None for shipper in shippers}
        assigned_shippers: Set[int] = set()
        best_delivery: Dict[int, Optional[Target]] = {}

        for shipper in shippers:
            target = self._best_delivery_target(shipper, orders, t, horizon)
            best_delivery[shipper.id] = target
            if target is not None and self._needs_delivery_first(
                shipper, orders, t, horizon,
            ):
                self._drop_pickup_commitment(shipper.id, "forced_delivery")
                targets[shipper.id] = target
                assigned_shippers.add(shipper.id)
                continue

            if ENABLE_PICKUP_COMMITMENT:
                committed_target = self._committed_pickup_target(
                    shipper,
                    target,
                    orders,
                    t,
                    horizon,
                )
                if committed_target is not None:
                    targets[shipper.id] = committed_target
                    assigned_shippers.add(shipper.id)

        forced_targets = dict(targets)
        available_shippers = [s for s in shippers if s.id not in assigned_shippers]
        forced_orders, forced_pickups = self._pickup_reservations(forced_targets)

        # Precompute all scores for available shippers
        score_matrix: Dict[int, List[Tuple[float, Target]]] = {}
        for shipper in available_shippers:
            score_matrix[shipper.id] = []

            # Delivery option (if any)
            delivery_target = best_delivery.get(shipper.id)
            if delivery_target is not None:
                score = delivery_target[3] + 2.0
                score_matrix[shipper.id].append((score, delivery_target))

            # Pickup options
            for order in self._pickup_orders_by_cell(shipper, orders):
                score = self._pickup_cell_score(shipper, order, orders, t, horizon)
                if score >= MIN_PICKUP_SCORE:
                    target = ("pickup", order.id, (order.sx, order.sy), score)
                    score_matrix[shipper.id].append((score, target))

            # Sort highest score first
            score_matrix[shipper.id].sort(key=lambda x: x[0], reverse=True)

        best_assignment = {}
        best_total_score = NEG_INF
        best_reserved_orders = set()
        best_reserved_pickups = set()

        if len(available_shippers) <= MAX_EXACT_ASSIGNMENT_SHIPPERS:
            assignment_orders: Iterable[Tuple[Shipper, ...]] = permutations(
                available_shippers,
            )
        else:
            assignment_orders = ()

        for perm in assignment_orders:
            current_assignment = {}
            current_reserved_orders = set(forced_orders)
            current_reserved_pickups = set(forced_pickups)
            total_score = 0.0

            for shipper in perm:
                for score, target in score_matrix[shipper.id]:
                    kind, order_id, goal, _ = target

                    if kind == "pickup":
                        if order_id in current_reserved_orders or goal in current_reserved_pickups:
                            continue
                        current_assignment[shipper.id] = target
                        current_reserved_orders.add(order_id)
                        current_reserved_pickups.add(goal)
                        total_score += score
                        break
                    else:
                        # Delivery is always valid (not exclusive)
                        current_assignment[shipper.id] = target
                        total_score += score
                        break

            if total_score > best_total_score:
                best_total_score = total_score
                best_assignment = current_assignment
                best_reserved_orders = current_reserved_orders
                best_reserved_pickups = current_reserved_pickups

        if len(available_shippers) > MAX_EXACT_ASSIGNMENT_SHIPPERS:
            all_candidates = [
                (score, -shipper.id, target)
                for shipper in available_shippers
                for score, target in score_matrix[shipper.id]
            ]
            all_candidates.sort(reverse=True)
            best_assignment = {}
            best_reserved_orders = set(forced_orders)
            best_reserved_pickups = set(forced_pickups)
            used_shippers = set()
            best_total_score = 0.0
            for score, neg_shipper_id, target in all_candidates:
                shipper_id = -neg_shipper_id
                if shipper_id in used_shippers:
                    continue
                kind, order_id, goal, _ = target
                if kind == "pickup":
                    if (
                        order_id in best_reserved_orders
                        or goal in best_reserved_pickups
                    ):
                        continue
                    best_reserved_orders.add(order_id)
                    best_reserved_pickups.add(goal)
                best_assignment[shipper_id] = target
                used_shippers.add(shipper_id)
                best_total_score += score

        if best_total_score > NEG_INF:
            for shipper_id, target in best_assignment.items():
                targets[shipper_id] = target

        # Fallback for completely unassigned shippers
        for shipper in available_shippers:
            if targets[shipper.id] is None and best_delivery.get(shipper.id) is not None:
                targets[shipper.id] = best_delivery[shipper.id]

        self._assignment_candidates = []

        # Update stickiness tracking
        self._commit_target_history(targets)

        return targets

    def _commit_target_history(
        self,
        targets: Dict[int, Optional[Target]],
    ) -> None:
        for sid, target in targets.items():
            if target is not None:
                self._target_by_shipper[sid] = (target[0], target[1])
            else:
                self._target_by_shipper.pop(sid, None)

    def _commit_selected_pickups(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Optional[Target]],
    ) -> None:
        if not ENABLE_PICKUP_COMMITMENT:
            self._pickup_commitments.clear()
            return
        for shipper in shippers:
            target = targets.get(shipper.id)
            if target is not None and target[0] == "pickup":
                pickup = target[2]
                previous = self._pickup_commitments.get(shipper.id)
                if previous is None or previous[0] != pickup:
                    self._pickup_commitments[shipper.id] = (
                        pickup,
                        self._distance(shipper.position, pickup),
                        0,
                    )
                continue
            self._drop_pickup_commitment(shipper.id, "delivery_override")

    def _assignment_heuristic_score(
        self,
        targets: Dict[int, Optional[Target]],
    ) -> float:
        return sum(target[3] for target in targets.values() if target is not None)

    def _assignment_signature(
        self,
        targets: Dict[int, Optional[Target]],
    ) -> Tuple[Tuple[int, str, int, Position], ...]:
        return tuple(
            (sid, target[0], target[1], target[2])
            for sid, target in sorted(targets.items())
            if target is not None
        )

    def _needs_kbest_assignment(
        self,
        available_shippers: List[Shipper],
        score_matrix: Dict[int, List[Tuple[float, Target]]],
    ) -> bool:
        if len(available_shippers) < 2:
            return False

        optionful_shippers = 0
        pickup_claims: Dict[Position, int] = {}
        delivery_pickup_tradeoff = False
        for shipper in available_shippers:
            options = score_matrix.get(shipper.id, [])
            kinds = {target[0] for _, target in options}
            if len(options) > 1:
                optionful_shippers += 1
            if "delivery" in kinds and "pickup" in kinds:
                delivery_pickup_tradeoff = True
            for _, target in options:
                if target[0] == "pickup":
                    pickup_claims[target[2]] = pickup_claims.get(target[2], 0) + 1

        contested_pickup = any(count > 1 for count in pickup_claims.values())
        return optionful_shippers >= 2 or contested_pickup or delivery_pickup_tradeoff

    def _trim_assignment_options(
        self,
        options: List[Tuple[float, Target]],
    ) -> List[Tuple[float, Target]]:
        delivery: List[Tuple[float, Target]] = []
        pickups: List[Tuple[float, Target]] = []
        for score, target in options:
            if target[0] == "delivery":
                if not delivery:
                    delivery.append((score, target))
            elif target[0] == "pickup" and len(pickups) < KBEST_PICKUPS_PER_SHIPPER:
                pickups.append((score, target))
        trimmed = delivery + pickups
        trimmed.sort(key=lambda item: item[0], reverse=True)
        return trimmed

    def _add_assignment_candidate(
        self,
        candidates: List[Tuple[float, Dict[int, Optional[Target]]]],
        seen: Set[Tuple[Tuple[int, str, int, Position], ...]],
        score: float,
        targets: Dict[int, Optional[Target]],
    ) -> None:
        signature = self._assignment_signature(targets)
        if signature in seen:
            return
        seen.add(signature)
        candidates.append((score, dict(targets)))

    def _build_assignment_candidates(
        self,
        baseline: Dict[int, Optional[Target]],
        forced_targets: Dict[int, Optional[Target]],
        available_shippers: List[Shipper],
        score_matrix: Dict[int, List[Tuple[float, Target]]],
    ) -> List[Tuple[float, Dict[int, Optional[Target]]]]:
        """
        Build a bounded set of short-term task assignments for path evaluation.

        The baseline assignment remains candidate zero. Alternatives only differ
        for flexible shippers and keep pickup exclusivity by cell/order.
        """
        baseline_score = self._assignment_heuristic_score(baseline)
        candidates: List[Tuple[float, Dict[int, Optional[Target]]]] = []
        seen: Set[Tuple[Tuple[int, str, int, Position], ...]] = set()
        self._add_assignment_candidate(candidates, seen, baseline_score, baseline)

        if not self._needs_kbest_assignment(available_shippers, score_matrix):
            return candidates

        flexible = sorted(
            available_shippers,
            key=lambda shipper: (len(score_matrix.get(shipper.id, [])), shipper.id),
        )
        option_matrix = {
            shipper.id: self._trim_assignment_options(score_matrix.get(shipper.id, []))
            for shipper in flexible
        }
        raw: List[Tuple[float, Dict[int, Optional[Target]]]] = []

        def search(
            index: int,
            current: Dict[int, Optional[Target]],
            reserved_orders: Set[int],
            reserved_pickups: Set[Position],
            score: float,
        ) -> None:
            if index >= len(flexible):
                raw.append((score, dict(current)))
                return

            shipper = flexible[index]
            shipper_id = shipper.id
            chose_valid_target = False
            for candidate_score, target in option_matrix[shipper_id]:
                kind, order_id, goal, _ = target
                if kind == "pickup" and (
                    order_id in reserved_orders or goal in reserved_pickups
                ):
                    continue

                chose_valid_target = True
                current[shipper_id] = target
                if kind == "pickup":
                    reserved_orders.add(order_id)
                    reserved_pickups.add(goal)
                search(
                    index + 1,
                    current,
                    reserved_orders,
                    reserved_pickups,
                    score + candidate_score,
                )
                if kind == "pickup":
                    reserved_orders.discard(order_id)
                    reserved_pickups.discard(goal)
                current[shipper_id] = None

            # Keep a legal idle branch when all attractive options collide. The
            # low heuristic score prevents it from displacing useful work.
            if not chose_valid_target:
                current[shipper_id] = None
                search(index + 1, current, reserved_orders, reserved_pickups, score)

        forced_orders, forced_pickups = self._pickup_reservations(forced_targets)
        search(
            0,
            dict(forced_targets),
            forced_orders,
            forced_pickups,
            self._assignment_heuristic_score(forced_targets),
        )
        raw.sort(key=lambda item: item[0], reverse=True)
        for score, targets in raw:
            self._add_assignment_candidate(candidates, seen, score, targets)
            if len(candidates) >= KBEST_ASSIGNMENTS:
                break
        return candidates


    def _priority(
        self,
        shipper: Shipper,
        target: Optional[Target],
        orders: Dict[int, Order],
        t: int,
    ) -> Tuple[int, float, int]:
        if target is None:
            return (0, 0.0, -shipper.id)
        kind, _, _, _ = target
        base = 3 if kind == "delivery" else 2 if kind == "pickup" else 1
        reward_loss = self._reward_loss_priority(shipper, target, orders, t)
        return (base, reward_loss, -shipper.id)

    def _reward_loss_priority(
        self,
        shipper: Shipper,
        target: Optional[Target],
        orders: Dict[int, Order],
        t: int,
    ) -> float:
        if target is None:
            return 0.0
        kind, order_id, goal, score = target
        order = orders.get(order_id)
        if order is None:
            return score

        distance = self._distance(shipper.position, goal)
        if distance >= INF:
            return NEG_INF

        if kind == "delivery":
            delivery_t = self._delivery_time_after_distance(t, distance)
            now_reward = delivery_reward(order, delivery_t, self.env.T)
            delayed_reward = delivery_reward(order, delivery_t + 2, self.env.T)
            slack = order.et - delivery_t
            return (
                score
                + 6.0 * order.p
                + 4.0 * max(0.0, now_reward - delayed_reward)
                + max(0.0, 20.0 - slack)
            )

        pickup_to_delivery = self._distance(goal, (order.ex, order.ey))
        delivery_t = t + max(distance - 1, 0) + pickup_to_delivery
        now_reward = delivery_reward(order, delivery_t, self.env.T)
        delayed_reward = delivery_reward(order, delivery_t + 2, self.env.T)
        return score + 3.0 * order.p + 2.0 * max(0.0, now_reward - delayed_reward)

    # ------------------------------------------------------------------
    # Windowed CBS-style planning
    # ------------------------------------------------------------------
    def _adaptive_window(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Optional[Target]],
    ) -> int:
        active_targets = sum(1 for target in targets.values() if target is not None)
        blocked_pressure = sum(min(2, count) for count in self._blocked_counts.values())
        pressure = blocked_pressure + self._recent_conflicts

        if len(shippers) >= 4:
            return MAX_WINDOW if pressure >= 3 else 12
        if pressure >= 4:
            return MAX_WINDOW
        if pressure >= 2:
            return WINDOW
        if len(shippers) <= 2 and active_targets <= 1:
            return MIN_WINDOW
        return 8

    def _violates_constraints(
        self,
        current: Position,
        nxt: Position,
        next_tau: int,
        vertex_constraints: Dict[int, Set[Position]],
        edge_constraints: Dict[int, Set[Tuple[Position, Position]]],
    ) -> bool:
        if nxt in vertex_constraints.get(next_tau, set()):
            return True
        if (current, nxt) in edge_constraints.get(next_tau, set()):
            return True
        return False

    def _constraint_risk(
        self,
        pos: Position,
        tau: int,
        vertex_constraints: Dict[int, Set[Position]],
    ) -> int:
        future_vertex_risk = sum(
            1
            for future_tau in range(tau + 1, tau + 4)
            if pos in vertex_constraints.get(future_tau, set())
        )
        free_degree = 0
        for move in MOVE_ONLY:
            if valid_next_pos(pos, move, self.grid) != pos:
                free_degree += 1
        bottleneck_risk = max(0, 3 - free_degree)
        return future_vertex_risk * 3 + bottleneck_risk

    def _state_rank(
        self,
        pos: Position,
        tau: int,
        goal: Position,
        vertex_constraints: Dict[int, Set[Position]],
    ) -> Tuple[int, int, int]:
        return (
            self._distance(pos, goal),
            self._constraint_risk(pos, tau, vertex_constraints),
            -tau,
        )

    def _low_level_search(
        self,
        start: Position,
        goal: Position,
        window: int,
        vertex_constraints: Dict[int, Set[Position]],
        edge_constraints: Dict[int, Set[Tuple[Position, Position]]],
    ) -> Path:
        if start == goal:
            return self._pad_path([start], window)

        queue: deque[Tuple[Position, int]] = deque([(start, 0)])
        parent: Dict[Tuple[Position, int], Tuple[Optional[Tuple[Position, int]], Position]] = {
            (start, 0): (None, start)
        }
        best_state = (start, 0)
        best_rank = self._state_rank(start, 0, goal, vertex_constraints)

        while queue:
            current, tau = queue.popleft()
            current_rank = self._state_rank(current, tau, goal, vertex_constraints)
            if current_rank < best_rank:
                best_rank = current_rank
                best_state = (current, tau)
            if current == goal:
                best_state = (current, tau)
                break
            if tau >= window:
                continue

            ordered_neighbors = sorted(
                self._neighbors(current),
                key=lambda item: self._state_rank(
                    item[1], tau + 1, goal, vertex_constraints,
                ),
            )
            for _, nxt in ordered_neighbors:
                next_tau = tau + 1
                state = (nxt, next_tau)
                if state in parent:
                    continue
                if self._violates_constraints(
                    current, nxt, next_tau,
                    vertex_constraints, edge_constraints,
                ):
                    continue
                parent[state] = ((current, tau), current)
                queue.append(state)

        path: Path = []
        state: Optional[Tuple[Position, int]] = best_state
        while state is not None:
            pos, _ = state
            path.append(pos)
            previous, _ = parent[state]
            state = previous
        path.reverse()
        return self._pad_path(path, window)

    def _add_path_constraints(
        self,
        path: Path,
        vertex_constraints: Dict[int, Set[Position]],
        edge_constraints: Dict[int, Set[Tuple[Position, Position]]],
    ) -> None:
        for tau in range(1, len(path)):
            vertex_constraints.setdefault(tau, set()).add(path[tau])
            # Prevent lower-priority agents from swapping with this path.
            edge_constraints.setdefault(tau, set()).add((path[tau], path[tau - 1]))

    def _detect_first_conflict(
        self,
        paths: Dict[int, Path],
    ) -> Optional[Tuple[int, int, int, str, Position]]:
        if not paths:
            return None
        horizon = min(len(path) for path in paths.values()) - 1
        for tau in range(1, horizon + 1):
            by_position: Dict[Position, int] = {}
            for sid, path in paths.items():
                pos = path[tau]
                if pos in by_position:
                    return (by_position[pos], sid, tau, "vertex", pos)
                by_position[pos] = sid

            ids = sorted(paths)
            for index, sid_a in enumerate(ids):
                for sid_b in ids[index + 1 :]:
                    a_swaps_with_b = (
                        paths[sid_a][tau - 1] == paths[sid_b][tau]
                        and paths[sid_a][tau] == paths[sid_b][tau - 1]
                    )
                    if a_swaps_with_b:
                        return (sid_a, sid_b, tau, "edge", paths[sid_a][tau])
        return None

    def _plan_paths_with_stats(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Optional[Target]],
        orders: Dict[int, Order],
        t: int,
        commit_history: bool,
    ) -> Tuple[Dict[int, Path], int]:
        window = self._adaptive_window(shippers, targets)
        ordered_shippers = sorted(
            shippers,
            key=lambda s: self._priority(s, targets.get(s.id), orders, t),
            reverse=True,
        )
        vertex_constraints: Dict[int, Set[Position]] = {}
        edge_constraints: Dict[int, Set[Tuple[Position, Position]]] = {}
        paths: Dict[int, Path] = {}

        for shipper in ordered_shippers:
            target = targets.get(shipper.id)
            goal = target[2] if target is not None else shipper.position
            path = self._low_level_search(
                shipper.position,
                goal,
                window,
                vertex_constraints,
                edge_constraints,
            )
            paths[shipper.id] = path
            self._add_path_constraints(path, vertex_constraints, edge_constraints)

        # A few CBS-like repair passes for conflicts that survived due to padding/fallback.
        conflict_count = 0
        for _ in range(MAX_REPLAN_ROUNDS):
            conflict = self._detect_first_conflict(paths)
            if conflict is None:
                break
            conflict_count += 1
            sid_a, sid_b, tau, kind, position = conflict
            shipper_a = next(shipper for shipper in shippers if shipper.id == sid_a)
            shipper_b = next(shipper for shipper in shippers if shipper.id == sid_b)
            target_a = targets.get(sid_a)
            target_b = targets.get(sid_b)
            goal_a = target_a[2] if target_a is not None else shipper_a.position
            goal_b = target_b[2] if target_b is not None else shipper_b.position

            # Simulate A yielding
            local_v_a: Dict[int, Set[Position]] = {}
            local_e_a: Dict[int, Set[Tuple[Position, Position]]] = {}
            for sid, path in paths.items():
                if sid == sid_a: continue
                self._add_path_constraints(path, local_v_a, local_e_a)
            if kind == "vertex":
                local_v_a.setdefault(tau, set()).add(position)
            else:
                local_e_a.setdefault(tau, set()).add((paths[sid_a][tau - 1], paths[sid_a][tau]))

            path_a_yields = self._low_level_search(shipper_a.position, goal_a, window, local_v_a, local_e_a)
            cost_a_yields = self._distance(path_a_yields[-1], goal_a) + self._distance(paths[sid_b][-1], goal_b)

            # Simulate B yielding
            local_v_b: Dict[int, Set[Position]] = {}
            local_e_b: Dict[int, Set[Tuple[Position, Position]]] = {}
            for sid, path in paths.items():
                if sid == sid_b: continue
                self._add_path_constraints(path, local_v_b, local_e_b)
            if kind == "vertex":
                local_v_b.setdefault(tau, set()).add(position)
            else:
                local_e_b.setdefault(tau, set()).add((paths[sid_b][tau - 1], paths[sid_b][tau]))

            path_b_yields = self._low_level_search(shipper_b.position, goal_b, window, local_v_b, local_e_b)
            cost_b_yields = self._distance(paths[sid_a][-1], goal_a) + self._distance(path_b_yields[-1], goal_b)

            if cost_a_yields < cost_b_yields:
                paths[sid_a] = path_a_yields
            elif cost_b_yields < cost_a_yields:
                paths[sid_b] = path_b_yields
            else:
                # Tie: fallback to fixed priority
                p_a = self._priority(shipper_a, target_a, orders, t)
                p_b = self._priority(shipper_b, target_b, orders, t)
                if p_a >= p_b:
                    paths[sid_b] = path_b_yields
                else:
                    paths[sid_a] = path_a_yields

        if commit_history:
            self._recent_conflicts = conflict_count
            self._last_chosen_conflicts = conflict_count
        return paths, conflict_count

    def _plan_paths(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Optional[Target]],
        orders: Dict[int, Order],
        t: int,
    ) -> Dict[int, Path]:
        paths, _ = self._plan_paths_with_stats(
            shippers,
            targets,
            orders,
            t,
            commit_history=True,
        )
        return paths

    def _target_action_time(
        self,
        shipper: Shipper,
        target: Target,
        path: Path,
        t: int,
    ) -> Tuple[int, bool, int]:
        """
        Return the expected timestep of pickup/delivery at target goal.

        A path index is a movement count. Env applies cargo after that move in
        the current timestep, hence index 1 is still action time t.
        """
        _, _, goal, _ = target
        if not path:
            distance = self._distance(shipper.position, goal)
            return self._delivery_time_after_distance(t, distance), False, distance

        for tau, pos in enumerate(path):
            if pos == goal:
                return t + max(tau - 1, 0), True, 0

        final_pos = path[-1]
        remaining = self._distance(final_pos, goal)
        action_t = self._delivery_time_after_distance(t + len(path) - 1, remaining)
        return action_t, False, remaining

    def _reward_delta_for_eta(
        self,
        order: Order,
        static_t: int,
        eta_t: int,
        horizon: int,
    ) -> float:
        return (
            delivery_reward(order, eta_t, horizon)
            - delivery_reward(order, static_t, horizon)
            - LATE_PENALTY * (
                max(0, eta_t - order.et) - max(0, static_t - order.et)
            )
        )

    def _path_aware_target_score(
        self,
        shipper: Shipper,
        target: Target,
        path: Path,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> float:
        kind, order_id, goal, heuristic_score = target
        order = orders.get(order_id)
        if order is None:
            return heuristic_score

        goal_action_t, reached, remaining = self._target_action_time(
            shipper,
            target,
            path,
            t,
        )
        score = heuristic_score

        if kind == "delivery":
            static_distance = self._distance(shipper.position, goal)
            static_t = self._delivery_time_after_distance(t, static_distance)
            same_goal = [
                carried
                for carried in self._carried_orders(shipper, orders)
                if (carried.ex, carried.ey) == goal
            ] or [order]
            score += sum(
                self._reward_delta_for_eta(carried, static_t, goal_action_t, horizon)
                for carried in same_goal
            )
        else:
            pickup_distance = self._distance(shipper.position, goal)
            trip = self._distance(goal, (order.ex, order.ey))
            if pickup_distance >= INF or trip >= INF:
                return NEG_INF
            static_delivery_t = t + max(pickup_distance - 1, 0) + trip
            eta_delivery_t = goal_action_t + trip
            score += self._reward_delta_for_eta(
                order,
                static_delivery_t,
                eta_delivery_t,
                horizon,
            )

        start_distance = self._distance(shipper.position, goal)
        if not reached and start_distance < INF:
            final_distance = remaining
            if final_distance >= start_distance:
                score -= KBEST_NO_PROGRESS_PENALTY * (
                    1.0 + min(4, final_distance - start_distance)
                )
        return score

    def _predicted_first_step_blocks(
        self,
        paths: Dict[int, Path],
        shippers: List[Shipper],
    ) -> int:
        old_positions = {shipper.id: shipper.position for shipper in shippers}
        occupied = set(old_positions.values())
        blocks = 0
        for shipper_id in sorted(old_positions):
            old = old_positions[shipper_id]
            path = paths.get(shipper_id, [old])
            desired = path[1] if len(path) >= 2 else old
            occupied.discard(old)
            if desired in occupied:
                desired = old
                blocks += 1
            occupied.add(desired)
        return blocks

    def _path_aware_assignment_score(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Optional[Target]],
        paths: Dict[int, Path],
        conflict_count: int,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> float:
        score = 0.0
        for shipper in shippers:
            target = targets.get(shipper.id)
            if target is None:
                continue
            score += self._path_aware_target_score(
                shipper,
                target,
                paths.get(shipper.id, [shipper.position]),
                orders,
                t,
                horizon,
            )

        score -= KBEST_CONFLICT_PENALTY * conflict_count
        score -= (
            KBEST_FIRST_STEP_CONFLICT_PENALTY
            * self._predicted_first_step_blocks(paths, shippers)
        )
        return score

    def _select_path_aware_assignment(
        self,
        shippers: List[Shipper],
        baseline_targets: Dict[int, Optional[Target]],
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> Tuple[Dict[int, Optional[Target]], Dict[int, Path]]:
        candidates = self._assignment_candidates or [
            (self._assignment_heuristic_score(baseline_targets), baseline_targets)
        ]
        start = time.perf_counter()
        baseline_signature = self._assignment_signature(candidates[0][1])
        best_score = NEG_INF
        best_targets = baseline_targets
        best_paths: Dict[int, Path] = {}
        best_conflicts = 0
        best_blocks = INF
        evaluated = 0

        for index, (_, targets) in enumerate(candidates):
            if index > 0 and time.perf_counter() - start >= KBEST_TICK_BUDGET_SEC:
                break
            paths, conflict_count = self._plan_paths_with_stats(
                shippers,
                targets,
                orders,
                t,
                commit_history=False,
            )
            path_score = self._path_aware_assignment_score(
                shippers,
                targets,
                paths,
                conflict_count,
                orders,
                t,
                horizon,
            )
            first_step_blocks = self._predicted_first_step_blocks(paths, shippers)
            evaluated += 1
            traffic_gain = (
                conflict_count < best_conflicts
                or first_step_blocks < best_blocks
            )
            safe_switch = (
                index == 0
                or (
                    traffic_gain
                    and path_score > best_score + KBEST_SWITCH_MARGIN
                )
            )
            if safe_switch:
                best_score = path_score
                best_targets = targets
                best_paths = paths
                best_conflicts = conflict_count
                best_blocks = first_step_blocks

        if not best_paths:
            best_paths, best_conflicts = self._plan_paths_with_stats(
                shippers,
                baseline_targets,
                orders,
                t,
                commit_history=False,
            )
            best_targets = baseline_targets
            best_score = self._path_aware_assignment_score(
                shippers,
                best_targets,
                best_paths,
                best_conflicts,
                orders,
                t,
                horizon,
            )
            evaluated = max(evaluated, 1)

        self._recent_conflicts = best_conflicts
        self._last_chosen_conflicts = best_conflicts
        self._last_kbest_evaluated = evaluated
        if evaluated > 1:
            self._kbest_eval_ticks += 1
            self._kbest_heuristic_score_total += candidates[0][0]
            self._kbest_path_score_total += best_score
            if self._assignment_signature(best_targets) != baseline_signature:
                self._kbest_changed_ticks += 1
        self._commit_target_history(best_targets)
        return best_targets, best_paths

    # ------------------------------------------------------------------
    # Action construction and main loop
    # ------------------------------------------------------------------
    def _action_for_path(
        self,
        shipper: Shipper,
        target: Optional[Target],
        path: Path,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> Action:
        # Deliver at current position first (free, no move needed).
        if self._has_delivery_at_position(shipper, orders, shipper.position):
            return ("S", 2)
        if len(path) < 2:
            return ("S", 0)

        move = self._move_between(shipper.position, path[1])
        next_position = valid_next_pos(shipper.position, move, self.grid)

        # Opportunistically deliver at next_position if any carried order lands there,
        # regardless of what the current target is (pickup or delivery of another order).
        if self._has_delivery_at_position(shipper, orders, next_position):
            return (move, 2)

        if target is None:
            return (move, 0)

        kind, order_id, goal, _ = target
        if next_position == goal and kind == "pickup":
            order = orders.get(order_id)
            if order is not None and not self._endgame_pickup_allowed(
                shipper,
                order,
                t,
                horizon,
            ):
                return (move, 0)
            return (move, 1)
        if next_position == goal and kind == "delivery":
            return (move, 2)
        return (move, 0)

    def _blocked_action(self, shipper: Shipper, orders: Dict[int, Order]) -> Action:
        if self._has_delivery_at_position(shipper, orders, shipper.position):
            return ("S", 2)
        return ("S", 0)

    def _resolve_first_step(
        self,
        actions: Dict[int, Action],
        shippers: List[Shipper],
        orders: Dict[int, Order],
    ) -> Dict[int, Action]:
        shipper_by_id = {shipper.id: shipper for shipper in shippers}
        old_positions = {shipper.id: shipper.position for shipper in shippers}
        occupied = set(old_positions.values())
        desired = {
            sid: valid_next_pos(shipper.position, actions.get(sid, ("S", 0))[0], self.grid)
            for sid, shipper in shipper_by_id.items()
        }
        actual: Dict[int, Position] = {}
        for sid in sorted(shipper_by_id):
            old = old_positions[sid]
            target = desired[sid]
            occupied.discard(old)
            if target in occupied:
                target = old
                self._blocked_counts[sid] = self._blocked_counts.get(sid, 0) + 1
            else:
                self._blocked_counts[sid] = 0
            occupied.add(target)
            actual[sid] = target

        resolved = dict(actions)
        for sid, target in desired.items():
            if actual[sid] != target:
                resolved[sid] = self._blocked_action(shipper_by_id[sid], orders)
        return resolved

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs["t"])
        horizon = int(obs["T"])
        targets = self._assign_targets_global(shippers, orders, t, horizon)

        paths = self._plan_paths(shippers, targets, orders, t)
        self._last_selected_targets = dict(targets)
        actions = {
            shipper.id: self._action_for_path(
                shipper,
                targets.get(shipper.id),
                paths.get(shipper.id, [shipper.position]),
                orders,
                t,
                horizon,
            )
            for shipper in shippers
        }
        return self._resolve_first_step(actions, shippers, orders)

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()
        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)
