"""Self-contained Greedy BFS solver for the online Graph Shopee MAPD task."""

from __future__ import annotations

import time
from collections import deque
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]
ScoredOrder = Tuple[Order, float]

INF = 10**9
NEG_INF = -10**18
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")

MIN_PICKUP_SCORE = 0.0
PICKUP_OVER_DELIVERY_MARGIN = 4.0
URGENT_SLACK_BUFFER = 4
ENDGAME_WINDOW_FRACTION = 0.15

DISTANCE_PENALTY = 0.08
LATE_PENALTY = 18.0
MAX_LATE_PENALTY_STEPS = 1000
PRIORITY_BONUS = 3.0
SAME_DESTINATION_BONUS = 4.0
PICKUP_WAIT_PENALTY = 0.03
DELIVERY_MARGIN_BONUS = 0.0

# Online-routing refinements kept local to this solver:
# - stickiness avoids thrashing when a slightly better order appears;
# - visible clustering exploits only public observation, not hidden hotspot params;
# - insertion detour discourages pickups that would delay already-carried orders.
STICKY_TARGET_BONUS = 5.0
PICKUP_CLUSTER_RADIUS = 2
PICKUP_CLUSTER_BONUS = 0.9
VISIBLE_DESTINATION_BONUS = 1.2
INSERTION_DETOUR_PENALTY = 0.12
IDLE_REPOSITION_MAX_FRACTION = 0.0
ROUTE_DELTA_WEIGHT = 0.35
DELIVERY_ROUTE_MIN_N = 20


class GreedyBFS(Solver):
    """
    Self-contained Greedy BFS dispatcher for the online MAPD environment.

    This file deliberately owns its pathfinding and scoring logic. Other methods
    may copy ideas, but should not import this solver's internals, so Phase 2 can
    safely submit/run one method without coupling algorithm files together.
    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}
        self._target_by_shipper: Dict[int, Tuple[str, int]] = {}

    # ------------------------------------------------------------------
    # Local BFS utilities: intentionally not shared across solver files.
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parents(
        self,
        start: Position,
        goal: Position,
    ) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        queue: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}

        while queue:
            current = queue.popleft()
            if current == goal:
                return parent

            for move, nxt in self._neighbors(current):
                if nxt in parent:
                    continue
                parent[nxt] = (current, move)
                queue.append(nxt)

        return None

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0

        key = (start, goal)
        if key in self._distance_cache:
            return self._distance_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._distance_cache[key] = INF
            return INF

        distance = 0
        current = goal
        while current != start:
            previous, _ = parent[current]
            if previous is None:
                self._distance_cache[key] = INF
                return INF
            current = previous
            distance += 1

        self._distance_cache[key] = distance
        return distance

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal:
            return "S"

        key = (start, goal)
        if key in self._next_move_cache:
            return self._next_move_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._next_move_cache[key] = "S"
            return "S"

        current = goal
        while True:
            previous, move = parent[current]
            if previous is None:
                self._next_move_cache[key] = "S"
                return "S"
            if previous == start:
                self._next_move_cache[key] = move
                return move
            current = previous

    # ------------------------------------------------------------------
    # Local scoring helpers: follow env.py/code reward and move semantics.
    # ------------------------------------------------------------------
    def _delivery_time_after_distance(self, t: int, distance: int) -> int:
        # In env.step, final move and delivery can happen in the same timestep.
        if distance >= INF:
            return INF
        return t + max(distance - 1, 0)

    def _carried_orders(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]

    def _same_destination_count(
        self,
        shipper: Shipper,
        target: Position,
        orders: Dict[int, Order],
    ) -> int:
        return sum(
            1
            for order in self._carried_orders(shipper, orders)
            if (order.ex, order.ey) == target
        )

    def _lateness_penalty(self, lateness: int) -> float:
        if lateness <= 0:
            return 0.0
        return LATE_PENALTY * min(lateness, MAX_LATE_PENALTY_STEPS)

    def _endgame_started(self, t: int, horizon: int) -> bool:
        return horizon - t <= max(12, int(horizon * ENDGAME_WINDOW_FRACTION))

    def _stickiness_bonus(self, shipper: Shipper, kind: str, order: Order) -> float:
        if self._target_by_shipper.get(shipper.id) == (kind, order.id):
            return STICKY_TARGET_BONUS
        return 0.0

    def _visible_pickup_cluster_bonus(self, order: Order, orders: Dict[int, Order]) -> float:
        """
        Reward currently visible pickup clusters without reading hidden hotspot params.
        This is a legal online proxy for demand concentration.
        """
        pickup = (order.sx, order.sy)
        bonus = 0.0
        for other in orders.values():
            if other.id == order.id or other.picked or other.delivered:
                continue
            distance = abs(other.sx - pickup[0]) + abs(other.sy - pickup[1])
            if distance == 0:
                bonus += 1.5 * PICKUP_CLUSTER_BONUS
            elif distance <= PICKUP_CLUSTER_RADIUS:
                bonus += PICKUP_CLUSTER_BONUS / distance
        return bonus

    def _visible_destination_cluster_bonus(self, order: Order, orders: Dict[int, Order]) -> float:
        """
        Destination batching is valuable because cargo_op=2 can deliver all carried
        orders sharing the current destination. Count only visible orders.
        """
        destination = (order.ex, order.ey)
        same_destination = 0
        near_destination = 0
        for other in orders.values():
            if other.id == order.id or other.picked or other.delivered:
                continue
            other_destination = (other.ex, other.ey)
            distance = (
                abs(other_destination[0] - destination[0])
                + abs(other_destination[1] - destination[1])
            )
            if distance == 0:
                same_destination += 1
            elif distance <= 1:
                near_destination += 1
        return (
            VISIBLE_DESTINATION_BONUS * same_destination
            + 0.5 * VISIBLE_DESTINATION_BONUS * near_destination
        )

    def _insertion_detour(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
    ) -> int:
        """
        Estimate the marginal extra route length of accepting order before/around
        the best currently carried delivery. Smaller is better; INF means impossible.
        """
        carried = self._carried_orders(shipper, orders)
        if not carried:
            return 0

        current = shipper.position
        pickup = (order.sx, order.sy)
        new_destination = (order.ex, order.ey)
        to_pickup = self._distance(current, pickup)
        pickup_to_new_destination = self._distance(pickup, new_destination)
        if to_pickup >= INF or pickup_to_new_destination >= INF:
            return INF

        best_detour = INF
        for carried_order in carried:
            carried_destination = (carried_order.ex, carried_order.ey)
            direct = self._distance(current, carried_destination)
            pickup_to_carried = self._distance(pickup, carried_destination)
            new_to_carried = self._distance(new_destination, carried_destination)
            carried_to_new = self._distance(carried_destination, new_destination)

            if direct >= INF:
                continue
            if new_to_carried < INF:
                best_detour = min(
                    best_detour,
                    max(0, to_pickup + pickup_to_new_destination + new_to_carried - direct),
                )
            if pickup_to_carried < INF and carried_to_new < INF:
                best_detour = min(
                    best_detour,
                    max(0, to_pickup + pickup_to_carried + carried_to_new - direct),
                )

        return best_detour

    def _route_delivery_time(
        self,
        start_t: int,
        cumulative_distance: int,
        after_pickup: bool,
    ) -> int:
        if after_pickup:
            # Pickup consumes this timestep, so even same-cell delivery waits
            # until the next env step.
            return start_t + max(cumulative_distance, 1)
        return start_t + max(cumulative_distance - 1, 0)

    def _route_score(
        self,
        route_orders: List[Order],
        start: Position,
        start_t: int,
        horizon: int,
        after_pickup: bool = False,
    ) -> float:
        """
        Score a complete delivery route over <= K_max carried orders.
        This brute-force route evaluator is cheap because official K_max <= 3.
        """
        if not route_orders:
            return 0.0

        groups: Dict[Position, List[Order]] = {}
        for order in route_orders:
            groups.setdefault((order.ex, order.ey), []).append(order)

        best_score = NEG_INF
        for destinations in permutations(groups):
            position = start
            cumulative_distance = 0
            route_score = 0.0
            feasible = True

            for destination in destinations:
                leg_distance = self._distance(position, destination)
                if leg_distance >= INF:
                    feasible = False
                    break

                cumulative_distance += leg_distance
                delivery_t = self._route_delivery_time(
                    start_t,
                    cumulative_distance,
                    after_pickup,
                )
                for order in groups[destination]:
                    lateness = max(0, delivery_t - order.et)
                    horizon_penalty = 40.0 if delivery_t >= horizon else 0.0
                    route_score += (
                        delivery_reward(order, delivery_t, horizon)
                        + PRIORITY_BONUS * order.p
                        - self._lateness_penalty(lateness)
                        - horizon_penalty
                    )
                position = destination

            if not feasible:
                continue

            route_score -= DISTANCE_PENALTY * cumulative_distance
            best_score = max(best_score, route_score)

        return best_score

    def _best_delivery_route(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> Optional[ScoredOrder]:
        carried = self._carried_orders(shipper, orders)
        if not carried:
            return None

        groups: Dict[Position, List[Order]] = {}
        for order in carried:
            groups.setdefault((order.ex, order.ey), []).append(order)

        best_first_order: Optional[Order] = None
        best_score = NEG_INF
        for destinations in permutations(groups):
            route_orders = [
                order
                for destination in destinations
                for order in groups[destination]
            ]
            score = self._route_score(route_orders, shipper.position, t, horizon)
            if score <= NEG_INF / 2:
                continue

            first_group = groups[destinations[0]]
            first_order = min(first_group, key=lambda o: (-o.p, o.et, o.id))
            score += self._stickiness_bonus(shipper, "delivery", first_order)
            if score > best_score or (
                score == best_score
                and best_first_order is not None
                and (first_order.et, -first_order.p, first_order.id)
                < (best_first_order.et, -best_first_order.p, best_first_order.id)
            ):
                best_first_order = first_order
                best_score = score

        if best_first_order is None:
            return None
        return best_first_order, best_score

    def _delivery_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> float:
        if order.id not in shipper.bag or order.delivered:
            return NEG_INF

        destination = (order.ex, order.ey)
        distance = self._distance(shipper.position, destination)
        if distance >= INF:
            return NEG_INF

        delivery_t = self._delivery_time_after_distance(t, distance)
        reward = delivery_reward(order, delivery_t, horizon)
        lateness = max(0, delivery_t - order.et)
        same_destination = self._same_destination_count(shipper, destination, orders)
        deadline_margin = max(0, order.et - delivery_t)

        return (
            reward
            + PRIORITY_BONUS * order.p
            + self._stickiness_bonus(shipper, "delivery", order)
            + SAME_DESTINATION_BONUS * max(0, same_destination - 1)
            + DELIVERY_MARGIN_BONUS * deadline_margin
            - DISTANCE_PENALTY * distance
            - self._lateness_penalty(lateness)
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
        distance_to_pickup = self._distance(shipper.position, pickup)
        trip_distance = self._distance(pickup, destination)
        if distance_to_pickup >= INF or trip_distance >= INF:
            return NEG_INF

        delivery_t = t + max(distance_to_pickup - 1, 0) + trip_distance
        reward = delivery_reward(order, delivery_t, horizon)
        lateness = max(0, delivery_t - order.et)
        deadline_margin = max(0, order.et - delivery_t)
        same_destination = self._same_destination_count(shipper, destination, orders)
        insertion_detour = self._insertion_detour(shipper, order, orders)
        if insertion_detour >= INF:
            return NEG_INF
        pickup_t = t + max(distance_to_pickup - 1, 0)
        carried = self._carried_orders(shipper, orders)
        current_route_score = self._route_score(
            carried,
            shipper.position,
            t,
            horizon,
        )
        new_route_score = self._route_score(
            carried + [order],
            pickup,
            pickup_t,
            horizon,
            after_pickup=True,
        )
        route_delta = new_route_score - current_route_score

        return (
            reward
            + (
                ROUTE_DELTA_WEIGHT
                if len(self.grid) >= DELIVERY_ROUTE_MIN_N
                else 0.0
            ) * route_delta
            + PRIORITY_BONUS * order.p
            + self._stickiness_bonus(shipper, "pickup", order)
            + SAME_DESTINATION_BONUS * same_destination
            + self._visible_pickup_cluster_bonus(order, orders)
            + self._visible_destination_cluster_bonus(order, orders)
            + DELIVERY_MARGIN_BONUS * deadline_margin
            - DISTANCE_PENALTY * (distance_to_pickup + trip_distance)
            - INSERTION_DETOUR_PENALTY * insertion_detour
            - self._lateness_penalty(lateness)
            - PICKUP_WAIT_PENALTY * max(0, t - order.appear_t)
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
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

    def _bag_is_full(self, shipper: Shipper) -> bool:
        return len(shipper.bag) >= shipper.K_max

    def _has_urgent_carried_order(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
    ) -> bool:
        for order in self._carried_orders(shipper, orders):
            distance = self._distance(shipper.position, (order.ex, order.ey))
            if distance >= INF:
                continue
            estimated_delivery_t = self._delivery_time_after_distance(t, distance)
            if estimated_delivery_t + URGENT_SLACK_BUFFER >= order.et:
                return True
        return False

    def _can_pick_more(self, shipper: Shipper, orders: Dict[int, Order]) -> bool:
        return any(shipper.can_carry(order, orders) for order in orders.values())

    def _nearest_valid_cell_to(self, rough_target: Position) -> Optional[Position]:
        if is_valid_cell(rough_target, self.grid):
            return rough_target

        best_cell: Optional[Position] = None
        best_distance = INF
        for row_index, row in enumerate(self.grid):
            for col_index, value in enumerate(row):
                if value != 0:
                    continue
                distance = abs(row_index - rough_target[0]) + abs(col_index - rough_target[1])
                if distance < best_distance:
                    best_distance = distance
                    best_cell = (row_index, col_index)
        return best_cell

    def _idle_reposition_action(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> Action:
        if (
            IDLE_REPOSITION_MAX_FRACTION <= 0
            or shipper.bag
            or t > int(horizon * IDLE_REPOSITION_MAX_FRACTION)
        ):
            self._target_by_shipper.pop(shipper.id, None)
            return ("S", 0)

        visible_orders = [
            order
            for order in orders.values()
            if not order.picked and not order.delivered
        ]
        if visible_orders:
            rough_target = (
                round(sum(order.sx for order in visible_orders) / len(visible_orders)),
                round(sum(order.sy for order in visible_orders) / len(visible_orders)),
            )
        else:
            rough_target = (len(self.grid) // 2, len(self.grid[0]) // 2)

        target = self._nearest_valid_cell_to(rough_target)
        if target is None or target == shipper.position:
            self._target_by_shipper.pop(shipper.id, None)
            return ("S", 0)

        move, next_position = self._move_towards(shipper, target)
        if next_position == shipper.position:
            self._target_by_shipper.pop(shipper.id, None)
            return ("S", 0)

        self._target_by_shipper[shipper.id] = ("idle", -1)
        return (move, 0)

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    def _select_delivery(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        t: int,
        horizon: int,
    ) -> Optional[ScoredOrder]:
        if len(self.grid) >= DELIVERY_ROUTE_MIN_N:
            route_candidate = self._best_delivery_route(shipper, orders, t, horizon)
            if route_candidate is not None:
                return route_candidate

        best_order: Optional[Order] = None
        best_score = NEG_INF

        for order in self._carried_orders(shipper, orders):
            score = self._delivery_score(shipper, order, orders, t, horizon)
            if score > best_score or (
                score == best_score
                and best_order is not None
                and (order.et, -order.p, order.id) < (best_order.et, -best_order.p, best_order.id)
            ):
                best_order = order
                best_score = score

        if best_order is None:
            return None
        return best_order, best_score

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: Set[int],
        reserved_pickup_cells: Set[Position],
        t: int,
        horizon: int,
    ) -> Optional[ScoredOrder]:
        best_order: Optional[Order] = None
        best_score = NEG_INF

        for order in orders.values():
            pickup = (order.sx, order.sy)
            if order.id in reserved_order_ids or pickup in reserved_pickup_cells:
                continue
            score = self._pickup_score(shipper, order, orders, t, horizon)
            if score <= NEG_INF / 2:
                continue
            if score > best_score or (
                score == best_score
                and best_order is not None
                and (-order.p, order.et, order.id) < (-best_order.p, best_order.et, best_order.id)
            ):
                best_order = order
                best_score = score

        if best_order is None:
            return None
        return best_order, best_score

    # ------------------------------------------------------------------
    # Action construction
    # ------------------------------------------------------------------
    def _move_towards(self, shipper: Shipper, goal: Position) -> Tuple[Move, Position]:
        move = self._next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        return move, next_position

    def _delivery_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.ex, order.ey)
        move, next_position = self._move_towards(shipper, goal)
        return (move, 2) if next_position == goal else (move, 0)

    def _pickup_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.sx, order.sy)
        move, next_position = self._move_towards(shipper, goal)
        return (move, 1) if next_position == goal else (move, 0)

    def _blocked_action(self, shipper: Shipper, orders: Dict[int, Order]) -> Action:
        if self._has_delivery_at_position(shipper, orders, shipper.position):
            return ("S", 2)
        return ("S", 0)

    # ------------------------------------------------------------------
    # One-step conflict handling
    # ------------------------------------------------------------------
    def _find_yield_move(
        self,
        blocker: Shipper,
        old_positions: Dict[int, Position],
        desired_positions: Dict[int, Position],
        extra_forbidden: Set[Position],
    ) -> Optional[Move]:
        forbidden = set(old_positions.values())
        forbidden.discard(blocker.position)
        forbidden.update(pos for sid, pos in desired_positions.items() if sid != blocker.id)
        forbidden.update(extra_forbidden)

        for move, next_position in self._neighbors(blocker.position):
            if next_position not in forbidden:
                return move
        return None

    def _yield_idle_blockers(
        self,
        actions: Dict[int, Action],
        shippers: List[Shipper],
    ) -> Dict[int, Action]:
        """
        Ask empty non-delivering shippers to step away from a carried delivery path.
        This is local to Greedy and avoids permanent blocking without shared CBS code.
        """
        resolved = dict(actions)
        shipper_by_id = {shipper.id: shipper for shipper in shippers}
        old_positions = {shipper.id: shipper.position for shipper in shippers}
        occupant_by_pos = {shipper.position: shipper.id for shipper in shippers}

        desired_positions = {
            sid: valid_next_pos(shipper.position, resolved.get(sid, ("S", 0))[0], self.grid)
            for sid, shipper in shipper_by_id.items()
        }

        for sid in sorted(shipper_by_id):
            shipper = shipper_by_id[sid]
            target = desired_positions[sid]
            if target == shipper.position:
                continue

            blocker_id = occupant_by_pos.get(target)
            if blocker_id is None or blocker_id == sid:
                continue

            blocker_action = resolved.get(blocker_id, ("S", 0))
            _, blocker_op = blocker_action
            blocker = shipper_by_id[blocker_id]
            if blocker.bag or blocker_op == 2:
                continue

            yield_move = self._find_yield_move(
                blocker,
                old_positions,
                desired_positions,
                extra_forbidden={shipper.position, target},
            )
            if yield_move is None:
                continue

            resolved[blocker_id] = (yield_move, 0)
            desired_positions[blocker_id] = valid_next_pos(blocker.position, yield_move, self.grid)

        return resolved

    def _resolve_conflicts(
        self,
        actions: Dict[int, Action],
        shippers: List[Shipper],
        orders: Dict[int, Order],
    ) -> Dict[int, Action]:
        actions = self._yield_idle_blockers(actions, shippers)
        shipper_by_id = {shipper.id: shipper for shipper in shippers}
        old_positions = {shipper.id: shipper.position for shipper in shippers}
        occupied = set(old_positions.values())
        desired_positions: Dict[int, Position] = {}
        actual_positions: Dict[int, Position] = {}

        for sid, shipper in shipper_by_id.items():
            move, _ = actions.get(sid, ("S", 0))
            desired_positions[sid] = valid_next_pos(shipper.position, move, self.grid)

        for sid in sorted(shipper_by_id):
            shipper = shipper_by_id[sid]
            old = old_positions[sid]
            target = desired_positions[sid]
            occupied.discard(old)
            if target in occupied:
                target = old
            occupied.add(target)
            actual_positions[sid] = target

        resolved = dict(actions)
        for sid, desired in desired_positions.items():
            if actual_positions[sid] != desired:
                resolved[sid] = self._blocked_action(shipper_by_id[sid], orders)
        return resolved

    # ------------------------------------------------------------------
    # Greedy policy
    # ------------------------------------------------------------------
    def _decide_shipper_action(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: Set[int],
        reserved_pickup_cells: Set[Position],
        t: int,
        horizon: int,
    ) -> Action:
        if self._has_delivery_at_position(shipper, orders, shipper.position):
            deliverable = [
                order
                for order in self._carried_orders(shipper, orders)
                if (order.ex, order.ey) == shipper.position
            ]
            if deliverable:
                order = min(deliverable, key=lambda o: (-o.p, o.et, o.id))
                self._target_by_shipper[shipper.id] = ("delivery", order.id)
            return ("S", 2)

        delivery_candidate = self._select_delivery(shipper, orders, t, horizon)
        if delivery_candidate is not None:
            delivery_order, delivery_score = delivery_candidate
        else:
            delivery_order, delivery_score = None, NEG_INF

        if delivery_order is not None and (
            self._bag_is_full(shipper)
            or self._has_urgent_carried_order(shipper, orders, t)
        ):
            self._target_by_shipper[shipper.id] = ("delivery", delivery_order.id)
            return self._delivery_action(shipper, delivery_order)

        pickup_candidate = None
        if self._can_pick_more(shipper, orders):
            pickup_candidate = self._select_pickup(
                shipper,
                orders,
                reserved_order_ids,
                reserved_pickup_cells,
                t,
                horizon,
            )

        if pickup_candidate is not None:
            pickup_order, pickup_score = pickup_candidate
            pickup_is_worth_it = pickup_score >= MIN_PICKUP_SCORE and (
                delivery_order is None
                or pickup_score >= delivery_score + PICKUP_OVER_DELIVERY_MARGIN
            )
            if pickup_is_worth_it:
                reserved_order_ids.add(pickup_order.id)
                reserved_pickup_cells.add((pickup_order.sx, pickup_order.sy))
                self._target_by_shipper[shipper.id] = ("pickup", pickup_order.id)
                return self._pickup_action(shipper, pickup_order)

        if delivery_order is not None:
            self._target_by_shipper[shipper.id] = ("delivery", delivery_order.id)
            return self._delivery_action(shipper, delivery_order)

        self._target_by_shipper.pop(shipper.id, None)
        return ("S", 0)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        t = int(obs["t"])
        horizon = int(obs["T"])

        if int(obs["N"]) > 12:
            actions: Dict[int, Action] = {}
            reserved_order_ids: Set[int] = set()
            reserved_pickup_cells: Set[Position] = set()

            for shipper in sorted(shippers, key=lambda s: s.id):
                actions[shipper.id] = self._decide_shipper_action(
                    shipper,
                    orders,
                    reserved_order_ids,
                    reserved_pickup_cells,
                    t,
                    horizon,
                )

            return self._resolve_conflicts(actions, shippers, orders)

        actions: Dict[int, Action] = {}
        flexible_shippers: List[Shipper] = []
        delivery_by_shipper: Dict[int, ScoredOrder] = {}
        shipper_by_id = {shipper.id: shipper for shipper in shippers}

        for shipper in sorted(shippers, key=lambda s: s.id):
            if self._has_delivery_at_position(shipper, orders, shipper.position):
                deliverable = [
                    order
                    for order in self._carried_orders(shipper, orders)
                    if (order.ex, order.ey) == shipper.position
                ]
                if deliverable:
                    order = min(deliverable, key=lambda o: (-o.p, o.et, o.id))
                    self._target_by_shipper[shipper.id] = ("delivery", order.id)
                actions[shipper.id] = ("S", 2)
                continue

            delivery_candidate = self._select_delivery(shipper, orders, t, horizon)
            if delivery_candidate is not None:
                delivery_by_shipper[shipper.id] = delivery_candidate

            must_deliver = delivery_candidate is not None and (
                self._bag_is_full(shipper)
                or self._has_urgent_carried_order(shipper, orders, t)
                or (shipper.bag and self._endgame_started(t, horizon))
            )
            if must_deliver and delivery_candidate is not None:
                delivery_order, _ = delivery_candidate
                self._target_by_shipper[shipper.id] = ("delivery", delivery_order.id)
                actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                continue

            flexible_shippers.append(shipper)

        pickup_candidates: List[Tuple[float, int, int, int, int, Order]] = []
        for shipper in flexible_shippers:
            if not self._can_pick_more(shipper, orders):
                continue

            delivery_candidate = delivery_by_shipper.get(shipper.id)
            if delivery_candidate is not None:
                _, delivery_score = delivery_candidate
            else:
                delivery_score = NEG_INF

            for order in orders.values():
                if not shipper.can_carry(order, orders):
                    continue
                pickup_score = self._pickup_score(shipper, order, orders, t, horizon)
                if pickup_score < MIN_PICKUP_SCORE:
                    continue
                if (
                    delivery_candidate is not None
                    and pickup_score < delivery_score + PICKUP_OVER_DELIVERY_MARGIN
                ):
                    continue

                distance_to_pickup = self._distance(shipper.position, (order.sx, order.sy))
                if distance_to_pickup >= INF:
                    continue
                pickup_candidates.append(
                    (
                        pickup_score,
                        order.p,
                        -distance_to_pickup,
                        -shipper.id,
                        -order.id,
                        order,
                    )
                )

        assigned_shippers: Set[int] = set()
        reserved_order_ids: Set[int] = set()
        reserved_pickup_cells: Set[Position] = set()
        for _, _, _, negative_shipper_id, _, order in sorted(pickup_candidates, reverse=True):
            shipper_id = -negative_shipper_id
            pickup = (order.sx, order.sy)
            if (
                shipper_id in assigned_shippers
                or order.id in reserved_order_ids
                or pickup in reserved_pickup_cells
            ):
                continue

            shipper = shipper_by_id[shipper_id]
            actions[shipper_id] = self._pickup_action(shipper, order)
            self._target_by_shipper[shipper_id] = ("pickup", order.id)
            assigned_shippers.add(shipper_id)
            reserved_order_ids.add(order.id)
            reserved_pickup_cells.add(pickup)

        for shipper in flexible_shippers:
            if shipper.id in actions:
                continue

            delivery_candidate = delivery_by_shipper.get(shipper.id)
            if delivery_candidate is not None:
                delivery_order, _ = delivery_candidate
                self._target_by_shipper[shipper.id] = ("delivery", delivery_order.id)
                actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                continue

            actions[shipper.id] = self._idle_reposition_action(shipper, orders, t, horizon)

        return self._resolve_conflicts(actions, shippers, orders)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
