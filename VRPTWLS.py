"""
Hybrid Constructive Heuristic for VRPTW - With Post-Optimization & Enhanced Output
===================================================================================
Base: Proven savings approach with instance-adaptive urgency
Enhancement: Post-optimization phase + Comprehensive output (logs & visualizations)
Author: Iurii Rusalev
Date: 20251017

Output Features:
- Simple route logs (log_simple_route/)
- Detailed reports (log_detailed_route/)
- Matplotlib visualizations (route_matp/)
"""

import numpy as np
import math
import os
import time
import platform
import psutil
import matplotlib.pyplot as plt
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float

    @property
    def time_window_width(self) -> float:
        return self.due_date - self.ready_time


class VRPTWInstance:
    def __init__(self, filename: str):
        self.filename = filename
        self.customers = []
        self.depot = None
        self.vehicle_capacity = 0
        self.num_vehicles = 0
        self.distances = None
        self.instance_type = ""
        self.parse_file(filename)
        self.compute_distance_matrix()
        self.classify_instance()

    def parse_file(self, filename: str):
        with open(filename, 'r') as f:
            lines = f.readlines()
        parts = lines[4].split()
        self.num_vehicles = int(parts[0])
        self.vehicle_capacity = float(parts[1])
        for i in range(9, len(lines)):
            parts = lines[i].split()
            if len(parts) < 7:
                continue
            customer = Customer(
                id=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2]),
                demand=float(parts[3]),
                ready_time=float(parts[4]),
                due_date=float(parts[5]),
                service_time=float(parts[6])
            )
            if customer.id == 0:
                self.depot = customer
            else:
                self.customers.append(customer)

    def compute_distance_matrix(self):
        n = len(self.customers) + 1
        self.distances = np.zeros((n, n))
        all_nodes = [self.depot] + self.customers
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = all_nodes[i].x - all_nodes[j].x
                    dy = all_nodes[i].y - all_nodes[j].y
                    self.distances[i][j] = math.sqrt(dx * dx + dy * dy)

    def classify_instance(self):
        basename = os.path.basename(self.filename).upper()
        if basename.startswith('C'):
            self.instance_type = 'C'
        elif basename.startswith('RC'):
            self.instance_type = 'RC'
        elif basename.startswith('R'):
            self.instance_type = 'R'
        else:
            self.instance_type = 'R'

    def get_customer(self, cust_id: int) -> Customer:
        if cust_id == 0:
            return self.depot
        return self.customers[cust_id - 1]


class Route:
    def __init__(self, instance: VRPTWInstance):
        self.instance = instance
        self.customers = []
        self.load = 0.0
        self.distance = 0.0
        self.times = []

    def copy(self):
        new_route = Route(self.instance)
        new_route.customers = self.customers.copy()
        new_route.load = self.load
        new_route.distance = self.distance
        new_route.times = self.times.copy()
        return new_route

    def calculate_metrics(self):
        if not self.customers:
            self.distance = 0.0
            self.times = []
            return

        self.distance = 0.0
        self.times = []
        current_time = self.instance.depot.ready_time

        for i, cust_id in enumerate(self.customers):
            prev_id = self.customers[i - 1] if i > 0 else 0
            self.distance += self.instance.distances[prev_id][cust_id]
            arrival = current_time + self.instance.distances[prev_id][cust_id]
            customer = self.instance.get_customer(cust_id)
            start_service = max(arrival, customer.ready_time)
            self.times.append(start_service)
            current_time = start_service + customer.service_time

        last_id = self.customers[-1]
        self.distance += self.instance.distances[last_id][0]


class Solution:
    def __init__(self, instance: VRPTWInstance):
        self.instance = instance
        self.routes = []

    def copy(self):
        new_sol = Solution(self.instance)
        new_sol.routes = [r.copy() for r in self.routes]
        return new_sol

    def calculate_total_distance(self) -> float:
        return sum(r.distance for r in self.routes)

    def num_vehicles(self) -> int:
        return len(self.routes)

    def is_complete(self) -> bool:
        routed = set()
        for route in self.routes:
            routed.update(route.customers)
        all_customers = set(range(1, len(self.instance.customers) + 1))
        return routed == all_customers


class HybridSolver:
    """Hybrid: Proven savings base + post-optimization + comprehensive output"""

    def __init__(self, instance: VRPTWInstance, enable_logging: bool = False):
        self.instance = instance
        self.enable_logging = enable_logging
        self.diagnostics = {
            'route_utilizations': [],
            'customers_per_route': [],
            'premature_closures': 0,
            'tight_window_rejections': 0,
            'capacity_rejections': 0,
            'best_strategy': None
        }

        # EXACT parameters from baseline
        if instance.instance_type == 'C':
            self.alpha = 1.0
            self.beta = 0.2
            self.gamma = 0.4
            self.delta = 0.2
            self.lambda_param = 1.5
        elif instance.instance_type == 'R':
            self.alpha = 1.0
            self.beta = 0.05
            self.gamma = 0.2
            self.delta = 0.0
            self.lambda_param = 2.0
        else:  # RC
            self.alpha = 1.0
            self.beta = 0.1
            self.gamma = 0.3
            self.delta = 0.1
            self.lambda_param = 1.8

    def solve(self) -> Solution:
        """Select by VEHICLES FIRST, then distance"""
        best_solution = None
        best_vehicles = float('inf')
        best_distance = float('inf')
        strategy_results = []

        for strategy in range(8):
            solution = self._construct_with_strategy(strategy)

            if solution and solution.is_complete():
                vehicles = solution.num_vehicles()
                distance = solution.calculate_total_distance()

                strategy_results.append((strategy, vehicles, distance))

                if (vehicles < best_vehicles) or \
                   (vehicles == best_vehicles and distance < best_distance):
                    best_vehicles = vehicles
                    best_distance = distance
                    best_solution = solution.copy()
                    self.diagnostics['best_strategy'] = strategy

        if self.enable_logging:
            self._log_strategy_comparison(strategy_results)

        # Store pre-optimization solution
        self.pre_opt_solution = best_solution.copy() if best_solution else None

        # POST-OPTIMIZATION
        if best_solution:
            if self.enable_logging:
                print(f"\n  Pre-optimization: {best_solution.num_vehicles()}v, {best_solution.calculate_total_distance():.0f}d")

            # CRITICAL: Verify completeness before optimization
            pre_customer_count = sum(len(r.customers) for r in best_solution.routes)

            best_solution = self._post_optimize(best_solution)

            # CRITICAL: Verify completeness after optimization
            post_customer_count = sum(len(r.customers) for r in best_solution.routes)

            if post_customer_count != pre_customer_count:
                print(f"\n⚠ WARNING: Customer loss detected during optimization!")
                print(f"  Before: {pre_customer_count} customers")
                print(f"  After: {post_customer_count} customers")
                print(f"  Using pre-optimization solution instead.")
                best_solution = self.pre_opt_solution.copy()
            elif not best_solution.is_complete():
                print(f"\n⚠ WARNING: Incomplete solution after optimization!")
                print(f"  Using pre-optimization solution instead.")
                best_solution = self.pre_opt_solution.copy()

            if self.enable_logging:
                print(f"  Post-optimization: {best_solution.num_vehicles()}v, {best_solution.calculate_total_distance():.0f}d")

        return best_solution

    def _log_strategy_comparison(self, results: List):
        """Log which strategies performed best"""
        print(f"\n  Strategy Performance:")
        for strategy, vehicles, distance in results:
            marker = "✓" if strategy == self.diagnostics['best_strategy'] else " "
            print(f"  {marker} Strategy {strategy}: {vehicles}v, {distance:.0f}d")

    def _construct_with_strategy(self, strategy: int) -> Solution:
        """Sequential route construction"""
        solution = Solution(self.instance)
        assigned = np.zeros(len(self.instance.customers), dtype=bool)

        while not np.all(assigned):
            seed = self._get_seed(assigned, strategy)
            if seed == -1:
                break

            route = Route(self.instance)
            route.customers = [seed]
            route.load = self.instance.get_customer(seed).demand
            route.calculate_metrics()
            solution.routes.append(route)
            assigned[seed - 1] = True

            while True:
                best_customer, best_position, best_saving = self._get_best_insertion(
                    solution.routes[-1], assigned
                )

                if best_customer == -1:
                    route = solution.routes[-1]
                    utilization = route.load / self.instance.vehicle_capacity
                    customers_count = len(route.customers)

                    self.diagnostics['route_utilizations'].append(utilization)
                    self.diagnostics['customers_per_route'].append(customers_count)

                    if utilization < 0.6 and customers_count < 8:
                        self.diagnostics['premature_closures'] += 1
                    break

                route = solution.routes[-1]
                route.customers.insert(best_position, best_customer)
                route.load += self.instance.get_customer(best_customer).demand
                route.calculate_metrics()
                assigned[best_customer - 1] = True

        return solution

    def _get_best_insertion(self, route: Route, assigned: np.ndarray) -> Tuple[int, int, float]:
        """Find customer with maximum savings"""
        best_customer = -1
        best_position = -1
        max_saving = -float('inf')

        for cust_id in range(1, len(self.instance.customers) + 1):
            if assigned[cust_id - 1]:
                continue

            customer = self.instance.get_customer(cust_id)

            if route.load + customer.demand > self.instance.vehicle_capacity:
                continue

            min_cost = float('inf')
            best_pos = -1

            for pos in range(len(route.customers) + 1):
                cost = self._calculate_insertion_cost(route, cust_id, pos)

                if cost < min_cost:
                    min_cost = cost
                    best_pos = pos

            if best_pos == -1:
                continue

            depot_distance = self.lambda_param * self.instance.distances[0][cust_id]
            saving = depot_distance - min_cost

            route_utilization = route.load / self.instance.vehicle_capacity
            if route_utilization < 0.5 and saving > -20:
                saving = max(saving, 0)

            if saving > max_saving:
                max_saving = saving
                best_customer = cust_id
                best_position = best_pos

        return best_customer, best_position, max_saving

    def _calculate_insertion_cost(self, route: Route, cust_id: int, position: int) -> float:
        """4-component insertion cost"""
        customer_u = self.instance.get_customer(cust_id)

        i = route.customers[position - 1] if position > 0 else 0
        j = route.customers[position] if position < len(route.customers) else 0

        d_iu = self.instance.distances[i][cust_id]
        d_uj = self.instance.distances[cust_id][j]
        d_ij = self.instance.distances[i][j]
        C_dist = self.alpha * (d_iu + d_uj - d_ij)

        if position == 0:
            current_time = self.instance.depot.ready_time
        else:
            prev_idx = position - 1
            current_time = route.times[prev_idx] + \
                          self.instance.get_customer(route.customers[prev_idx]).service_time

        arrival_u = current_time + d_iu

        if arrival_u > customer_u.due_date:
            return float('inf')

        begin_u = max(customer_u.ready_time, arrival_u)
        C_wait = self.beta * (begin_u - arrival_u)

        C_time_delay = 0.0
        current_time = begin_u + customer_u.service_time

        for k_idx in range(position, len(route.customers)):
            k_id = route.customers[k_idx]
            customer_k = self.instance.get_customer(k_id)
            prev_id = cust_id if k_idx == position else route.customers[k_idx - 1]

            travel_time = self.instance.distances[prev_id][k_id]
            arrival_k = current_time + travel_time
            start_k = max(customer_k.ready_time, arrival_k)

            if start_k > customer_k.due_date:
                return float('inf')

            old_start = route.times[k_idx]
            delay = max(0, start_k - old_start)
            C_time_delay += delay

            current_time = start_k + customer_k.service_time

        if route.customers:
            last_id = route.customers[-1] if position >= len(route.customers) else route.customers[-1]
            return_time = current_time + self.instance.distances[last_id][0]
            if return_time > self.instance.depot.due_date:
                return float('inf')

        C_time_delay_cost = self.gamma * C_time_delay

        C_compactness = 0.0
        if route.customers and self.delta > 0:
            sum_x = sum(self.instance.get_customer(c_id).x for c_id in route.customers)
            sum_y = sum(self.instance.get_customer(c_id).y for c_id in route.customers)
            center_x = sum_x / len(route.customers)
            center_y = sum_y / len(route.customers)
            dx = customer_u.x - center_x
            dy = customer_u.y - center_y
            dist_to_center = math.sqrt(dx * dx + dy * dy)
            C_compactness = self.delta * dist_to_center

        return C_dist + C_wait + C_time_delay_cost + C_compactness

    def _get_seed(self, assigned: np.ndarray, strategy: int) -> int:
        """8 diverse seed strategies"""
        unrouted = [i for i in range(1, len(self.instance.customers) + 1)
                    if not assigned[i - 1]]

        if not unrouted:
            return -1

        if strategy == 0:
            return max(unrouted, key=lambda c: self.instance.distances[0][c])
        elif strategy == 1:
            return min(unrouted, key=lambda c: self.instance.distances[0][c])
        elif strategy == 2:
            return max(unrouted, key=lambda c: self.instance.get_customer(c).due_date)
        elif strategy == 3:
            return min(unrouted, key=lambda c: self.instance.get_customer(c).due_date)
        elif strategy == 4:
            return min(unrouted, key=lambda c: self.instance.get_customer(c).ready_time)
        elif strategy == 5:
            return max(unrouted, key=lambda c: self.instance.get_customer(c).ready_time)
        elif strategy == 6:
            return min(unrouted, key=lambda c: self.instance.get_customer(c).time_window_width)
        elif strategy == 7:
            return max(unrouted, key=lambda c: self.instance.get_customer(c).time_window_width)

        return unrouted[0]

    def get_enhanced_diagnostics(self, pre_opt_solution: Solution, post_opt_solution: Solution,
                                 computation_time: float) -> dict:
        """Generate enhanced diagnostics with more detailed analysis"""

        # Basic metrics
        pre_vehicles = pre_opt_solution.num_vehicles()
        pre_distance = pre_opt_solution.calculate_total_distance()
        post_vehicles = post_opt_solution.num_vehicles()
        post_distance = post_opt_solution.calculate_total_distance()

        # Calculate improvements
        vehicle_reduction = pre_vehicles - post_vehicles
        distance_reduction = pre_distance - post_distance
        vehicle_reduction_pct = (vehicle_reduction / pre_vehicles * 100) if pre_vehicles > 0 else 0
        distance_reduction_pct = (distance_reduction / pre_distance * 100) if pre_distance > 0 else 0

        # Route length distribution
        route_lengths = [len(r.customers) for r in post_opt_solution.routes]
        route_distances = [r.distance for r in post_opt_solution.routes]

        # Utilization analysis
        utilizations = [r.load / self.instance.vehicle_capacity for r in post_opt_solution.routes]

        # Time window analysis
        total_slack = 0
        tight_windows = 0
        total_tw_count = 0

        for route in post_opt_solution.routes:
            for idx, cust_id in enumerate(route.customers):
                if idx < len(route.times):
                    customer = self.instance.get_customer(cust_id)
                    arrival = route.times[idx]
                    slack = customer.due_date - arrival
                    total_slack += slack
                    total_tw_count += 1

                    if slack < 10:
                        tight_windows += 1

        avg_slack = total_slack / total_tw_count if total_tw_count > 0 else 0
        tight_window_pct = (tight_windows / total_tw_count * 100) if total_tw_count > 0 else 0

        # Strategy effectiveness
        strategy_name = {
            0: "Farthest from Depot",
            1: "Nearest to Depot",
            2: "Latest Due Date",
            3: "Earliest Due Date",
            4: "Earliest Ready Time",
            5: "Latest Ready Time",
            6: "Tightest Time Window",
            7: "Widest Time Window"
        }.get(self.diagnostics['best_strategy'], "Unknown")

        # Compile enhanced diagnostics
        enhanced = {
            'instance_name': os.path.basename(self.instance.filename).replace('.txt', ''),
            'instance_type': self.instance.instance_type,
            'computation_time': computation_time,

            # Pre-optimization
            'pre_vehicles': pre_vehicles,
            'pre_distance': pre_distance,
            'pre_avg_route_length': np.mean([len(r.customers) for r in pre_opt_solution.routes]) if pre_opt_solution.routes else 0,

            # Post-optimization
            'post_vehicles': post_vehicles,
            'post_distance': post_distance,
            'post_avg_route_length': np.mean(route_lengths) if route_lengths else 0,

            # Improvements
            'vehicle_reduction': vehicle_reduction,
            'distance_reduction': distance_reduction,
            'vehicle_reduction_pct': vehicle_reduction_pct,
            'distance_reduction_pct': distance_reduction_pct,

            # Route analysis
            'avg_customers_per_route': np.mean(route_lengths) if route_lengths else 0,
            'min_customers_per_route': min(route_lengths) if route_lengths else 0,
            'max_customers_per_route': max(route_lengths) if route_lengths else 0,
            'std_customers_per_route': np.std(route_lengths) if route_lengths else 0,

            # Distance analysis
            'avg_route_distance': np.mean(route_distances) if route_distances else 0,
            'min_route_distance': min(route_distances) if route_distances else 0,
            'max_route_distance': max(route_distances) if route_distances else 0,
            'std_route_distance': np.std(route_distances) if route_distances else 0,

            # Utilization analysis
            'avg_utilization': np.mean(utilizations) * 100 if utilizations else 0,
            'min_utilization': min(utilizations) * 100 if utilizations else 0,
            'max_utilization': max(utilizations) * 100 if utilizations else 0,
            'utilization_variance': np.var(utilizations) * 10000 if utilizations else 0,

            # Time window analysis
            'avg_time_slack': avg_slack,
            'tight_windows_count': tight_windows,
            'tight_windows_pct': tight_window_pct,

            # Construction phase issues
            'premature_closures': self.diagnostics.get('premature_closures', 0),
            'capacity_rejections': self.diagnostics.get('capacity_rejections', 0),
            'time_window_rejections': self.diagnostics.get('tight_window_rejections', 0),

            # Strategy
            'best_strategy_id': self.diagnostics.get('best_strategy', -1),
            'best_strategy_name': strategy_name,
        }

        return enhanced

    def _post_optimize(self, solution: Solution) -> Solution:
        """Post-optimization: route elimination, relocate, 2-opt"""
        improved = True
        iteration = 0
        max_iterations = 3

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            old_vehicles = solution.num_vehicles()
            solution = self._eliminate_routes(solution)
            if solution.num_vehicles() < old_vehicles:
                improved = True
                if self.enable_logging:
                    print(f"    Iter {iteration}: Eliminated {old_vehicles - solution.num_vehicles()} routes")

            old_distance = solution.calculate_total_distance()
            solution = self._relocate_customers(solution)
            new_distance = solution.calculate_total_distance()
            if new_distance < old_distance - 1:
                improved = True
                if self.enable_logging:
                    print(f"    Iter {iteration}: Relocate improved distance by {old_distance - new_distance:.0f}")

            for route in solution.routes:
                if len(route.customers) > 3:
                    self._two_opt_route(route)

        return solution

    def _eliminate_routes(self, solution: Solution) -> Solution:
        """Try to eliminate routes by redistributing customers"""
        if len(solution.routes) <= 1:
            return solution

        sorted_indices = sorted(range(len(solution.routes)),
                               key=lambda i: len(solution.routes[i].customers))

        for idx in sorted_indices[:min(5, len(sorted_indices))]:
            route = solution.routes[idx]
            if not route.customers:
                continue

            customers_to_place = route.customers.copy()
            placement_found = [False] * len(customers_to_place)

            for cust_idx, cust_id in enumerate(customers_to_place):
                customer = self.instance.get_customer(cust_id)

                best_route_idx = -1
                best_position = -1
                min_cost_increase = float('inf')

                for other_idx, other_route in enumerate(solution.routes):
                    if other_idx == idx:
                        continue

                    if other_route.load + customer.demand > self.instance.vehicle_capacity:
                        continue

                    for pos in range(len(other_route.customers) + 1):
                        old_distance = other_route.distance

                        test_route = other_route.copy()
                        test_route.customers.insert(pos, cust_id)
                        test_route.load += customer.demand
                        test_route.calculate_metrics()

                        if self._is_route_feasible(test_route):
                            cost_increase = test_route.distance - old_distance
                            if cost_increase < min_cost_increase:
                                min_cost_increase = cost_increase
                                best_route_idx = other_idx
                                best_position = pos

                if best_route_idx != -1:
                    placement_found[cust_idx] = True

            if all(placement_found):
                new_solution = Solution(self.instance)
                for other_idx, other_route in enumerate(solution.routes):
                    if other_idx == idx:
                        continue
                    new_solution.routes.append(other_route.copy())

                for cust_id in customers_to_place:
                    customer = self.instance.get_customer(cust_id)
                    best_route_idx = -1
                    best_position = -1
                    min_cost = float('inf')

                    for route_idx, route in enumerate(new_solution.routes):
                        if route.load + customer.demand > self.instance.vehicle_capacity:
                            continue

                        for pos in range(len(route.customers) + 1):
                            test_route = route.copy()
                            test_route.customers.insert(pos, cust_id)
                            test_route.load += customer.demand
                            test_route.calculate_metrics()

                            if self._is_route_feasible(test_route) and test_route.distance < min_cost:
                                min_cost = test_route.distance
                                best_route_idx = route_idx
                                best_position = pos

                    if best_route_idx != -1:
                        new_solution.routes[best_route_idx].customers.insert(best_position, cust_id)
                        new_solution.routes[best_route_idx].load += customer.demand
                        new_solution.routes[best_route_idx].calculate_metrics()

                return new_solution

        return solution

    def _relocate_customers(self, solution: Solution) -> Solution:
        """Relocate customers between routes"""
        improved = True

        while improved:
            improved = False

            for route_idx, route in enumerate(solution.routes):
                if not route.customers:
                    continue

                for cust_pos, cust_id in enumerate(route.customers):
                    customer = self.instance.get_customer(cust_id)

                    for other_idx, other_route in enumerate(solution.routes):
                        if other_idx == route_idx:
                            continue

                        if other_route.load + customer.demand > self.instance.vehicle_capacity:
                            continue

                        for pos in range(len(other_route.customers) + 1):
                            old_total = route.distance + other_route.distance

                            test_route1 = route.copy()
                            test_route1.customers.pop(cust_pos)
                            test_route1.load -= customer.demand
                            test_route1.calculate_metrics()

                            test_route2 = other_route.copy()
                            test_route2.customers.insert(pos, cust_id)
                            test_route2.load += customer.demand
                            test_route2.calculate_metrics()

                            if (self._is_route_feasible(test_route1) and
                                self._is_route_feasible(test_route2)):
                                new_total = test_route1.distance + test_route2.distance

                                if new_total < old_total - 0.01:
                                    route.customers.pop(cust_pos)
                                    route.load -= customer.demand
                                    route.calculate_metrics()

                                    other_route.customers.insert(pos, cust_id)
                                    other_route.load += customer.demand
                                    other_route.calculate_metrics()

                                    improved = True
                                    break

                        if improved:
                            break

                    if improved:
                        break

                if improved:
                    break

        return solution

    def _two_opt_route(self, route: Route):
        """2-opt optimization within a route"""
        if len(route.customers) <= 3:
            return

        improved = True
        while improved:
            improved = False

            for i in range(len(route.customers) - 1):
                for j in range(i + 2, len(route.customers)):
                    new_customers = (route.customers[:i+1] +
                                   route.customers[i+1:j+1][::-1] +
                                   route.customers[j+1:])

                    test_route = Route(self.instance)
                    test_route.customers = new_customers
                    test_route.load = route.load
                    test_route.calculate_metrics()

                    if (self._is_route_feasible(test_route) and
                        test_route.distance < route.distance - 0.01):
                        route.customers = new_customers
                        route.calculate_metrics()
                        improved = True
                        break

                if improved:
                    break

    def _is_route_feasible(self, route: Route) -> bool:
        """Check route feasibility"""
        if route.load > self.instance.vehicle_capacity:
            return False

        if not route.customers:
            return True

        current_time = self.instance.depot.ready_time

        for cust_id in route.customers:
            customer = self.instance.get_customer(cust_id)
            prev_id = route.customers[route.customers.index(cust_id) - 1] if route.customers.index(cust_id) > 0 else 0

            travel_time = self.instance.distances[prev_id][cust_id]
            arrival = current_time + travel_time

            if arrival > customer.due_date:
                return False

            start_service = max(arrival, customer.ready_time)
            current_time = start_service + customer.service_time

        last_id = route.customers[-1]
        return_time = current_time + self.instance.distances[last_id][0]
        return return_time <= self.instance.depot.due_date


# ============================================================================
# OUTPUT FUNCTIONS (NEW)
# ============================================================================

def create_output_folders():
    """Create folders for output files"""
    folders = ['log_simple_route', 'log_detailed_route', 'route_matp']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def save_simple_log(instance_name: str, pre_opt_solution: Solution,
                    post_opt_solution: Solution, best_strategy: int,
                    computation_time: float):
    """Save simple route log format"""
    filepath = os.path.join('log_simple_route', f'{instance_name}_simple.txt')

    with open(filepath, 'w', encoding='utf-8') as f:
        # Pre-optimization
        pre_dist = pre_opt_solution.calculate_total_distance()
        f.write(f"Best Strategy: {best_strategy}  Total Distance: {pre_dist:.3f}\n")
        f.write(f"Number of vehicles: {pre_opt_solution.num_vehicles()}       ")
        total_customers = sum(len(r.customers) for r in pre_opt_solution.routes)
        f.write(f"Loaded customers: {total_customers}       ")
        f.write(f"Total distance: {pre_dist:.3f}\n")

        for idx, route in enumerate(pre_opt_solution.routes):
            customers_str = ' '.join(map(str, route.customers))
            f.write(f"Route: {idx} {int(route.load)} Vehicle: {idx} ")
            f.write(f"Distance: {route.distance:.4f} ")
            f.write(f"Customers: {len(route.customers)}: 0 {customers_str} 0\n")

        # Post-optimization
        f.write("Local Search optimization process started\n")
        f.write("Local Search optimization process completed\n")

        post_dist = post_opt_solution.calculate_total_distance()
        f.write(f"Number of vehicles: {post_opt_solution.num_vehicles()}       ")
        total_customers = sum(len(r.customers) for r in post_opt_solution.routes)
        f.write(f"Loaded customers: {total_customers}       ")
        f.write(f"Total distance: {post_dist:.3f}\n")

        for idx, route in enumerate(post_opt_solution.routes):
            customers_str = ' '.join(map(str, route.customers))
            f.write(f"Route: {idx} {int(route.load)} Vehicle: {idx} ")
            f.write(f"Distance: {route.distance:.4f} ")
            f.write(f"Customers: {len(route.customers)}: 0 {customers_str} 0\n")

        f.write(f"Time spent: {computation_time*1000:.2f} milliseconds\n")


def save_detailed_log(instance: VRPTWInstance, solution: Solution,
                      best_strategy: int, computation_time: float):
    """Save detailed route log"""
    instance_name = Path(instance.filename).stem
    filepath = os.path.join('log_detailed_route', f'{instance_name}_detailed.txt')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"HYBRID VRPTW SOLUTION - {instance_name.upper()}\n")
        f.write("="*80 + "\n\n")

        # Computing environment
        f.write("COMPUTING ENVIRONMENT\n")
        f.write("-"*40 + "\n")
        f.write(f"Programming Language: Python {platform.python_version()}\n")
        f.write(f"Platform: {platform.platform()}\n")
        f.write(f"Processor: {platform.processor() or platform.machine()}\n")
        f.write(f"CPU Cores: {psutil.cpu_count()}\n")
        f.write(f"RAM (GB): {round(psutil.virtual_memory().total / (1024**3), 2)}\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Problem instance
        f.write("PROBLEM INSTANCE\n")
        f.write("-"*40 + "\n")
        f.write(f"Instance: {instance_name}\n")
        f.write(f"Instance Type: {instance.instance_type}\n")
        f.write(f"Total Customers: {len(instance.customers)}\n")
        f.write(f"Vehicle Capacity: {instance.vehicle_capacity}\n")
        f.write(f"Best Strategy Used: {best_strategy}\n\n")

        # Solution summary
        total_distance = solution.calculate_total_distance()
        total_load = sum(r.load for r in solution.routes)
        avg_distance = total_distance / max(1, len(solution.routes))
        max_distance = max(r.distance for r in solution.routes) if solution.routes else 0

        f.write("SOLUTION SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Total Vehicles Used: {solution.num_vehicles()}\n")
        f.write(f"Total Distance: {total_distance:.2f}\n")
        f.write(f"Average Distance per Vehicle: {avg_distance:.2f}\n")
        f.write(f"Maximum Route Distance: {max_distance:.2f}\n")
        f.write(f"Total Load Delivered: {total_load:.0f}\n")
        f.write(f"Computation Time: {computation_time:.2f} seconds\n\n")

        # Detailed routes
        f.write("DETAILED ROUTES\n")
        f.write("-"*40 + "\n")

        for idx, route in enumerate(solution.routes):
            f.write(f"\nRoute {idx}:\n")
            route_str = " -> ".join(['0'] + [str(c) for c in route.customers] + ['0'])
            f.write(f"  Sequence: {route_str}\n")
            f.write(f"  Distance: {route.distance:.2f}\n")
            f.write(f"  Load: {route.load:.0f}/{instance.vehicle_capacity}\n")
            f.write(f"  Customers: {len(route.customers)}\n")
            f.write(f"  Utilization: {(route.load/instance.vehicle_capacity)*100:.1f}%\n")

            if route.times:
                f.write("  Time Windows:\n")
                for i, cust_id in enumerate(route.customers):
                    if i < len(route.times):
                        customer = instance.get_customer(cust_id)
                        arrival = route.times[i]
                        slack = customer.due_date - arrival
                        f.write(f"    Customer {cust_id}: Arrive={arrival:.1f}, "
                               f"Window=[{customer.ready_time:.0f}, {customer.due_date:.0f}], "
                               f"Slack={slack:.1f}\n")


def plot_solution(instance: VRPTWInstance, solution: Solution,
                  save_path: str = None):
    """Create matplotlib visualization of routes"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot depot
    depot = instance.depot
    ax.scatter(depot.x, depot.y, c='red', s=400, marker='s',
              label='Depot', zorder=6, edgecolors='darkred', linewidth=3)
    ax.annotate('DEPOT', (depot.x, depot.y), fontsize=10, fontweight='bold',
               ha='center', va='center', color='white')

    # Plot all customers
    all_nodes = [depot] + instance.customers
    for customer in instance.customers:
        size = 50 + customer.demand * 3
        ax.scatter(customer.x, customer.y, c='lightblue', s=size,
                  zorder=5, edgecolors='darkblue', linewidth=1.5, alpha=0.8)
        ax.annotate(str(customer.id), (customer.x, customer.y),
                   fontsize=8, fontweight='bold', ha='center', va='center')

    # Color palette
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(solution.routes))))

    # Plot routes
    for idx, route in enumerate(solution.routes):
        if not route.customers:
            continue

        color = colors[idx % len(colors)]

        # Build route coordinates
        route_x = [depot.x]
        route_y = [depot.y]

        for cust_id in route.customers:
            customer = all_nodes[cust_id]
            route_x.append(customer.x)
            route_y.append(customer.y)

        route_x.append(depot.x)
        route_y.append(depot.y)

        # Plot line
        ax.plot(route_x, route_y, color=color, linewidth=2.5, alpha=0.6,
               label=f'V{idx}: {len(route.customers)}c, d={route.distance:.1f}')

        # Add vehicle label at first segment
        if len(route_x) > 2:
            mid_x = (route_x[0] + route_x[1]) / 2
            mid_y = (route_y[0] + route_y[1]) / 2
            ax.annotate(f'V{idx}', (mid_x, mid_y),
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
                       color='white', ha='center', va='center')

    # Labels and title
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    instance_name = Path(instance.filename).stem
    ax.set_title(f'Hybrid VRPTW Solution - {instance_name.upper()}\n'
                f'Vehicles: {solution.num_vehicles()}, '
                f'Distance: {solution.calculate_total_distance():.2f}',
                fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    if len(solution.routes) <= 10:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                 fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def save_diagnostics_report(diagnostics: dict, filepath: str):
    """Save comprehensive diagnostics report"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"DIAGNOSTIC REPORT - {diagnostics['instance_name'].upper()}\n")
        f.write("="*80 + "\n\n")

        # Instance information
        f.write("INSTANCE INFORMATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Instance Name: {diagnostics['instance_name']}\n")
        f.write(f"Instance Type: {diagnostics['instance_type']}\n")
        f.write(f"Computation Time: {diagnostics['computation_time']:.2f} seconds\n\n")

        # Solution quality metrics
        f.write("SOLUTION QUALITY METRICS\n")
        f.write("-"*40 + "\n")
        f.write("Pre-Optimization:\n")
        f.write(f"  Vehicles: {diagnostics['pre_vehicles']}\n")
        f.write(f"  Total Distance: {diagnostics['pre_distance']:.2f}\n")
        f.write(f"  Avg Route Length: {diagnostics['pre_avg_route_length']:.1f} customers\n\n")

        f.write("Post-Optimization:\n")
        f.write(f"  Vehicles: {diagnostics['post_vehicles']}\n")
        f.write(f"  Total Distance: {diagnostics['post_distance']:.2f}\n")
        f.write(f"  Avg Route Length: {diagnostics['post_avg_route_length']:.1f} customers\n\n")

        f.write("Improvements:\n")
        f.write(f"  Vehicle Reduction: {diagnostics['vehicle_reduction']} ({diagnostics['vehicle_reduction_pct']:.1f}%)\n")
        f.write(f"  Distance Reduction: {diagnostics['distance_reduction']:.2f} ({diagnostics['distance_reduction_pct']:.1f}%)\n\n")

        # Route distribution analysis
        f.write("ROUTE DISTRIBUTION ANALYSIS\n")
        f.write("-"*40 + "\n")
        f.write("Customers per Route:\n")
        f.write(f"  Average: {diagnostics['avg_customers_per_route']:.1f}\n")
        f.write(f"  Min: {diagnostics['min_customers_per_route']}\n")
        f.write(f"  Max: {diagnostics['max_customers_per_route']}\n")
        f.write(f"  Std Dev: {diagnostics['std_customers_per_route']:.2f}\n\n")

        f.write("Route Distances:\n")
        f.write(f"  Average: {diagnostics['avg_route_distance']:.2f}\n")
        f.write(f"  Min: {diagnostics['min_route_distance']:.2f}\n")
        f.write(f"  Max: {diagnostics['max_route_distance']:.2f}\n")
        f.write(f"  Std Dev: {diagnostics['std_route_distance']:.2f}\n\n")

        # Capacity utilization
        f.write("CAPACITY UTILIZATION\n")
        f.write("-"*40 + "\n")
        f.write(f"  Average: {diagnostics['avg_utilization']:.1f}%\n")
        f.write(f"  Min: {diagnostics['min_utilization']:.1f}%\n")
        f.write(f"  Max: {diagnostics['max_utilization']:.1f}%\n")
        f.write(f"  Variance: {diagnostics['utilization_variance']:.2f}\n")

        # Quality indicators
        if diagnostics['avg_utilization'] < 60:
            f.write("  ⚠ WARNING: Low average utilization\n")
        if diagnostics['utilization_variance'] > 500:
            f.write("  ⚠ WARNING: High variance in utilization\n")
        f.write("\n")

        # Time window analysis
        f.write("TIME WINDOW ANALYSIS\n")
        f.write("-"*40 + "\n")
        f.write(f"  Average Time Slack: {diagnostics['avg_time_slack']:.2f}\n")
        f.write(f"  Tight Windows: {diagnostics['tight_windows_count']} ({diagnostics['tight_windows_pct']:.1f}%)\n")

        if diagnostics['tight_windows_pct'] > 50:
            f.write("  ⚠ WARNING: Many tight time windows - may limit optimization\n")
        f.write("\n")

        # Construction phase issues
        f.write("CONSTRUCTION PHASE DIAGNOSTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"  Premature Closures: {diagnostics['premature_closures']}\n")
        f.write(f"  Capacity Rejections: {diagnostics['capacity_rejections']}\n")
        f.write(f"  Time Window Rejections: {diagnostics['time_window_rejections']}\n")

        if diagnostics['premature_closures'] > 10:
            f.write("  ⚠ WARNING: High premature closures - routes ending early\n")
        if diagnostics['time_window_rejections'] > diagnostics['capacity_rejections']:
            f.write("  ℹ INFO: Time windows are more restrictive than capacity\n")
        f.write("\n")

        # Strategy information
        f.write("STRATEGY SELECTION\n")
        f.write("-"*40 + "\n")
        f.write(f"  Best Strategy ID: {diagnostics['best_strategy_id']}\n")
        f.write(f"  Best Strategy: {diagnostics['best_strategy_name']}\n\n")

        # Overall assessment
        f.write("OVERALL ASSESSMENT\n")
        f.write("-"*40 + "\n")

        score = 0
        max_score = 5

        # Scoring criteria
        if diagnostics['vehicle_reduction'] > 0:
            score += 1
            f.write("  ✓ Successfully reduced vehicle count\n")
        else:
            f.write("  ✗ No vehicle reduction achieved\n")

        if diagnostics['distance_reduction'] > 0:
            score += 1
            f.write("  ✓ Successfully reduced total distance\n")
        else:
            f.write("  ✗ No distance reduction achieved\n")

        if diagnostics['avg_utilization'] >= 70:
            score += 1
            f.write("  ✓ Good capacity utilization (≥70%)\n")
        else:
            f.write("  ✗ Low capacity utilization (<70%)\n")

        if diagnostics['avg_customers_per_route'] >= 7:
            score += 1
            f.write("  ✓ Good route density (≥7 customers/route)\n")
        else:
            f.write("  ✗ Low route density (<7 customers/route)\n")

        if diagnostics['premature_closures'] < 5:
            score += 1
            f.write("  ✓ Few premature route closures\n")
        else:
            f.write("  ✗ Many premature route closures\n")

        f.write(f"\nQuality Score: {score}/{max_score}\n")

        if score >= 4:
            f.write("Overall Rating: EXCELLENT\n")
        elif score >= 3:
            f.write("Overall Rating: GOOD\n")
        elif score >= 2:
            f.write("Overall Rating: FAIR\n")
        else:
            f.write("Overall Rating: NEEDS IMPROVEMENT\n")

        f.write("\n" + "="*80 + "\n")


def create_benchmark_tables(results_dict: dict):
    """Create benchmark comparison tables like Table 1 and Table 2"""

    # Calculate averages for each group
    def calc_group_avg_pre(group_results):
        """Calculate average for pre-optimization (uses indices 1 and 2)"""
        if not group_results:
            return 0, 0
        total_v = sum(x[1] for x in group_results)  # pre_v
        total_d = sum(x[2] for x in group_results)  # pre_d
        count = len(group_results)
        return total_v / count, total_d / count

    def calc_group_avg_post(group_results):
        """Calculate average for post-optimization (uses indices 3 and 4)"""
        if not group_results:
            return 0, 0
        total_v = sum(x[3] for x in group_results)  # post_v
        total_d = sum(x[4] for x in group_results)  # post_d
        count = len(group_results)
        return total_v / count, total_d / count

    # Organize results by group - CHECK RC BEFORE R!
    groups = {'R1': [], 'R2': [], 'C1': [], 'C2': [], 'RC1': [], 'RC2': []}

    for inst_name, pre_v, pre_d, post_v, post_d in results_dict.values():
        # Check RC first (before R)
        if inst_name.startswith('RC'):
            if int(inst_name[2:]) <= 108:
                groups['RC1'].append((inst_name, pre_v, pre_d, post_v, post_d))
            else:
                groups['RC2'].append((inst_name, pre_v, pre_d, post_v, post_d))
        elif inst_name.startswith('R'):
            if int(inst_name[1:]) <= 112:
                groups['R1'].append((inst_name, pre_v, pre_d, post_v, post_d))
            else:
                groups['R2'].append((inst_name, pre_v, pre_d, post_v, post_d))
        elif inst_name.startswith('C'):
            if int(inst_name[1:]) <= 109:
                groups['C1'].append((inst_name, pre_v, pre_d, post_v, post_d))
            else:
                groups['C2'].append((inst_name, pre_v, pre_d, post_v, post_d))

    # Calculate totals and averages
    total_pre_v, total_pre_d = 0, 0
    total_post_v, total_post_d = 0, 0

    for group_results in groups.values():
        for _, pre_v, pre_d, post_v, post_d in group_results:
            total_pre_v += pre_v
            total_pre_d += pre_d
            total_post_v += post_v
            total_post_d += post_d

    # Table 1: Route Construction Heuristics (Before Optimization)
    table1_path = os.path.join('logs_results_tables', 'Table1_Construction_Heuristics.txt')
    with open(table1_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("Table 1    Route Construction Heuristics\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Author':<35} {'R1':>12} {'R2':>12} {'C1':>12} {'C2':>12} {'RC1':>12} {'RC2':>12} {'CNV/CTD':>12}\n")
        f.write("-"*100 + "\n")

        # Benchmark results from literature
        benchmarks = [
            ("(1) Solomon (1987)", 13.58, 1436.7, 3.27, 1402.4, 10.00, 951.9, 3.13, 692.7, 13.50, 1596.5, 3.88, 1682.1, 453, 73004),
            ("(2) Potvin et al. (1993)", 13.33, 1509.04, 3.09, 1386.67, 10.67, 1343.69, 3.38, 797.59, 13.38, 1723.72, 3.63, 1651.05, 453, 78834),
            ("(3) Ioannou et al. (2001)", 12.67, 1370, 3.09, 1310, 10.00, 865, 3.13, 662, 12.50, 1512, 3.50, 1483, 429, 67891),
        ]

        for name, r1_v, r1_d, r2_v, r2_d, c1_v, c1_d, c2_v, c2_d, rc1_v, rc1_d, rc2_v, rc2_d, cnv, ctd in benchmarks:
            f.write(f"{name:<35} {r1_v:>12.2f} {r2_v:>12.2f} {c1_v:>12.2f} {c2_v:>12.2f} {rc1_v:>12.2f} {rc2_v:>12.2f} {cnv:>12}\n")
            f.write(f"{'':<35} {r1_d:>12.1f} {r2_d:>12.1f} {c1_d:>12.1f} {c2_d:>12.1f} {rc1_d:>12.1f} {rc2_d:>12.1f} {ctd:>12.0f}\n")

        # Our results (before optimization) - use calc_group_avg_pre
        r1_avg_v, r1_avg_d = calc_group_avg_pre(groups['R1'])
        r2_avg_v, r2_avg_d = calc_group_avg_pre(groups['R2'])
        c1_avg_v, c1_avg_d = calc_group_avg_pre(groups['C1'])
        c2_avg_v, c2_avg_d = calc_group_avg_pre(groups['C2'])
        rc1_avg_v, rc1_avg_d = calc_group_avg_pre(groups['RC1'])
        rc2_avg_v, rc2_avg_d = calc_group_avg_pre(groups['RC2'])

        f.write("-"*100 + "\n")
        f.write(f"{'(4) This Implementation (2025)':<35} {r1_avg_v:>12.2f} {r2_avg_v:>12.2f} {c1_avg_v:>12.2f} {c2_avg_v:>12.2f} {rc1_avg_v:>12.2f} {rc2_avg_v:>12.2f} {total_pre_v:>12}\n")
        f.write(f"{'':<35} {r1_avg_d:>12.1f} {r2_avg_d:>12.1f} {c1_avg_d:>12.1f} {c2_avg_d:>12.1f} {rc1_avg_d:>12.1f} {rc2_avg_d:>12.1f} {total_pre_d:>12.0f}\n")
        f.write("="*100 + "\n")

        f.write("\nNote: For all algorithms, the average results for Solomon's benchmarks are described.\n")
        f.write("CNV = Cumulative Number of Vehicles, CTD = Cumulative Total Distance over all 56 test problems.\n")

    # Table 2: Local Search Algorithms (After Optimization)
    table2_path = os.path.join('logs_results_tables', 'Table2_Local_Search.txt')
    with open(table2_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("Table 2    Local Search Algorithms\n")
        f.write("="*100 + "\n\n")
        f.write(f"{'Author':<35} {'R1':>12} {'R2':>12} {'C1':>12} {'C2':>12} {'RC1':>12} {'RC2':>12} {'CNV/CTD':>12}\n")
        f.write("-"*100 + "\n")

        # Benchmark results from literature
        benchmarks = [
            ("(1) Thompson et al. (1993)", 13.00, 1356.92, 3.18, 1276.00, 10.00, 916.67, 3.00, 644.63, 13.00, 1514.29, 3.71, 1634.43, 438, 68916),
            ("(2) Potvin et al. (1995)", 13.33, 1381.9, 3.27, 1293.4, 10.00, 902.9, 3.13, 653.2, 13.25, 1545.3, 3.88, 1595.1, 448, 69285),
            ("(3) Russell (1995)", 12.66, 1317, 2.91, 1167, 10.00, 930, 3.00, 681, 12.38, 1523, 3.38, 1398, 424, 65827),
            ("(4) Antes et al. (1995)", 12.83, 1386.46, 3.09, 1366.48, 10.00, 955.39, 3.00, 717.31, 12.50, 1545.92, 3.38, 1598.06, 429, 71158),
            ("(5) Prosser et al. (1996)", 13.50, 1242.40, 4.09, 977.12, 10.00, 843.84, 3.13, 607.58, 13.50, 1408.76, 5.13, 1111.37, 471, 58273),
            ("(6) Caseau et al. (1999)", 12.42, 1233.34, 3.09, 990.99, 10.00, 828.38, 3.00, 596.63, 12.00, 1403.74, 3.38, 1220.99, 420, 58927),
            ("(7) Schrimpf et al. (2000)", 12.08, 1211.53, 2.82, 949.27, 10.00, 828.38, 3.00, 589.86, 11.88, 1361.76, 3.38, 1097.63, 412, 56830),
            ("(8) Cordone et al. (2001)", 12.50, 1241.89, 2.91, 995.39, 10.00, 834.05, 3.00, 591.78, 12.38, 1408.87, 3.38, 1139.70, 422, 58481),
            ("(9) Bräysy (2003)", 12.17, 1253.24, 2.82, 1039.56, 10.00, 832.88, 3.00, 593.49, 11.88, 1408.44, 3.25, 1244.96, 412, 59945),
        ]

        for name, r1_v, r1_d, r2_v, r2_d, c1_v, c1_d, c2_v, c2_d, rc1_v, rc1_d, rc2_v, rc2_d, cnv, ctd in benchmarks:
            f.write(f"{name:<35} {r1_v:>12.2f} {r2_v:>12.2f} {c1_v:>12.2f} {c2_v:>12.2f} {rc1_v:>12.2f} {rc2_v:>12.2f} {cnv:>12}\n")
            f.write(f"{'':<35} {r1_d:>12.1f} {r2_d:>12.1f} {c1_d:>12.1f} {c2_d:>12.1f} {rc1_d:>12.1f} {rc2_d:>12.1f} {ctd:>12.0f}\n")

        # Our results (after optimization) - use calc_group_avg_post
        r1_avg_v, r1_avg_d = calc_group_avg_post(groups['R1'])
        r2_avg_v, r2_avg_d = calc_group_avg_post(groups['R2'])
        c1_avg_v, c1_avg_d = calc_group_avg_post(groups['C1'])
        c2_avg_v, c2_avg_d = calc_group_avg_post(groups['C2'])
        rc1_avg_v, rc1_avg_d = calc_group_avg_post(groups['RC1'])
        rc2_avg_v, rc2_avg_d = calc_group_avg_post(groups['RC2'])

        f.write("-"*100 + "\n")
        f.write(f"{'(10) This Implementation (2025)':<35} {r1_avg_v:>12.2f} {r2_avg_v:>12.2f} {c1_avg_v:>12.2f} {c2_avg_v:>12.2f} {rc1_avg_v:>12.2f} {rc2_avg_v:>12.2f} {total_post_v:>12}\n")
        f.write(f"{'':<35} {r1_avg_d:>12.1f} {r2_avg_d:>12.1f} {c1_avg_d:>12.1f} {c2_avg_d:>12.1f} {rc1_avg_d:>12.1f} {rc2_avg_d:>12.1f} {total_post_d:>12.0f}\n")
        f.write("="*100 + "\n")

        f.write("\nNote: For each method, average results for Solomon's benchmarks are presented.\n")
        f.write("CNV = Cumulative Number of Vehicles, CTD = Cumulative Total Distance over all test problems.\n")

    print(f"\n✓ Benchmark tables created:")
    print(f"  - Table 1: logs_results_tables/Table1_Construction_Heuristics.txt")
    print(f"  - Table 2: logs_results_tables/Table2_Local_Search.txt")


def main():
    print("=" * 80)
    print("Hybrid VRPTW Heuristic with Enhanced Output")
    print("=" * 80)

    # Create output folders
    create_output_folders()
    os.makedirs('logs_results_common', exist_ok=True)
    os.makedirs('logs_results_tables', exist_ok=True)

    instance_name = input("\nEnter instance name (e.g., c101) or 'all': ").strip().lower()
    directory = input("Enter directory path: ").strip()

    if instance_name == 'all':
        process_all_instances(directory)
    else:
        filename = os.path.join(directory, f"{instance_name.upper()}.txt")
        if not os.path.exists(filename):
            print(f"Error: File {filename} not found")
            return

        instance = VRPTWInstance(filename)
        solver = HybridSolver(instance)

        start_time = time.time()
        solution = solver.solve()
        computation_time = time.time() - start_time

        instance_name = Path(filename).stem

        # Get pre-optimization metrics
        if hasattr(solver, 'pre_opt_solution') and solver.pre_opt_solution:
            pre_vehicles = solver.pre_opt_solution.num_vehicles()
            pre_distance = solver.pre_opt_solution.calculate_total_distance()
        else:
            pre_vehicles = solution.num_vehicles()
            pre_distance = solution.calculate_total_distance()

        post_vehicles = solution.num_vehicles()
        post_distance = solution.calculate_total_distance()

        # Console output
        print(f"\n{instance_name}... Before Optimization: {pre_vehicles} vehicles, {pre_distance:.2f} distance")
        print(f"{'':9}After Optimization: {post_vehicles} vehicles, {post_distance:.2f} distance")

        # Save outputs
        if hasattr(solver, 'pre_opt_solution') and solver.pre_opt_solution:
            save_simple_log(instance_name, solver.pre_opt_solution, solution,
                          solver.diagnostics['best_strategy'], computation_time)

            # Generate and save enhanced diagnostics
            enhanced_diag = solver.get_enhanced_diagnostics(
                solver.pre_opt_solution, solution, computation_time
            )
            diag_path = os.path.join('diagnostics', f'{instance_name}_diagnostics.txt')
            save_diagnostics_report(enhanced_diag, diag_path)

        save_detailed_log(instance, solution, solver.diagnostics['best_strategy'],
                         computation_time)

        plot_path = os.path.join('route_matp', f'{instance_name}_routes.png')
        plot_solution(instance, solution, plot_path)

        print(f"\n✓ All outputs saved successfully")


def process_all_instances(directory: str):
    instance_groups = {
        'R1': ['R101', 'R102', 'R103', 'R104', 'R105', 'R106', 'R107', 'R108', 'R109', 'R110', 'R111', 'R112'],
        'R2': ['R201', 'R202', 'R203', 'R204', 'R205', 'R206', 'R207', 'R208', 'R209', 'R210', 'R211'],
        'C1': ['C101', 'C102', 'C103', 'C104', 'C105', 'C106', 'C107', 'C108', 'C109'],
        'C2': ['C201', 'C202', 'C203', 'C204', 'C205', 'C206', 'C207', 'C208'],
        'RC1': ['RC101', 'RC102', 'RC103', 'RC104', 'RC105', 'RC106', 'RC107', 'RC108'],
        'RC2': ['RC201', 'RC202', 'RC203', 'RC204', 'RC205', 'RC206', 'RC207', 'RC208']
    }

    results = {}
    all_results = {}  # For benchmark tables

    for group_name, instances in instance_groups.items():
        print(f"\n{'=' * 80}")
        print(f"Processing {group_name} instances...")
        print(f"{'=' * 80}")

        group_results = []

        for inst_name in instances:
            filename = os.path.join(directory, f"{inst_name}.txt")
            if not os.path.exists(filename):
                continue

            instance = VRPTWInstance(filename)
            solver = HybridSolver(instance, enable_logging=False)

            start_time = time.time()
            solution = solver.solve()
            computation_time = time.time() - start_time

            # Get pre and post optimization metrics
            if hasattr(solver, 'pre_opt_solution') and solver.pre_opt_solution:
                pre_vehicles = solver.pre_opt_solution.num_vehicles()
                pre_distance = solver.pre_opt_solution.calculate_total_distance()
            else:
                pre_vehicles = solution.num_vehicles()
                pre_distance = solution.calculate_total_distance()

            post_vehicles = solution.num_vehicles()
            post_distance = solution.calculate_total_distance()

            # Console output with before/after format
            print(f"\n{inst_name}... Before Optimization: {pre_vehicles} vehicles, {pre_distance:.2f} distance")
            print(f"{'':9}After Optimization: {post_vehicles} vehicles, {post_distance:.2f} distance")

            # Save all outputs
            if hasattr(solver, 'pre_opt_solution') and solver.pre_opt_solution:
                save_simple_log(inst_name, solver.pre_opt_solution, solution,
                              solver.diagnostics['best_strategy'], computation_time)

                # Generate and save enhanced diagnostics
                enhanced_diag = solver.get_enhanced_diagnostics(
                    solver.pre_opt_solution, solution, computation_time
                )
                diag_path = os.path.join('diagnostics', f'{inst_name}_diagnostics.txt')
                save_diagnostics_report(enhanced_diag, diag_path)

            save_detailed_log(instance, solution, solver.diagnostics['best_strategy'],
                            computation_time)

            plot_path = os.path.join('route_matp', f'{inst_name}_routes.png')
            plot_solution(instance, solution, plot_path)

            group_results.append((inst_name, post_vehicles, post_distance))

            # Store for benchmark tables
            all_results[inst_name] = (inst_name, pre_vehicles, pre_distance, post_vehicles, post_distance)

        results[group_name] = group_results

    # Save summary table to logs_results_common
    summary_path = os.path.join('logs_results_common', 'summary_results.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("HYBRID HEURISTIC RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Instance':<12} {'Vehicles':<12} {'Distance':<12}\n")
        f.write("-" * 80 + "\n")

        total_vehicles = 0
        total_distance = 0

        for group_name in ['R1', 'R2', 'C1', 'C2', 'RC1', 'RC2']:
            if group_name in results:
                for inst_name, vehicles, distance in results[group_name]:
                    f.write(f"{inst_name:<12} {vehicles:<12} {distance:<12.2f}\n")
                    total_vehicles += vehicles
                    total_distance += distance

        f.write("=" * 80 + "\n")
        f.write(f"{'TOTAL':<12} {total_vehicles:<12} {total_distance:<12.0f}\n")
        f.write("=" * 80 + "\n")

    # Print summary to console
    print("\n" + "=" * 80)
    print("HYBRID HEURISTIC RESULTS")
    print("=" * 80)
    print(f"{'Instance':<12} {'Vehicles':<12} {'Distance':<12}")
    print("-" * 80)

    total_vehicles = 0
    total_distance = 0

    for group_name in ['R1', 'R2', 'C1', 'C2', 'RC1', 'RC2']:
        if group_name in results:
            for inst_name, vehicles, distance in results[group_name]:
                print(f"{inst_name:<12} {vehicles:<12} {distance:<12.2f}")
                total_vehicles += vehicles
                total_distance += distance

    print("=" * 80)
    print(f"{'TOTAL':<12} {total_vehicles:<12} {total_distance:<12.0f}")
    print("=" * 80)

    # Create benchmark comparison tables
    create_benchmark_tables(all_results)

    # Create aggregate diagnostics summary
    create_aggregate_diagnostics(all_results, results)

    print(f"\n✓ All outputs saved:")
    print(f"  - Simple logs: log_simple_route/")
    print(f"  - Detailed logs: log_detailed_route/")
    print(f"  - Plots: route_matp/")
    print(f"  - Summary table: logs_results_common/summary_results.txt")
    print(f"  - Benchmark tables: logs_results_tables/")
    print(f"  - Diagnostics: diagnostics/")


def create_aggregate_diagnostics(all_results: dict, results: dict):
    """Create aggregate diagnostics summary across all instances"""

    summary_path = os.path.join('diagnostics', 'aggregate_diagnostics.txt')

    # Collect stats by instance type - CHECK RC BEFORE R!
    type_stats = {'R1': [], 'R2': [], 'C1': [], 'C2': [], 'RC1': [], 'RC2': []}

    for inst_name, pre_v, pre_d, post_v, post_d in all_results.values():
        vehicle_reduction = pre_v - post_v
        distance_reduction = pre_d - post_d

        # Check RC first (before R)
        if inst_name.startswith('RC'):
            if int(inst_name[2:]) <= 108:
                type_stats['RC1'].append((inst_name, pre_v, pre_d, post_v, post_d, vehicle_reduction, distance_reduction))
            else:
                type_stats['RC2'].append((inst_name, pre_v, pre_d, post_v, post_d, vehicle_reduction, distance_reduction))
        elif inst_name.startswith('R'):
            if int(inst_name[1:]) <= 112:
                type_stats['R1'].append((inst_name, pre_v, pre_d, post_v, post_d, vehicle_reduction, distance_reduction))
            else:
                type_stats['R2'].append((inst_name, pre_v, pre_d, post_v, post_d, vehicle_reduction, distance_reduction))
        elif inst_name.startswith('C'):
            if int(inst_name[1:]) <= 109:
                type_stats['C1'].append((inst_name, pre_v, pre_d, post_v, post_d, vehicle_reduction, distance_reduction))
            else:
                type_stats['C2'].append((inst_name, pre_v, pre_d, post_v, post_d, vehicle_reduction, distance_reduction))

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("AGGREGATE DIAGNOSTICS SUMMARY - ALL INSTANCES\n")
        f.write("="*100 + "\n\n")

        # Overall statistics
        total_pre_v = sum(pre_v for _, pre_v, _, _, _ in all_results.values())
        total_pre_d = sum(pre_d for _, _, pre_d, _, _ in all_results.values())
        total_post_v = sum(post_v for _, _, _, post_v, _ in all_results.values())
        total_post_d = sum(post_d for _, _, _, _, post_d in all_results.values())

        total_v_reduction = total_pre_v - total_post_v
        total_d_reduction = total_pre_d - total_post_d

        f.write("OVERALL STATISTICS\n")
        f.write("-"*100 + "\n")
        f.write(f"Total Instances Processed: {len(all_results)}\n")
        f.write(f"Total Vehicles Before: {total_pre_v}\n")
        f.write(f"Total Vehicles After: {total_post_v}\n")
        f.write(f"Total Vehicle Reduction: {total_v_reduction} ({total_v_reduction/total_pre_v*100:.1f}%)\n")
        f.write(f"Total Distance Before: {total_pre_d:.2f}\n")
        f.write(f"Total Distance After: {total_post_d:.2f}\n")
        f.write(f"Total Distance Reduction: {total_d_reduction:.2f} ({total_d_reduction/total_pre_d*100:.1f}%)\n\n")

        # Statistics by instance type
        f.write("STATISTICS BY INSTANCE TYPE\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Type':<8} {'Count':<8} {'Avg Pre-V':<12} {'Avg Post-V':<12} {'Avg V-Red':<12} {'Avg D-Red':<15} {'Max V-Red':<12}\n")
        f.write("-"*100 + "\n")

        for type_name in ['R1', 'R2', 'C1', 'C2', 'RC1', 'RC2']:
            if type_stats[type_name]:
                instances = type_stats[type_name]
                count = len(instances)
                avg_pre_v = np.mean([x[1] for x in instances])
                avg_post_v = np.mean([x[3] for x in instances])
                avg_v_red = np.mean([x[5] for x in instances])
                avg_d_red = np.mean([x[6] for x in instances])
                max_v_red = max([x[5] for x in instances])

                f.write(f"{type_name:<8} {count:<8} {avg_pre_v:<12.2f} {avg_post_v:<12.2f} "
                       f"{avg_v_red:<12.2f} {avg_d_red:<15.2f} {max_v_red:<12}\n")

        f.write("\n")

        # Best improvements
        f.write("TOP 10 IMPROVEMENTS (by vehicle reduction)\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Instance':<12} {'Pre-V':<10} {'Post-V':<10} {'V-Reduction':<15} {'Distance Reduction':<20}\n")
        f.write("-"*100 + "\n")

        all_improvements = [(inst_name, pre_v, post_v, v_red, d_red)
                           for inst_name, pre_v, pre_d, post_v, post_d in all_results.values()
                           for v_red in [pre_v - post_v]
                           for d_red in [pre_d - post_d]]

        top_improvements = sorted(all_improvements, key=lambda x: x[3], reverse=True)[:10]

        for inst_name, pre_v, post_v, v_red, d_red in top_improvements:
            f.write(f"{inst_name:<12} {pre_v:<10} {post_v:<10} {v_red:<15} {d_red:<20.2f}\n")

        f.write("\n")

        # Instances with no improvement
        f.write("INSTANCES WITH NO VEHICLE REDUCTION\n")
        f.write("-"*100 + "\n")

        no_improvement = [(inst_name, pre_v, post_v, pre_d, post_d)
                         for inst_name, pre_v, pre_d, post_v, post_d in all_results.values()
                         if pre_v == post_v]

        if no_improvement:
            f.write(f"{'Instance':<12} {'Vehicles':<12} {'Pre-Distance':<15} {'Post-Distance':<15} {'D-Reduction':<15}\n")
            f.write("-"*100 + "\n")
            for inst_name, pre_v, post_v, pre_d, post_d in no_improvement:
                d_red = pre_d - post_d
                f.write(f"{inst_name:<12} {post_v:<12} {pre_d:<15.2f} {post_d:<15.2f} {d_red:<15.2f}\n")
        else:
            f.write("All instances achieved vehicle reduction!\n")

        f.write("\n")

        # Performance assessment
        f.write("PERFORMANCE ASSESSMENT\n")
        f.write("-"*100 + "\n")

        instances_with_v_reduction = sum(1 for _, pre_v, _, post_v, _ in all_results.values() if post_v < pre_v)
        instances_with_d_reduction = sum(1 for _, _, pre_d, _, post_d in all_results.values() if post_d < pre_d)

        v_reduction_pct = instances_with_v_reduction / len(all_results) * 100
        d_reduction_pct = instances_with_d_reduction / len(all_results) * 100

        f.write(f"Instances with Vehicle Reduction: {instances_with_v_reduction}/{len(all_results)} ({v_reduction_pct:.1f}%)\n")
        f.write(f"Instances with Distance Reduction: {instances_with_d_reduction}/{len(all_results)} ({d_reduction_pct:.1f}%)\n")

        avg_v_reduction = total_v_reduction / len(all_results)
        avg_d_reduction = total_d_reduction / len(all_results)

        f.write(f"Average Vehicle Reduction per Instance: {avg_v_reduction:.2f}\n")
        f.write(f"Average Distance Reduction per Instance: {avg_d_reduction:.2f}\n\n")

        # Quality rating
        score = 0
        max_score = 4

        if v_reduction_pct >= 80:
            score += 1
            f.write("✓ Excellent vehicle reduction rate (≥80%)\n")
        elif v_reduction_pct >= 60:
            f.write("○ Good vehicle reduction rate (≥60%)\n")
        else:
            f.write("✗ Low vehicle reduction rate (<60%)\n")

        if avg_v_reduction >= 1.0:
            score += 1
            f.write("✓ Strong average vehicle reduction (≥1.0)\n")
        elif avg_v_reduction >= 0.5:
            score += 0.5
            f.write("○ Moderate average vehicle reduction (≥0.5)\n")
        else:
            f.write("✗ Weak average vehicle reduction (<0.5)\n")

        if d_reduction_pct >= 90:
            score += 1
            f.write("✓ Excellent distance reduction rate (≥90%)\n")
        elif d_reduction_pct >= 70:
            f.write("○ Good distance reduction rate (≥70%)\n")
        else:
            f.write("✗ Low distance reduction rate (<70%)\n")

        if avg_d_reduction >= 50:
            score += 1
            f.write("✓ Strong average distance reduction (≥50)\n")
        elif avg_d_reduction >= 20:
            score += 0.5
            f.write("○ Moderate average distance reduction (≥20)\n")
        else:
            f.write("✗ Weak average distance reduction (<20)\n")

        f.write(f"\nOverall Score: {score:.1f}/{max_score}\n")

        if score >= 3.5:
            f.write("Overall Rating: EXCELLENT - Post-optimization is highly effective\n")
        elif score >= 2.5:
            f.write("Overall Rating: GOOD - Post-optimization provides solid improvements\n")
        elif score >= 1.5:
            f.write("Overall Rating: FAIR - Post-optimization provides moderate improvements\n")
        else:
            f.write("Overall Rating: NEEDS IMPROVEMENT - Post-optimization is limited\n")

        f.write("\n" + "="*100 + "\n")


def create_output_folders():
    """Create folders for output files"""
    folders = ['log_simple_route', 'log_detailed_route', 'route_matp',
               'logs_results_common', 'logs_results_tables', 'diagnostics']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def main():
    print("=" * 80)
    print("Hybrid VRPTW Heuristic with Enhanced Output")
    print("=" * 80)

    # Create output folders
    create_output_folders()

    instance_name = input("\nEnter instance name (e.g., c101) or 'all': ").strip().lower()
    directory = input("Enter directory path: ").strip()

    if instance_name == 'all':
        process_all_instances(directory)
    else:
        filename = os.path.join(directory, f"{instance_name.upper()}.txt")
        if not os.path.exists(filename):
            print(f"Error: File {filename} not found")
            return

        instance = VRPTWInstance(filename)
        solver = HybridSolver(instance)

        start_time = time.time()
        solution = solver.solve()
        computation_time = time.time() - start_time

        instance_name = Path(filename).stem

        # Get pre-optimization metrics
        if hasattr(solver, 'pre_opt_solution') and solver.pre_opt_solution:
            pre_vehicles = solver.pre_opt_solution.num_vehicles()
            pre_distance = solver.pre_opt_solution.calculate_total_distance()
        else:
            pre_vehicles = solution.num_vehicles()
            pre_distance = solution.calculate_total_distance()

        post_vehicles = solution.num_vehicles()
        post_distance = solution.calculate_total_distance()

        # Console output
        print(f"\n{instance_name}... Before Optimization: {pre_vehicles} vehicles, {pre_distance:.2f} distance")
        print(f"{'':9}After Optimization: {post_vehicles} vehicles, {post_distance:.2f} distance")

        # Save outputs
        if hasattr(solver, 'pre_opt_solution') and solver.pre_opt_solution:
            save_simple_log(instance_name, solver.pre_opt_solution, solution,
                          solver.diagnostics['best_strategy'], computation_time)

        save_detailed_log(instance, solution, solver.diagnostics['best_strategy'],
                         computation_time)

        plot_path = os.path.join('route_matp', f'{instance_name}_routes.png')
        plot_solution(instance, solution, plot_path)

        print(f"\n✓ All outputs saved successfully")


if __name__ == "__main__":
    main()
