"""
Alloc8 v5.4: Multimodal Backend - Parameter Echo & Strict Constraints
"""

import json
import logging
import math
import os

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.optimize import linprog

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
ALLOWED_ORIGINS = [
    "https://alloc8-rho.vercel.app",
    "http://127.0.0.1:8080",
    "http://localhost:8080",
]

CORS(
    app,
    resources={r"/*": {"origins": ALLOWED_ORIGINS}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    supports_credentials=True,
)


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


CONSTANTS = {
    "osrm_base_url": "http://router.project-osrm.org",
    "loading_time_per_kg": 0.01,  # Reduced to negligible to prevent time-out issues
    "fixed_stop_time": 600,
    "max_driver_dist_km": 5000,
    "max_shift_time_sec": 86400,
    "vehicle_capacity": 5000,  # Default fallback if no input
    "speed_mps_road": 13.0,
    "speed_mps_boat": 8.5,
    "speed_mps_air": 220.0,
    "air_threshold_km": 600,
    "air_docking_time": 3600,
}


def get_haversine_distance(coord1, coord2):
    R = 6371000
    phi1, phi2 = math.radians(coord1[0]), math.radians(coord2[0])
    dphi = math.radians(coord2[0] - coord1[0])
    dlambda = math.radians(coord2[1] - coord1[1])

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return int(R * c)


def get_multimodal_matrix(coords):
    n = len(coords)
    formatted_coords = ";".join([f"{c[1]},{c[0]}" for c in coords])
    url = f"{CONSTANTS['osrm_base_url']}/table/v1/driving/{formatted_coords}"
    params = {"annotations": "distance,duration", "skip_waypoints": "false"}

    dist_matrix = [[0] * n for _ in range(n)]
    time_matrix = [[0] * n for _ in range(n)]
    mode_matrix = [["road"] * n for _ in range(n)]

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if data["code"] != "Ok":
            raise Exception("OSRM Error")

        raw_dists = data["distances"]
        raw_times = data["durations"]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                geo_dist = get_haversine_distance(coords[i], coords[j])
                if geo_dist > (CONSTANTS["air_threshold_km"] * 1000):
                    dist_matrix[i][j] = geo_dist
                    time_matrix[i][j] = int(
                        (geo_dist / CONSTANTS["speed_mps_air"])
                        + CONSTANTS["air_docking_time"]
                    )
                    mode_matrix[i][j] = "air"
                    continue

                osrm_dist = raw_dists[i][j]
                osrm_time = raw_times[i][j]

                if osrm_dist is None or osrm_dist > (geo_dist * 3.0):
                    dist_matrix[i][j] = geo_dist
                    time_matrix[i][j] = int(geo_dist / CONSTANTS["speed_mps_boat"])
                    mode_matrix[i][j] = "boat"
                else:
                    dist_matrix[i][j] = int(osrm_dist)
                    time_matrix[i][j] = int(osrm_time)
                    mode_matrix[i][j] = "road"

    except Exception as e:
        logging.warning(f"OSRM Error ({e}). Using fallback Haversine.")
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d_m = get_haversine_distance(coords[i], coords[j])
                dist_matrix[i][j] = int(d_m)
                time_matrix[i][j] = int(d_m / CONSTANTS["speed_mps_road"])
                mode_matrix[i][j] = "road"

    return dist_matrix, time_matrix, mode_matrix


def get_leg_geometry(coord_start, coord_end, mode):
    if mode == "road":
        formatted_coords = (
            f"{coord_start[1]},{coord_start[0]};{coord_end[1]},{coord_end[0]}"
        )
        url = f"{CONSTANTS['osrm_base_url']}/route/v1/driving/{formatted_coords}"
        params = {"overview": "full", "geometries": "geojson"}
        try:
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            if data["code"] == "Ok":
                return data["routes"][0]["geometry"]["coordinates"]
        except:
            pass
    return [[coord_start[1], coord_start[0]], [coord_end[1], coord_end[0]]]


def solve_allocation_lp(demands, fleet_cap, priorities):
    n = len(demands)
    total_demand = sum(demands)
    if n == 0:
        return []

    # If capacity is sufficient, give full demand
    if total_demand <= fleet_cap:
        return demands

    # Linear Programming for Scarcity
    c = [-p for p in priorities]
    A_ub, b_ub = [[1] * n], [fleet_cap]
    bounds = []

    # Give everyone a fair share baseline, then optimize the rest
    fair_share_ratio = fleet_cap / total_demand
    for d in demands:
        if d == 0:
            bounds.append((0, 0))
        else:
            # Min 10% of demand to avoid zero-service if possible
            min_alloc = int(d * 0.1)
            bounds.append((min_alloc, int(d)))

    if sum(b[0] for b in bounds) > fleet_cap:
        # Bounds too tight, revert to strict proportional
        return [int(d * fair_share_ratio) for d in demands]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.success:
        return [int(x) for x in res.x]

    return [int(d * fair_share_ratio) for d in demands]


@app.route("/generate-plan", methods=["POST", "OPTIONS"])
def generate_plan():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json(force=True)

        strategy = data.get("strategy", "welfare")
        raw_parsed_needs = data.get("parsedNeeds", {})
        raw_locations = raw_parsed_needs.get("locations", [])

        # --- 1. Filter Valid Locations ---
        #
        # --- 0. Resolve Depot FIRST ---
        depot_data = data.get("depot", {})
        if depot_data and "lat" in depot_data:
            depot = depot_data
        else:
            depot = {
                "lat": raw_locations[0]["lat"] if raw_locations else 28.61,
                "lon": raw_locations[0]["lon"] if raw_locations else 77.23,
                "name": "Main Distribution Center",
            }

        depot_lat = depot.get("lat")
        depot_lon = depot.get("lon")

        # --- 1. Filter Valid Locations (EXCLUDING DEPOT) ---
        valid_locations = []

        for loc in raw_locations:
            # 🚫 Skip depot entirely
            if (
                depot_lat is not None
                and depot_lon is not None
                and abs(loc.get("lat") - depot_lat) < 1e-6
                and abs(loc.get("lon") - depot_lon) < 1e-6
            ):
                continue

            needs = loc.get("needs", {})
            total_req = sum(int(v) for v in needs.values())

            if total_req > 0:
                valid_locations.append(loc)

        if not valid_locations:
            valid_locations = []

        # --- 2. Process Constraints (STRICT) ---
        # Parse capacity safely
        try:
            input_capacity = int(data.get("vehicle_capacity", 5000))
        except (ValueError, TypeError):
            input_capacity = 5000

        try:
            input_max_fleet = int(data.get("max_fleet_size", 20))
        except (ValueError, TypeError):
            input_max_fleet = 20

        # Calculate Demands
        raw_demands = []
        priorities = []
        for loc in valid_locations:
            needs = loc.get("needs", {})
            total_req = sum(int(v) for v in needs.values())

            # --- STRATEGY-BASED PRIORITY CALCULATION ---
            # Default Welfare = Volume of resources (proxy for people served)
            p_score = total_req

            if strategy == "need":
                # Critical Needs (Medical) get a huge multiplier
                p_score = (int(needs.get("medical", 0)) * 50) + total_req
            elif strategy == "fastest":
                # Flat priority, effectively falls back to minimizing distance only
                p_score = 1

            raw_demands.append(total_req)
            priorities.append(p_score)

        total_demand = sum(raw_demands)

        # Fleet Sizing Logic
        if input_capacity > 0:
            min_needed = math.ceil(total_demand / input_capacity)
        else:
            min_needed = 1

        # Respect Max Fleet Size
        num_vehicles = max(1, input_max_fleet)

        # Calculate Real Capacity Limit for LP
        fleet_cap_real = num_vehicles * input_capacity

        # LP Allocation
        allocated_amounts = solve_allocation_lp(raw_demands, fleet_cap_real, priorities)

        # --- 3. Build VRP Inputs ---
        coords = [[depot["lat"], depot["lon"]]] + [
            [loc["lat"], loc["lon"]] for loc in valid_locations
        ]
        base_dist_matrix, base_time_matrix, base_mode_matrix = get_multimodal_matrix(
            coords
        )

        # Node Splitting (VRP Nodes)
        vrp_demands = [0]
        vrp_map_to_orig = [0]

        for i, amount in enumerate(allocated_amounts):
            orig_idx = i + 1
            remaining = amount
            while remaining > 0:
                chunk = min(remaining, input_capacity)  # STRICT capacity split
                vrp_demands.append(chunk)
                vrp_map_to_orig.append(orig_idx)
                remaining -= chunk

        num_vrp_nodes = len(vrp_demands)

        # VRP Matrices
        vrp_dist_matrix = [[0] * num_vrp_nodes for _ in range(num_vrp_nodes)]
        vrp_time_matrix = [[0] * num_vrp_nodes for _ in range(num_vrp_nodes)]

        for i in range(num_vrp_nodes):
            for j in range(num_vrp_nodes):
                orig_i = vrp_map_to_orig[i]
                orig_j = vrp_map_to_orig[j]
                vrp_dist_matrix[i][j] = base_dist_matrix[orig_i][orig_j]
                vrp_time_matrix[i][j] = base_time_matrix[orig_i][orig_j]

        # --- 4. Solve VRP ---
        manager = pywrapcp.RoutingIndexManager(num_vrp_nodes, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)
        # Forbid depot as intermediate node (only start/end allowed)
        depot_index = manager.NodeToIndex(0)
        routing.SetAllowedVehiclesForIndex(range(num_vehicles), depot_index)
        routing.AddDisjunction([depot_index], 10**9)

        def time_cb(from_idx, to_idx):
            # Include service time
            to_node = manager.IndexToNode(to_idx)
            service = 0
            if to_node != 0:
                service = int(
                    CONSTANTS["fixed_stop_time"]
                    + (vrp_demands[to_node] * CONSTANTS["loading_time_per_kg"])
                )
            return vrp_time_matrix[manager.IndexToNode(from_idx)][to_node] + service

        transit_idx = routing.RegisterTransitCallback(time_cb)

        # --- OBJECTIVE FUNCTION SETUP ---
        # Base cost is travel time
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        routing.AddDimension(
            transit_idx, 86400, CONSTANTS["max_shift_time_sec"] * 3, True, "Time"
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        # --- TRUE WELFARE IMPLEMENTATION (Weighted Latency) ---
        # If strategy is welfare/need, we want to visit high priority nodes EARLIER.
        # We achieve this by adding a cost to the arrival time at specific nodes.
        if strategy in ["welfare", "need"]:
            for i in range(1, num_vrp_nodes):  # Skip depot
                index = manager.NodeToIndex(i)
                orig_idx = vrp_map_to_orig[i] - 1  # Map back to valid_locations index

                # Get the priority score calculated earlier
                priority = priorities[orig_idx]

                # Heuristic Scaling:
                # Time is in seconds (e.g. 3600). Priority is in units (e.g. 1000).
                # Coefficient ensures the trade-off is meaningful.
                # A coefficient of 10 means 1 minute delay cost = 60 * 10 = 600 cost units.
                penalty_coefficient = int(max(1, priority / 10))

                # SetSoftUpperBound(index, bound, coefficient)
                # Setting bound to 0 means EVERY second of arrival time incurs the penalty cost.
                # Objective += (Arrival_Time_i * Penalty_i)
                time_dimension.SetCumulVarSoftUpperBound(index, 0, penalty_coefficient)

                # Also Add Span Cost to ensure equity (don't leave one driver out for 10 hours while others do 1)
                time_dimension.SetGlobalSpanCostCoefficient(100)

        def demand_cb(from_idx):
            node = manager.IndexToNode(from_idx)
            if node == 0:
                return 0  # DEPOT MUST NEVER HAVE DEMAND
            return vrp_demands[node]

        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)

        # STRICT Capacity Constraint
        routing.AddDimensionWithVehicleCapacity(
            demand_idx, 0, [input_capacity] * num_vehicles, True, "Capacity"
        )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_params.time_limit.seconds = int(data.get("time_limit_seconds", 30))

        solution = routing.SolveWithParameters(search_params)

        if not solution:
            return jsonify({"error": "Unable to calculate a valid plan."}), 500

        # --- 5. Extract Solution ---
        final_routes = []
        total_distance = 0
        total_resources = 0

        for v_id in range(num_vehicles):
            index = routing.Start(v_id)
            stops = []
            route_segments = []
            route_dist = 0
            route_load = 0

            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue

            while not routing.IsEnd(index):
                node_vrp = manager.IndexToNode(index)
                next_index = solution.Value(routing.NextVar(index))

                # Geometry
                node_orig = vrp_map_to_orig[node_vrp]
                next_orig = vrp_map_to_orig[manager.IndexToNode(next_index)]

                if not routing.IsEnd(next_index):
                    mode = base_mode_matrix[node_orig][next_orig]
                    geom = get_leg_geometry(coords[node_orig], coords[next_orig], mode)
                    dist = base_dist_matrix[node_orig][next_orig]
                    route_segments.append(
                        {"mode": mode, "geometry": geom, "distance_leg": dist}
                    )
                    route_dist += dist

                if node_vrp != 0 and node_orig != 0:
                    loc = valid_locations[node_orig - 1]
                    load = vrp_demands[node_vrp]
                    stops.append(
                        {
                            "name": loc["name"],
                            "lat": loc["lat"],
                            "lon": loc["lon"],
                            "load": load,
                            "needs": loc["needs"],
                        }
                    )
                    route_load += load

                index = next_index

            if stops:
                mode_counts = {}
                for s in route_segments:
                    mode_counts[s["mode"]] = mode_counts.get(s["mode"], 0) + 1
                p_mode = (
                    max(mode_counts, key=mode_counts.get) if mode_counts else "road"
                )

                final_routes.append(
                    {
                        "vehicle_id": v_id,
                        "vehicle_type": p_mode,
                        "stops": stops,
                        "segments": route_segments,
                        "distance_meters": route_dist,
                        "load": route_load,
                    }
                )
                total_distance += route_dist
                total_resources += route_load

        # ECHO BACK PARAMETERS
        return jsonify(
            {
                "status": "success",
                "depot": depot,
                "locations": valid_locations,
                "routes": final_routes,
                "parameters": {
                    "used_capacity": input_capacity,
                    "used_max_fleet": input_max_fleet,
                    "strategy": strategy,
                },
                "summary": {
                    "title": f"{strategy.capitalize()} Plan",
                    "description": f"Optimized with {len(final_routes)} vehicles (Cap: {input_capacity})",
                    "strategy": strategy,
                    "totalDistanceMeters": total_distance,
                    "totalResources": sum(raw_demands),
                    "assignedResources": total_resources,
                    "totalTrucks": len(final_routes),
                },
            }
        )

    except Exception as e:
        logging.exception("Critical Error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=5000)
