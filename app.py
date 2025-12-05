"""
Alloc8 v5.0: Fixed - Overflow and Worker Timeout Issues
"""

import json
import logging
import math

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.optimize import linprog

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
CORS(app)

# INT64 max value for OR-Tools
MAX_INT64 = 9223372036854775807
TIME_SCALE_FACTOR = 1000  # Scale down times to prevent overflow

CONSTANTS = {
    "osrm_base_url": "http://router.project-osrm.org",
    "loading_time_per_kg": 0.002,  # Scaled down from 2.0
    "fixed_stop_time": 900,
    "max_driver_dist_km": 5000,
    "max_shift_time_sec": 86400,
    "vehicle_capacity": 5000,
    "speed_mps_road": 13.0,
    "speed_mps_boat": 8.5,
    "speed_mps_air": 220.0,
    "air_threshold_km": 600,
    "air_docking_time": 3600,
}


def get_haversine_distance(coord1, coord2):
    """Simple great-circle distance in meters"""
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
    """Builds distance/time/mode matrix with overflow protection"""
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

                # Air travel for long distances
                if geo_dist > (CONSTANTS["air_threshold_km"] * 1000):
                    dist_matrix[i][j] = geo_dist
                    # Scale down time and cap it
                    raw_time = int(
                        (geo_dist / CONSTANTS["speed_mps_air"])
                        + CONSTANTS["air_docking_time"]
                    )
                    time_matrix[i][j] = min(
                        raw_time // TIME_SCALE_FACTOR, MAX_INT64 // 10
                    )
                    mode_matrix[i][j] = "air"
                    continue

                osrm_dist = raw_dists[i][j]
                osrm_time = raw_times[i][j]

                if osrm_dist is None or osrm_dist > (geo_dist * 3.0):
                    # Use boat
                    dist_matrix[i][j] = geo_dist
                    raw_time = int(geo_dist / CONSTANTS["speed_mps_boat"])
                    time_matrix[i][j] = min(
                        raw_time // TIME_SCALE_FACTOR, MAX_INT64 // 10
                    )
                    mode_matrix[i][j] = "boat"
                else:
                    # Use road
                    dist_matrix[i][j] = int(osrm_dist)
                    # Scale down time
                    time_matrix[i][j] = min(
                        int(osrm_time) // TIME_SCALE_FACTOR, MAX_INT64 // 10
                    )
                    mode_matrix[i][j] = "road"

    except Exception as e:
        logging.warning(f"OSRM Error ({e}). Using fallback.")
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d_m = get_haversine_distance(coords[i], coords[j])
                dist_matrix[i][j] = int(d_m)

                if d_m > (CONSTANTS["air_threshold_km"] * 1000):
                    raw_time = int(
                        (d_m / CONSTANTS["speed_mps_air"])
                        + CONSTANTS["air_docking_time"]
                    )
                    time_matrix[i][j] = min(
                        raw_time // TIME_SCALE_FACTOR, MAX_INT64 // 10
                    )
                    mode_matrix[i][j] = "air"
                else:
                    raw_time = int(d_m / CONSTANTS["speed_mps_road"])
                    time_matrix[i][j] = min(
                        raw_time // TIME_SCALE_FACTOR, MAX_INT64 // 10
                    )
                    mode_matrix[i][j] = "road"

    return dist_matrix, time_matrix, mode_matrix


def get_leg_geometry(coord_start, coord_end, mode):
    """Returns geometry for route visualization"""
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
    """LP allocation with equity constraints"""
    n = len(demands)
    if n == 0 or sum(demands) <= fleet_cap:
        return demands

    c = [-p for p in priorities]
    A_ub, b_ub = [[1] * n], [fleet_cap]

    bounds = []
    for d in demands:
        if d == 0:
            bounds.append((0, 0))
        else:
            bounds.append((int(d * 0.20), int(d)))

    if sum(b[0] for b in bounds) > fleet_cap:
        return [int(d * (fleet_cap / sum(demands))) for d in demands]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return [int(x) for x in res.x] if res.success else demands


@app.route("/generate-plan", methods=["POST"])
def generate_plan():
    try:
        data = request.get_json(force=True)

        strategy = data.get("strategy", "welfare")
        parsed_needs = data.get("parsedNeeds", {})
        locations = parsed_needs.get("locations", [])

        if not locations:
            return jsonify({"error": "No locations provided"}), 400

        # Limit locations to prevent timeout (max 20 for free tier)
        if len(locations) > 20:
            logging.warning(f"Too many locations ({len(locations)}), limiting to 20")
            locations = locations[:20]

        depot_data = data.get("depot", {})
        if depot_data and "lat" in depot_data:
            depot = depot_data
        else:
            depot = {
                "lat": locations[0]["lat"],
                "lon": locations[0]["lon"],
                "name": "Main Distribution Center",
            }

        # Calculate demands
        raw_demands, priorities = [], []
        for loc in locations:
            needs = loc.get("needs", {})
            total_req = sum(int(v) for v in needs.values())

            if strategy == "need":
                p_score = (
                    (int(needs.get("medical", 0)) * 10)
                    + (int(needs.get("water", 0)) * 3)
                    + int(needs.get("food", 0))
                )
            elif strategy == "fastest":
                p_score = 5
            else:
                p_score = total_req

            raw_demands.append(total_req)
            priorities.append(p_score)

        vehicle_capacity = int(
            data.get("vehicle_capacity", CONSTANTS["vehicle_capacity"])
        )
        total_demand = sum(raw_demands)

        if vehicle_capacity > 0:
            estimated_trucks_needed = math.ceil(total_demand / vehicle_capacity)
        else:
            estimated_trucks_needed = 1

        max_fleet_limit = 50  # Reduced for free tier
        requested_max = data.get("max_fleet_size")

        if requested_max:
            max_fleet_size = min(int(requested_max), max_fleet_limit)
        else:
            max_fleet_size = min(estimated_trucks_needed + 3, max_fleet_limit)

        num_vehicles = min(max(estimated_trucks_needed, 2), max_fleet_size)

        logging.info(f"Total demand: {total_demand}, Vehicles: {num_vehicles}")

        # Allocation
        fleet_cap_total = num_vehicles * vehicle_capacity
        allocated_amounts = solve_allocation_lp(
            raw_demands, fleet_cap_total, priorities
        )
        demands = [0] + allocated_amounts

        logging.info(f"Allocated: {allocated_amounts[:5]}... (showing first 5)")

        # Build matrices
        coords = [[depot["lat"], depot["lon"]]] + [
            [loc["lat"], loc["lon"]] for loc in locations
        ]
        n = len(coords)

        dist_matrix, time_matrix, mode_matrix = get_multimodal_matrix(coords)

        # Service times (scaled down)
        service_times = [0] * n
        for i in range(1, n):
            service_times[i] = min(
                int(
                    CONSTANTS["fixed_stop_time"] // TIME_SCALE_FACTOR
                    + (allocated_amounts[i - 1] * CONSTANTS["loading_time_per_kg"])
                ),
                MAX_INT64 // 100,
            )

        # Final time matrix with overflow protection
        final_time_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Cap values to prevent overflow
                final_time_matrix[i][j] = min(
                    int(time_matrix[i][j] + service_times[j]), MAX_INT64 // 100
                )

        # Verify no overflow values
        max_time = max(max(row) for row in final_time_matrix)
        logging.info(f"Max time value: {max_time} (safe limit: {MAX_INT64 // 100})")

        # OR-Tools setup
        manager = pywrapcp.RoutingIndexManager(n, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def time_cb(from_idx, to_idx):
            try:
                from_node = manager.IndexToNode(from_idx)
                to_node = manager.IndexToNode(to_idx)
                return final_time_matrix[from_node][to_node]
            except Exception as e:
                logging.error(f"Time callback error: {e}, from={from_idx}, to={to_idx}")
                return 1000  # Safe fallback

        transit_idx = routing.RegisterTransitCallback(time_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # Reduced limits for free tier
        routing.AddDimension(
            transit_idx,
            3600,  # Reduced slack
            CONSTANTS["max_shift_time_sec"] // TIME_SCALE_FACTOR,
            True,
            "Time",
        )

        def dist_cb(from_idx, to_idx):
            try:
                from_node = manager.IndexToNode(from_idx)
                to_node = manager.IndexToNode(to_idx)
                return dist_matrix[from_node][to_node]
            except Exception as e:
                logging.error(f"Distance callback error: {e}")
                return 1000

        dist_idx = routing.RegisterTransitCallback(dist_cb)
        routing.AddDimension(dist_idx, 0, 10000000, True, "Distance")

        def demand_cb(from_idx):
            try:
                node = manager.IndexToNode(from_idx)
                return demands[node]
            except Exception as e:
                logging.error(f"Demand callback error: {e}")
                return 0

        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx, 0, [vehicle_capacity] * num_vehicles, True, "Capacity"
        )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        # Reduced time limit for free tier
        search_params.time_limit.seconds = min(
            int(data.get("time_limit_seconds", 10)), 20
        )

        penalty = 100000
        for i in range(1, n):
            routing.AddDisjunction([manager.NodeToIndex(i)], penalty)

        logging.info("Starting OR-Tools solver...")
        solution = routing.SolveWithParameters(search_params)

        if not solution:
            logging.error("OR-Tools failed to find solution")
            return jsonify({"error": "Unable to calculate a valid plan."}), 500

        logging.info("Solution found, building routes...")

        # Format output
        routes = []
        total_distance = 0
        total_resources = 0

        for v_id in range(num_vehicles):
            index = routing.Start(v_id)
            stops = []
            route_segments = []
            route_dist = 0
            route_load = 0

            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                next_index = solution.Value(routing.NextVar(index))
                next_node = manager.IndexToNode(next_index)

                if not routing.IsEnd(next_index):
                    travel_mode = mode_matrix[node][next_node]
                    geometry = get_leg_geometry(
                        coords[node], coords[next_node], travel_mode
                    )

                    route_segments.append(
                        {
                            "from_node": node,
                            "to_node": next_node,
                            "mode": travel_mode,
                            "geometry": geometry,
                            "distance_leg": dist_matrix[node][next_node],
                        }
                    )

                if node != 0:
                    stops.append(
                        {
                            "node_index": node,
                            "name": locations[node - 1]["name"],
                            "lat": locations[node - 1]["lat"],
                            "lon": locations[node - 1]["lon"],
                            "load": demands[node],
                            "needs": locations[node - 1]["needs"],
                        }
                    )
                    route_load += demands[node]

                route_dist += routing.GetArcCostForVehicle(index, next_index, v_id)
                index = next_index

            if len(stops) > 0:
                mode_counts = {"road": 0, "boat": 0, "air": 0}
                for seg in route_segments:
                    mode_counts[seg["mode"]] += 1
                primary_mode = (
                    max(mode_counts, key=mode_counts.get) if mode_counts else "road"
                )

                routes.append(
                    {
                        "vehicle_id": v_id,
                        "vehicle_type": primary_mode,
                        "stops": stops,
                        "segments": route_segments,
                        "distance_meters": route_dist,
                        "load": route_load,
                    }
                )
                total_distance += route_dist
                total_resources += route_load

        logging.info(f"Generated {len(routes)} routes successfully")

        return jsonify(
            {
                "status": "success",
                "depot": depot,
                "locations": locations,
                "routes": routes,
                "summary": {
                    "title": f"{strategy.capitalize()} Multimodal Plan",
                    "description": f"Optimized using {len(routes)} vehicles (Road/Sea/Air).",
                    "strategy": strategy.capitalize(),
                    "totalDistanceMeters": total_distance,
                    "totalResources": sum(raw_demands),
                    "assignedResources": total_resources,
                    "totalTrucks": len(routes),
                },
            }
        )

    except Exception as e:
        logging.exception("Critical Error")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
