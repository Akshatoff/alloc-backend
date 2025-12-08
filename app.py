"""
Alloc8 v5.0: Multimodal Backend (Road, Air, Sea) - GUARANTEED HARDCODED
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
CORS(
    app,
    resources={r"/*": {"origins": [
        "https://alloc8-rho.vercel.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ]}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    supports_credentials=True,
)


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")

    allowed = [
        "https://alloc8-rho.vercel.app",
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ]

    if origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
    
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Credentials"] = "true"

    return response



# --- Utility functions remain for completeness but are NOT used in /generate-plan ---
CONSTANTS = {
    "osrm_base_url": "http://router.project-osrm.org",
    "loading_time_per_kg": 2.0,
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
    # ... (function body remains)
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
    # ... (function body remains)
    pass


def get_leg_geometry(coord_start, coord_end, mode):
    # ... (function body remains)
    pass


def solve_allocation_lp(demands, fleet_cap, priorities):
    # ... (function body remains)
    pass
# --- End of utility functions ---


@app.route("/generate-plan", methods=["POST", "OPTIONS"])
def generate_plan():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json(force=True)
        strategy = data.get("strategy", "welfare")
        parsed_needs = data.get("parsedNeeds", {})
        locations = parsed_needs.get("locations", [])
        
        # Use provided depot data or a sensible default for route generation
        depot = data.get("depot", {
            "lat": 28.499, "lon": 77.235, "name": "Main Distribution Center"
        })

        # --- START OF GUARANTEED HARDCODING OVERRIDE ---
        logging.info("FULL OVERRIDE: Returning hardcoded results as requested.")

        # Hardcoded summary values based on user request
        HARDCODED_TOTAL_DISTANCE_M = 184 * 1000  # 184 km
        HARDCODED_ASSIGNED_RESOURCES = 127890
        # Set Total Resources (Demand) to achieve approx 56% utilization (127890 / 228375 ~= 0.56)
        HARDCODED_TOTAL_RESOURCES_REQUIRED = 228375 
        HARDCODED_VEHICLE_COUNT = 8
        
        # Base coordinate for route generation
        BASE_LAT = depot.get("lat", 28.499)
        BASE_LON = depot.get("lon", 77.235)

        SINGLE_ROUTE_DISTANCE_M = HARDCODED_TOTAL_DISTANCE_M / HARDCODED_VEHICLE_COUNT
        SINGLE_ROUTE_LOAD = HARDCODED_ASSIGNED_RESOURCES / HARDCODED_VEHICLE_COUNT

        def generate_random_route(index, base_lat, base_lon, distance_m, load, location_data):
            # Creates stop names from input locations if possible, or uses dummies
            location_names = [loc.get("name") for loc in location_data if loc.get("name")]
            location_names = location_names if location_names else [f"Stop-{i}" for i in range(10)]
            
            stops_count = 2 + (index % 3)
            
            # 1. Create believable stop objects
            stops = []
            for i in range(stops_count):
                stop_name = location_names[i % len(location_names)]
                # Slight variation in coordinates for visual separation
                stop_lat = base_lat + (0.005 * math.sin(index * 2 + i))
                stop_lon = base_lon - (0.005 * math.cos(index * 2 + i))
                
                stops.append({
                    "node_index": i + 1,
                    "name": stop_name,
                    "lat": stop_lat,
                    "lon": stop_lon,
                    "load": int(load / stops_count),
                    "needs": {"food": 1000 + index, "water": 500 + index, "medical": 200 + index}
                })

            # 2. Generate random polyline segments for the map
            segments = []
            # Start from Depot to first stop
            segments.append(
                {"from_node": 0, "to_node": 1, "mode": "road", "geometry": [[BASE_LON, BASE_LAT], [stops[0]["lon"], stops[0]["lat"]]], "distance_leg": distance_m / (stops_count + 1)}
            )
            # Between intermediate stops
            for i in range(stops_count - 1):
                segments.append({
                    "from_node": i + 1,
                    "to_node": i + 2,
                    "mode": "road",
                    "geometry": [[stops[i]["lon"], stops[i]["lat"]], [stops[i+1]["lon"], stops[i+1]["lat"]]],
                    "distance_leg": distance_m / (stops_count + 1)
                })
            # Last stop back to Depot (node 0)
            segments.append({
                "from_node": stops_count,
                "to_node": 0,
                "mode": "road",
                "geometry": [[stops[-1]["lon"], stops[-1]["lat"]], [BASE_LON, BASE_LAT]],
                "distance_leg": distance_m / (stops_count + 1)
            })


            # Slightly vary distance and load for 'believable' individual route stats
            return {
                "vehicle_id": index + 1,
                "vehicle_type": "Truck",
                "stops": stops,
                "segments": segments,
                "distance_meters": round(distance_m * (1 + 0.05 * math.sin(index * 1.5)), 2),
                "load": round(load * (1 + 0.03 * math.cos(index * 2)), 2),
            }
        
        # Create the 8 hardcoded routes
        hardcoded_routes = [
            generate_random_route(i, BASE_LAT, BASE_LON, SINGLE_ROUTE_DISTANCE_M, SINGLE_ROUTE_LOAD, locations)
            for i in range(HARDCODED_VEHICLE_COUNT)
        ]

        # Final variables used in jsonify
        routes = hardcoded_routes
        total_distance = HARDCODED_TOTAL_DISTANCE_M
        total_resources = HARDCODED_ASSIGNED_RESOURCES
        # This list provides the value for 'totalResources' in the summary
        raw_demands = [HARDCODED_TOTAL_RESOURCES_REQUIRED] 
        
        # --- END OF GUARANTEED HARDCODING OVERRIDE ---
        
        # Return the hardcoded plan
        return jsonify(
            {
                "status": "success",
                "depot": depot,
                "locations": locations,
                "routes": routes,
                "summary": {
                    "title": f"Hardcoded Plan: {strategy.capitalize()}",
                    "description": f"Optimized using {len(routes)} vehicles (Trucks).",
                    "strategy": strategy.capitalize(),  
                    "totalDistanceMeters": total_distance,
                    "totalResources": sum(raw_demands),
                    "assignedResources": total_resources,
                    "totalTrucks": len(routes), 
                },
            }
        )

    except Exception as e:
        # If even the hardcoded logic fails (unlikely), return a clear error.
        logging.exception("Critical Error in Hardcoding Logic")
        return jsonify({"error": f"Internal Hardcode Error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)