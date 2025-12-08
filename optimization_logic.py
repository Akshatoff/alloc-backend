# optimization_logic.py
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# import googlemaps  <-- No longer needed!
import math  # <-- Added for distance calculations


# --- 1. Get As-the-Crow-Flies Distance (NEW) ---


def get_haversine_distance(coord1, coord2):
    """
    Calculates the straight-line distance between two (lat, lon)
    coordinates on Earth.
    """
    R = 6371  # Earth radius in kilometers
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def create_distance_matrix(locations, depot_lat_lon):
    """
    Uses the Haversine formula to create a distance matrix.
    The cost will be distance in meters (as an integer).
    """
    # Create a list of all coordinates
    coords = [depot_lat_lon] + [(loc["lat"], loc["lon"]) for loc in locations]

    matrix = []
    for i in range(len(coords)):
        row = []
        for j in range(len(coords)):
            if i == j:
                row.append(0)
                continue

            dist = get_haversine_distance(coords[i], coords[j])

            # OR-Tools works best with integers.
            # We will use distance in METERS as the "cost".
            row.append(int(dist * 1000))
        matrix.append(row)

    return matrix


# --- 2. The Main Optimization Function ---
def run_optimization(collected_data):
    """
    This is the core "brain". It takes the collected data and
    uses OR-Tools to find the optimal plan.
    """
    # GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"  <-- No longer needed!

    strategy = collected_data.get("strategy", "fastest")
    depot_info = {"lat": 33.9416, "lon": -118.4085}  # Simulated depot
    locations = collected_data.get("parsedNeeds", {}).get("locations", [])

    if not locations:
        raise ValueError("No locations to plan for.")

    # --- A. Create the Data Model for the Solver ---

    # 1. Get the distance matrix
    # This now uses the Haversine formula and is FREE
    time_matrix = create_distance_matrix(
        locations, (depot_info["lat"], depot_info["lon"])
    )

    # 2. Define demands (what each location needs)
    demands = [0]  # Depot has 0 demand
    for loc in locations:
        total_units = (
            loc["needs"].get("water", 0)
            + loc["needs"].get("food", 0)
            + loc["needs"].get("medical", 0)
        )
        demands.append(total_units)

    # 3. Define vehicle capacities
    vehicle_capacities = [1000, 1000]
    num_vehicles = len(vehicle_capacities)

    # --- B. Setup the OR-Tools Solver ---
    manager = pywrapcp.RoutingIndexManager(len(time_matrix), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    # --- C. Define Constraints & Objectives ---

    # 1. Define the cost: We now use "distance" as the cost
    # The solver just sees a "cost," so we can keep the variable names
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 2. Add the "Demand" (Capacity) Constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        vehicle_capacities,
        True,
        "Capacity",
    )

    # 3. Set the Objective
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # --- D. Solve the Problem ---
    solution = routing.SolveWithParameters(search_parameters)

    # --- E. Format the Output JSON ---
    if not solution:
        raise Exception("No solution found by the optimization engine.")

    # (This part for parsing the solution would still need to be built)
    # ...

    # (Simplified example of what the output JSON would look like)
    final_plan_json = {
        "locations": locations,
        "depot": depot_info,
        "routes": [
            {
                "type": "truck",
                "from": "Main Depot (LAX)",
                "to": locations[0]["name"],
                "time": 0,  # Note: You'd parse solution for *cost* (distance)
                "dist": 0,  # not time
            }
        ],
        "summary": {
            "totalTime": 0,  # This is now "Total Cost/Distance"
            "totalResources": sum(demands),
            "totalTrucks": num_vehicles,
            "totalDrones": 0,
            "strategy": strategy,
            "title": f"Plan: {strategy.title()}",
            "description": "This plan minimizes straight-line distance.",
        },
    }
    return final_plan_json