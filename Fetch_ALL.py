import importlib
import json
import asyncio
import pandas as pd
import time
import requests
import USER_PROFILE
# Environment configuration - import first
try:
    import environment_config
    importlib.reload(environment_config)
    from environment_config import ENV_CONFIG, set_environment
    print(f"Environment set to: {ENV_CONFIG.environment}")
    print("To override, use: set_environment('personal') or set_environment('corporate')")
except ImportError:
    print("Warning: environment_config.py not found. Using default settings.")
    ENV_CONFIG = None

# Reload and import other modules
my_module = importlib.import_module('POI_Fetcher')
importlib.reload(my_module)
from POI_Fetcher import *


my_module = importlib.import_module('FLIGHT_Fetcher')
importlib.reload(my_module)
from FLIGHT_Fetcher import *

# Reload and import Route_Fetcher module
my_module = importlib.import_module('Route_Fetcher')
importlib.reload(my_module)
from Route_Fetcher import *

# Load POI data
POI_metadata = json.load(open('POI_list.json'))
POI_list = POI_metadata['POI_LIST']
POI_airports = POI_metadata['location_airports']




# Main execution
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"EXECUTING MAIN POI AND FLIGHT PROCESSING")
    print(f"{'='*60}")
    
    
    # Option 1: Use the modified async_perf function with logging
    # results = async_perf_with_logs(GOOGLE_API_KEY, POI_list, num_pois=None, max_concurrent=10)
    
    # Option 2: Use the direct approach for maximum logging detail
    results=process_poi_with_all_data(POI_list, max_concurrent=8)
    
    POI_data = pd.DataFrame(results)

    # Process POI results
    print(f"\n{'='*50}")
    print(f"PROCESSING POI RESULTS")
    print(f"{'='*50}")

    
    # Route fetching between POI pairs
    print(f"\n{'='*50}")
    print(f"STARTING ASYNC ROUTE FETCHING BETWEEN POI PAIRS")
    print(f"{'='*50}")
    
    # Process routes between POI pairs
    print("\n" + "="*60)
    print("STEP 2: PROCESSING ROUTES BETWEEN POI PAIRS")
    print("="*60)
    
    # Get max distances from user profile
    max_driving_distance = getattr(USER_PROFILE, 'MAX_DRIVING_DISTANCE_KM', 150.0)
    max_walking_distance = getattr(USER_PROFILE, 'MAX_WALKING_DISTANCE_KM', 2.0)
    
    print(f"Using distance limits from USER_PROFILE:")
    print(f"  - Driving: {max_driving_distance}km")
    print(f"  - Walking: {max_walking_distance}km")
    
    route_results = process_routes_with_all_fallback(
        poi_data=POI_data,
        max_driving_distance=max_driving_distance,
        max_walking_distance=max_walking_distance,
        max_concurrent=8,      # Conservative concurrency for corporate environments
        output_prefix="poi_routes"
    )
    
    print(f"\n{'='*50}")
    print(f"ROUTE FETCHING COMPLETED")
    print(f"{'='*50}")
    print(f"Total routes found: {len(route_results) if route_results else 0}")
    
    
    # Flight fetching between locations
    print(f"\n{'='*50}")
    print(f"STARTING FLIGHT FETCHING BETWEEN LOCATIONS")
    print(f"{'='*50}")

        
    # Use the POI_airports data for flight search
    flight_results = process_flights_with_all_fallback(
        location_airports=POI_airports,
        start_date=USER_PROFILE.START_DATE,
        end_date=USER_PROFILE.END_DATE,
        max_concurrent=5,  # Conservative concurrency for flight API
        output_prefix="location_flights"
    )
        
    print(f"\n{'='*50}")
    print(f"FLIGHT FETCHING COMPLETED")
    print(f"{'='*50}")
    
    
    print(f"\nOutput Files Generated:")
    print(f"  - POI data: (from POI_Fetcher)")
    print(f"  - Route data: poi_routes.csv, poi_routes.json, poi_routes_summary.json")
    print(f"  - Flight data: location_flights.csv, location_flights.json, location_flights_summary.json")
    print(f"  - Cheapest flights: location_flights_cheapest.json")
    
    print(f"\n{'='*60}")
    print(f"ALL PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")









