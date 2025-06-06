import requests
import asyncio
import aiohttp
import json
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import time
from itertools import combinations
import math

# Import environment configuration
try:
    from environment_config import ENV_CONFIG, get_requests_kwargs, get_aiohttp_kwargs
    print(f"Route Fetcher - Environment configuration loaded: {ENV_CONFIG.environment}")
except ImportError:
    print("Warning: environment_config.py not found. Using default settings for Route Fetcher.")
    ENV_CONFIG = None
    
    def get_requests_kwargs():
        return {'timeout': (10, 30), 'verify': True}
    
    def get_aiohttp_kwargs():
        import aiohttp
        return {'timeout': aiohttp.ClientTimeout(total=60), 'trust_env': False}

# Import API keys directly
try:
    from keys import GOOGLE_API_KEY
    print("Google Maps API key loaded successfully")
except ImportError:
    print("Warning: keys.py not found. Please ensure GOOGLE_API_KEY is available.")
    GOOGLE_API_KEY = None


class RouteDistanceFetcher:
    """
    Class to fetch transportation routes and distances between POI pairs using Google Distance Matrix API
    Supports driving and walking transportation modes
    """
    
    def __init__(self, api_key: str = None, debug: bool = True):
        # Use imported API key if not provided
        self.api_key = api_key or GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Google Maps API key is required")
            
        self.debug = debug
        self.base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        
        # Get environment-specific settings
        if ENV_CONFIG:
            self.rate_limit_settings = ENV_CONFIG.rate_limit_settings
            self.is_corporate = ENV_CONFIG.is_corporate
        else:
            self.rate_limit_settings = {
                'min_request_interval': 0.5,
                'max_concurrent': 5,
                'retry_attempts': 3,
                'backoff_factor': 1.5
            }
            self.is_corporate = False
        
        # Transportation modes supported
        self.transport_modes = ['driving', 'walking']
        
        # Default max distances for different transport modes
        self.default_max_distances = {
            'driving': 150.0,  # 150km for driving
            'walking': 2.0     # 2km for walking
        }
        
        # Maximum elements per request (10x10 = 100 to avoid API limits)
        self.max_elements_per_request = 100
        
        # Maximum origins/destinations per request (10 each for 10x10 matrix)
        self.max_locations_per_request = 10
        
        # Set up session configuration based on environment
        self._setup_session_config()
    
    def _setup_session_config(self):
        """Setup session configuration based on environment"""
        # Timeout configuration
        if self.is_corporate:
            self.connect_timeout = 30
            self.total_timeout = 120
            self.read_timeout = 90
        else:
            self.connect_timeout = 10
            self.total_timeout = 60
            self.read_timeout = 30
        
        # Connector configuration for corporate environments
        if self.is_corporate and ENV_CONFIG:
            import ssl
            import os
            
            # Create SSL context for corporate environment
            self.ssl_context = ssl.create_default_context()
            if ENV_CONFIG.cert_path and os.path.exists(ENV_CONFIG.cert_path):
                self.ssl_context.load_verify_locations(ENV_CONFIG.cert_path)
            
            # Trust environment for proxy settings
            self.trust_env = True
        else:
            self.ssl_context = None
            self.trust_env = False
    
    def create_session_kwargs(self):
        """Create aiohttp session kwargs based on environment"""
        timeout = aiohttp.ClientTimeout(
            total=self.total_timeout,
            connect=self.connect_timeout
        )
        
        kwargs = {
            'timeout': timeout,
            'trust_env': self.trust_env
        }
        
        # Add SSL context for corporate environments
        if self.ssl_context:
            kwargs['connector'] = aiohttp.TCPConnector(ssl=self.ssl_context)
        
        return kwargs
    
    def log(self, message: str):
        """Simple logging for debugging"""
        if self.debug:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] RouteDistanceFetcher: {message}")
    
    def calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth in kilometers
        Using Haversine formula - used for filtering out very distant POI pairs
        """
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
    
    def filter_poi_pairs_by_distance(self, poi_data: List[dict], transport_mode: str = 'driving', max_distance_km: float = None) -> List[Tuple[dict, dict]]:
        """
        Filter POI pairs to only include those within reasonable distance for the specified transport mode
        This helps avoid unnecessary API calls for very distant locations
        """
        if max_distance_km is None:
            max_distance_km = self.default_max_distances.get(transport_mode, 50.0)
            
        valid_pairs = []
        
        for poi1, poi2 in combinations(poi_data, 2):
            # Skip if either POI doesn't have coordinates
            if not all([poi1.get('latitude'), poi1.get('longitude'), 
                       poi2.get('latitude'), poi2.get('longitude')]):
                continue
            
            distance = self.calculate_haversine_distance(
                poi1['latitude'], poi1['longitude'],
                poi2['latitude'], poi2['longitude']
            )
            
            if distance <= max_distance_km:
                valid_pairs.append((poi1, poi2))
            else:
                self.log(f"Skipping distant pair for {transport_mode}: {poi1['searched_name']} <-> {poi2['searched_name']} ({distance:.1f}km > {max_distance_km}km)")
        
        return valid_pairs
    
    def create_location_string(self, poi: dict) -> str:
        """Create location string for Google API from POI data"""
        # Use coordinates if available (more accurate)
        if poi.get('latitude') and poi.get('longitude'):
            return f"{poi['latitude']},{poi['longitude']}"
        
        # Fallback to address
        if poi.get('address'):
            return poi['address']
        
        # Last resort: use name and location
        return f"{poi.get('actual_name', poi.get('searched_name', ''))}, {poi.get('searched_location', '')}"
    
    def fetch_distance_matrix_sync(self, origins: List[dict], destinations: List[dict], mode: str = 'driving') -> dict:
        """
        Fetch distance matrix for given origins and destinations using specified transport mode
        """
        if mode not in self.transport_modes:
            raise ValueError(f"Transport mode must be one of: {self.transport_modes}")
        
        # Create location strings
        origin_strings = [self.create_location_string(poi) for poi in origins]
        destination_strings = [self.create_location_string(poi) for poi in destinations]
        
        self.log(f"Fetching {mode} distances: {len(origins)} origins -> {len(destinations)} destinations")
        
        params = {
            'origins': '|'.join(origin_strings),
            'destinations': '|'.join(destination_strings),
            'mode': mode,
            'units': 'metric',
            'key': self.api_key
        }
        
        try:
            if ENV_CONFIG:
                response = requests.get(self.base_url, params=params, **ENV_CONFIG.get_requests_kwargs())
            else:
                response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('status') == 'OK':
                    self.log(f"✓ Successfully fetched {mode} distance matrix")
                    return result
                else:
                    self.log(f"✗ API returned status: {result.get('status')}")
                    return {'status': 'ERROR', 'error': result.get('status')}
            else:
                self.log(f"✗ HTTP Error: {response.status_code}")
                return {'status': 'HTTP_ERROR', 'code': response.status_code}
                
        except Exception as e:
            self.log(f"✗ Exception: {str(e)}")
            return {'status': 'EXCEPTION', 'error': str(e)}
    
    async def fetch_distance_matrix_async(self, session: aiohttp.ClientSession, origins: List[dict], destinations: List[dict], mode: str = 'driving') -> dict:
        """
        Async version of distance matrix fetching
        """
        if mode not in self.transport_modes:
            raise ValueError(f"Transport mode must be one of: {self.transport_modes}")
        
        # Create location strings
        origin_strings = [self.create_location_string(poi) for poi in origins]
        destination_strings = [self.create_location_string(poi) for poi in destinations]
        
        self.log(f"Async fetching {mode} distances: {len(origins)} origins -> {len(destinations)} destinations")
        
        params = {
            'origins': '|'.join(origin_strings),
            'destinations': '|'.join(destination_strings),
            'mode': mode,
            'units': 'metric',
            'key': self.api_key
        }
        
        try:
            if session.closed:
                self.log(f"✗ Session closed for distance matrix call")
                return {'status': 'SESSION_CLOSED'}
            
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get('status') == 'OK':
                        self.log(f"✓ Successfully fetched {mode} distance matrix (async)")
                        return result
                    else:
                        self.log(f"✗ API returned status: {result.get('status')}")
                        return {'status': 'ERROR', 'error': result.get('status')}
                else:
                    self.log(f"✗ HTTP Error: {response.status}")
                    return {'status': 'HTTP_ERROR', 'code': response.status}
                    
        except asyncio.TimeoutError:
            self.log(f"✗ Timeout error for {mode} distance matrix")
            return {'status': 'TIMEOUT'}
        except Exception as e:
            self.log(f"✗ Exception: {str(e)}")
            return {'status': 'EXCEPTION', 'error': str(e)}
    
    def process_distance_matrix_result(self, result: dict, origins: List[dict], destinations: List[dict], mode: str) -> List[dict]:
        """
        Process the distance matrix API result into structured data
        """
        processed_routes = []
        
        if result.get('status') != 'OK':
            self.log(f"Cannot process result with status: {result.get('status')}")
            return processed_routes
        
        rows = result.get('rows', [])
        
        for i, row in enumerate(rows):
            if i >= len(origins):
                break
                
            origin_poi = origins[i]
            elements = row.get('elements', [])
            
            for j, element in enumerate(elements):
                if j >= len(destinations):
                    break
                    
                destination_poi = destinations[j]
                
                # Skip self-to-self routes
                if origin_poi.get('place_id') == destination_poi.get('place_id'):
                    continue
                
                route_data = {
                    'origin_poi_id': origin_poi.get('place_id', ''),
                    'origin_name': origin_poi.get('searched_name', ''),
                    'origin_actual_name': origin_poi.get('actual_name', ''),
                    'origin_location': origin_poi.get('searched_location', ''),
                    'origin_coordinates': f"{origin_poi.get('latitude', 0)},{origin_poi.get('longitude', 0)}",
                    
                    'destination_poi_id': destination_poi.get('place_id', ''),
                    'destination_name': destination_poi.get('searched_name', ''),
                    'destination_actual_name': destination_poi.get('actual_name', ''),
                    'destination_location': destination_poi.get('searched_location', ''),
                    'destination_coordinates': f"{destination_poi.get('latitude', 0)},{destination_poi.get('longitude', 0)}",
                    
                    'transport_mode': mode,
                    'status': element.get('status', 'UNKNOWN'),
                }
                
                if element.get('status') == 'OK':
                    distance = element.get('distance', {})
                    duration = element.get('duration', {})
                    
                    route_data.update({
                        'distance_meters': distance.get('value', 0),
                        'distance_text': distance.get('text', ''),
                        'duration_seconds': duration.get('value', 0),
                        'duration_text': duration.get('text', ''),
                        'duration_minutes': round(duration.get('value', 0) / 60, 1),
                        'distance_km': round(distance.get('value', 0) / 1000, 2),
                    })
                else:
                    route_data.update({
                        'distance_meters': 0,
                        'distance_text': 'N/A',
                        'duration_seconds': 0,
                        'duration_text': 'N/A',
                        'duration_minutes': 0,
                        'distance_km': 0,
                        'error_reason': element.get('status', 'UNKNOWN_ERROR')
                    })
                
                processed_routes.append(route_data)
        
        return processed_routes
    
    async def fetch_routes_for_location_async(self, session: aiohttp.ClientSession, location_pois: List[dict], 
                                            location: str, max_driving_distance: float = None, 
                                            max_walking_distance: float = None, max_concurrent: int = 3) -> List[dict]:
        """
        Async version of fetch_routes_for_location with proper API limit handling
        """
        self.log(f"\nAsync fetching routes for location: {location}")
        
        if len(location_pois) < 2:
            self.log(f"Not enough POIs in {location} for route calculation (found {len(location_pois)})")
            return []
        
        all_routes = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Set max distances for this location
        max_distances = {
            'driving': max_driving_distance or self.default_max_distances['driving'],
            'walking': max_walking_distance or self.default_max_distances['walking']
        }
        
        # Process each transport mode with its specific distance filter
        for mode in self.transport_modes:
            self.log(f"\n--- Processing {mode.upper()} routes for {location} (max distance: {max_distances[mode]}km) ---")
            
            # Filter POI pairs by distance for this transport mode
            valid_pairs = self.filter_poi_pairs_by_distance(location_pois, mode, max_distances[mode])
            
            if not valid_pairs:
                self.log(f"No valid POI pairs found for {mode} within {max_distances[mode]}km")
                continue
            
            self.log(f"Processing {len(valid_pairs)} POI pairs for {mode} within {max_distances[mode]}km (async)")
            
            async def fetch_single_pair_routes(pair: Tuple[dict, dict]):
                """Fetch routes for a single POI pair"""
                async with semaphore:
                    origin, destination = pair
                    
                    try:
                        # Fetch distance matrix for this single pair
                        result = await self.fetch_distance_matrix_async(session, [origin], [destination], mode)
                        
                        # Process the result
                        routes = self.process_distance_matrix_result(result, [origin], [destination], mode)
                        return routes
                        
                    except Exception as e:
                        self.log(f"Error fetching {mode} route for {origin.get('searched_name')} -> {destination.get('searched_name')}: {e}")
                        return []
            
            # Process pairs in smaller concurrent batches to avoid overwhelming the API
            batch_size = min(max_concurrent * 2, 5)  # Process 5 pairs at a time max
            
            for batch_start in range(0, len(valid_pairs), batch_size):
                batch_pairs = valid_pairs[batch_start:batch_start + batch_size]
                
                self.log(f"Processing {mode} batch {batch_start//batch_size + 1}/{(len(valid_pairs)-1)//batch_size + 1}: {len(batch_pairs)} pairs")
                
                # Create tasks for this batch
                tasks = [fetch_single_pair_routes(pair) for pair in batch_pairs]
                
                # Execute batch with controlled concurrency
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results
                for result in batch_results:
                    if isinstance(result, Exception):
                        self.log(f"Error in batch processing: {result}")
                    else:
                        all_routes.extend(result)
                
                # Delay between batches to respect rate limits
                if batch_start + batch_size < len(valid_pairs):
                    delay = self.rate_limit_settings['min_request_interval'] * 2  # Extra delay for async
                    self.log(f"Waiting {delay}s before next batch...")
                    await asyncio.sleep(delay)
        
        self.log(f"Completed async route fetching for {location}: {len(all_routes)} routes")
        return all_routes
    
    async def fetch_all_routes_async_robust(self, poi_data: List[dict], max_driving_distance: float = None,
                                          max_walking_distance: float = None, max_concurrent: int = 3) -> List[dict]:
        """
        Async version of fetch_all_routes with robust session management
        """
        self.log(f"\n{'='*60}")
        self.log(f"STARTING ASYNC ROUTE FETCHING FOR ALL LOCATIONS")
        self.log(f"{'='*60}")
        
        # Adjust concurrency for corporate environments
        if self.is_corporate:
            max_concurrent = min(max_concurrent, 2)
        
        # Group POIs by location
        locations = {}
        for poi in poi_data:
            location = poi.get('searched_location', 'Unknown')
            if location not in locations:
                locations[location] = []
            locations[location].append(poi)
        
        self.log(f"Found {len(locations)} locations: {list(locations.keys())}")
        self.log(f"Distance limits: Driving {max_driving_distance or self.default_max_distances['driving']}km, Walking {max_walking_distance or self.default_max_distances['walking']}km")
        
        all_routes = []
        
        session_kwargs = self.create_session_kwargs()
        
        try:
            async with aiohttp.ClientSession(**session_kwargs) as session:
                for location, location_pois in locations.items():
                    if len(location_pois) >= 2:
                        location_routes = await self.fetch_routes_for_location_async(
                            session, location_pois, location, max_driving_distance, 
                            max_walking_distance, max_concurrent
                        )
                        all_routes.extend(location_routes)
                    else:
                        self.log(f"Skipping {location}: only {len(location_pois)} POI(s)")
        
        except Exception as e:
            self.log(f"Async route fetching error: {e}")
            # Fallback to sync processing for remaining locations
            self.log("Falling back to sync processing...")
            return self.fetch_all_routes(poi_data, max_driving_distance, max_walking_distance)
        
        self.log(f"\n{'='*60}")
        self.log(f"ASYNC ROUTE FETCHING COMPLETED")
        self.log(f"{'='*60}")
        self.log(f"Total routes fetched: {len(all_routes)}")
        
        # Summary by transport mode
        for mode in self.transport_modes:
            mode_routes = [r for r in all_routes if r.get('transport_mode') == mode]
            successful_routes = [r for r in mode_routes if r.get('status') == 'OK']
            self.log(f"{mode.capitalize()} routes: {len(successful_routes)}/{len(mode_routes)} successful")
        
        return all_routes
    
    def fetch_all_routes(self, poi_data: List[dict], max_driving_distance: float = None, 
                        max_walking_distance: float = None) -> List[dict]:
        """
        Fetch routes for all locations in the POI data (sync fallback)
        """
        self.log(f"\n{'='*60}")
        self.log(f"STARTING ROUTE FETCHING FOR ALL LOCATIONS (SYNC)")
        self.log(f"{'='*60}")
        
        # Group POIs by location
        locations = {}
        for poi in poi_data:
            location = poi.get('searched_location', 'Unknown')
            if location not in locations:
                locations[location] = []
            locations[location].append(poi)
        
        self.log(f"Found {len(locations)} locations: {list(locations.keys())}")
        self.log(f"Distance limits: Driving {max_driving_distance or self.default_max_distances['driving']}km, Walking {max_walking_distance or self.default_max_distances['walking']}km")
        
        all_routes = []
        
        for location, location_pois in locations.items():
            if len(location_pois) >= 2:
                location_routes = self.fetch_routes_for_location(location_pois, location, 
                                                                max_driving_distance, max_walking_distance)
                all_routes.extend(location_routes)
            else:
                self.log(f"Skipping {location}: only {len(location_pois)} POI(s)")
        
        self.log(f"\n{'='*60}")
        self.log(f"ROUTE FETCHING COMPLETED")
        self.log(f"{'='*60}")
        self.log(f"Total routes fetched: {len(all_routes)}")
        
        return all_routes
    
    def fetch_routes_for_location(self, location_pois: List[dict], location: str, 
                                 max_driving_distance: float = None, max_walking_distance: float = None) -> List[dict]:
        """
        Fetch all routes between POI pairs within a specific location (sync)
        Uses proper 10x10 matrix batching for efficient API usage
        """
        self.log(f"\n{'='*60}")
        self.log(f"Fetching routes for location: {location}")
        self.log(f"{'='*60}")
        
        if len(location_pois) < 2:
            self.log(f"Not enough POIs in {location} for route calculation (found {len(location_pois)})")
            return []
        
        self.log(f"Found {len(location_pois)} POIs in {location}")
        
        all_routes = []
        
        # Set max distances for this location
        max_distances = {
            'driving': max_driving_distance or self.default_max_distances['driving'],
            'walking': max_walking_distance or self.default_max_distances['walking']
        }
        
        # Process each transport mode with its specific distance filter
        for mode in self.transport_modes:
            self.log(f"\n--- Processing {mode.upper()} routes (max distance: {max_distances[mode]}km) ---")
            
            # Filter POI pairs by distance for this transport mode
            valid_pairs = self.filter_poi_pairs_by_distance(location_pois, mode, max_distances[mode])
            
            if not valid_pairs:
                self.log(f"No valid POI pairs found for {mode} within {max_distances[mode]}km")
                continue
            
            self.log(f"Processing {len(valid_pairs)} POI pairs for {mode} within {max_distances[mode]}km")
            
            # Process in batches of 10x10
            # For 10x10 matrix, we need to group origins and destinations efficiently
            # We'll use a sliding window approach to maximize matrix usage
            
            all_pois = location_pois
            num_pois = len(all_pois)
            
            # Process in blocks of max 10 origins x 10 destinations
            for origin_start in range(0, num_pois, self.max_locations_per_request):
                origin_end = min(origin_start + self.max_locations_per_request, num_pois)
                origins = all_pois[origin_start:origin_end]
                
                for dest_start in range(0, num_pois, self.max_locations_per_request):
                    dest_end = min(dest_start + self.max_locations_per_request, num_pois)
                    destinations = all_pois[dest_start:dest_end]
                    
                    # Check if any valid pairs exist in this block
                    block_has_valid_pairs = False
                    for origin in origins:
                        for destination in destinations:
                            if origin != destination and (origin, destination) in valid_pairs or (destination, origin) in valid_pairs:
                                block_has_valid_pairs = True
                                break
                        if block_has_valid_pairs:
                            break
                    
                    if not block_has_valid_pairs:
                        continue
                    
                    self.log(f"Processing {mode} matrix: {len(origins)} origins x {len(destinations)} destinations")
                    
                    # Fetch distance matrix for this block
                    result = self.fetch_distance_matrix_sync(origins, destinations, mode)
                    
                    # Process results, but only keep the valid pairs
                    if result.get('status') == 'OK':
                        rows = result.get('rows', [])
                        
                        for i, row in enumerate(rows):
                            if i >= len(origins):
                                break
                                
                            origin_poi = origins[i]
                            elements = row.get('elements', [])
                            
                            for j, element in enumerate(elements):
                                if j >= len(destinations):
                                    break
                                    
                                destination_poi = destinations[j]
                                
                                # Skip self-to-self routes
                                if origin_poi.get('place_id') == destination_poi.get('place_id'):
                                    continue
                                
                                # Check if this pair is in our valid pairs list
                                pair_is_valid = (origin_poi, destination_poi) in valid_pairs or \
                                              (destination_poi, origin_poi) in valid_pairs
                                
                                if pair_is_valid:
                                    # Process this valid route
                                    route_data = {
                                        'origin_poi_id': origin_poi.get('place_id', ''),
                                        'origin_name': origin_poi.get('searched_name', ''),
                                        'origin_actual_name': origin_poi.get('actual_name', ''),
                                        'origin_location': origin_poi.get('searched_location', ''),
                                        'origin_coordinates': f"{origin_poi.get('latitude', 0)},{origin_poi.get('longitude', 0)}",
                                        
                                        'destination_poi_id': destination_poi.get('place_id', ''),
                                        'destination_name': destination_poi.get('searched_name', ''),
                                        'destination_actual_name': destination_poi.get('actual_name', ''),
                                        'destination_location': destination_poi.get('searched_location', ''),
                                        'destination_coordinates': f"{destination_poi.get('latitude', 0)},{destination_poi.get('longitude', 0)}",
                                        
                                        'transport_mode': mode,
                                        'status': element.get('status', 'UNKNOWN'),
                                    }
                                    
                                    if element.get('status') == 'OK':
                                        distance = element.get('distance', {})
                                        duration = element.get('duration', {})
                                        
                                        route_data.update({
                                            'distance_meters': distance.get('value', 0),
                                            'distance_text': distance.get('text', ''),
                                            'duration_seconds': duration.get('value', 0),
                                            'duration_text': duration.get('text', ''),
                                            'duration_minutes': round(duration.get('value', 0) / 60, 1),
                                            'distance_km': round(distance.get('value', 0) / 1000, 2),
                                        })
                                    else:
                                        route_data.update({
                                            'distance_meters': 0,
                                            'distance_text': 'N/A',
                                            'duration_seconds': 0,
                                            'duration_text': 'N/A',
                                            'duration_minutes': 0,
                                            'distance_km': 0,
                                            'error_reason': element.get('status', 'UNKNOWN_ERROR')
                                        })
                                    
                                    all_routes.append(route_data)
                    
                    # Add delay between matrix requests to respect rate limits
                    delay = self.rate_limit_settings['min_request_interval']
                    self.log(f"Waiting {delay}s before next request...")
                    time.sleep(delay)
        
        self.log(f"\nCompleted route fetching for {location}: {len(all_routes)} routes")
        return all_routes
    
    def fetch_all_routes_with_fallback(self, poi_data: List[dict], max_driving_distance: float = None,
                                     max_walking_distance: float = None, max_concurrent: int = 3) -> List[dict]:
        """
        Public method with comprehensive fallback logic
        1. Try full async processing
        2. Fall back to sync processing if async fails completely
        """
        self.log(f"Starting comprehensive route fetch")
        self.log(f"Environment: {'Corporate' if self.is_corporate else 'Personal'}")
        self.log(f"Max concurrent: {max_concurrent}")
        self.log(f"Distance limits: Driving {max_driving_distance or self.default_max_distances['driving']}km, Walking {max_walking_distance or self.default_max_distances['walking']}km")
        
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                self.log("Event loop detected, using sync processing")
                return self.fetch_all_routes(poi_data, max_driving_distance, max_walking_distance)
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self.fetch_all_routes_async_robust(poi_data, max_driving_distance, 
                                                                     max_walking_distance, max_concurrent))
            
        except Exception as e:
            self.log(f"Async processing failed: {e}")
            self.log("Falling back to complete synchronous processing...")
            
            # Complete fallback to sync processing
            return self.fetch_all_routes(poi_data, max_driving_distance, max_walking_distance)
    
    def save_routes(self, routes: List[dict], filename_prefix: str = "poi_routes") -> pd.DataFrame:
        """
        Save route data to CSV and JSON files
        """
        if not routes:
            self.log("No routes to save")
            return None
        
        df = pd.DataFrame(routes)
        
        # Save to CSV
        csv_filename = f"{filename_prefix}.csv"
        df.to_csv(csv_filename, index=False)
        self.log(f"Saved routes to CSV: {csv_filename}")
        
        # Save to JSON (newline-delimited for consistency with POI fetcher)
        json_filename = f"{filename_prefix}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            for route in routes:
                json.dump(route, f, ensure_ascii=False)
                f.write('\n')
        self.log(f"Saved routes to JSON: {json_filename}")
        
        # Create summary statistics
        self.create_route_summary(df, filename_prefix)
        
        return df
    
    def create_route_summary(self, df: pd.DataFrame, filename_prefix: str):
        """
        Create and save summary statistics for the routes
        """
        summary = {}
        
        # Overall statistics
        summary['total_routes'] = len(df)
        summary['successful_routes'] = len(df[df['status'] == 'OK'])
        summary['locations'] = df['origin_location'].nunique()
        
        # Statistics by transport mode
        for mode in self.transport_modes:
            mode_df = df[df['transport_mode'] == mode]
            successful_mode_df = mode_df[mode_df['status'] == 'OK']
            
            if len(successful_mode_df) > 0:
                summary[f'{mode}_routes'] = {
                    'total': len(mode_df),
                    'successful': len(successful_mode_df),
                    'avg_distance_km': round(successful_mode_df['distance_km'].mean(), 2),
                    'avg_duration_minutes': round(successful_mode_df['duration_minutes'].mean(), 1),
                    'max_distance_km': round(successful_mode_df['distance_km'].max(), 2),
                    'max_duration_minutes': round(successful_mode_df['duration_minutes'].max(), 1),
                    'min_distance_km': round(successful_mode_df['distance_km'].min(), 2),
                    'min_duration_minutes': round(successful_mode_df['duration_minutes'].min(), 1)
                }
        
        # Statistics by location
        location_stats = {}
        for location in df['origin_location'].unique():
            location_df = df[df['origin_location'] == location]
            successful_location_df = location_df[location_df['status'] == 'OK']
            
            location_stats[location] = {
                'total_routes': len(location_df),
                'successful_routes': len(successful_location_df),
                'unique_pois': len(set(location_df['origin_poi_id'].tolist() + location_df['destination_poi_id'].tolist()))
            }
        
        summary['by_location'] = location_stats
        
        # Save summary
        summary_filename = f"{filename_prefix}_summary.json"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.log(f"Saved route summary to: {summary_filename}")
        
        # Print summary to console
        self.log(f"\n{'='*40}")
        self.log(f"ROUTE SUMMARY")
        self.log(f"{'='*40}")
        self.log(f"Total routes: {summary['total_routes']}")
        self.log(f"Successful routes: {summary['successful_routes']}")
        self.log(f"Locations processed: {summary['locations']}")
        
        for mode in self.transport_modes:
            if f'{mode}_routes' in summary:
                mode_stats = summary[f'{mode}_routes']
                self.log(f"\n{mode.capitalize()} routes:")
                self.log(f"  Successful: {mode_stats['successful']}/{mode_stats['total']}")
                self.log(f"  Avg distance: {mode_stats['avg_distance_km']} km")
                self.log(f"  Avg duration: {mode_stats['avg_duration_minutes']} min")


# Convenience function for easy integration
def create_route_fetcher(api_key: str = None, debug: bool = True):
    """
    Factory function to create the route fetcher
    """
    return RouteDistanceFetcher(api_key, debug)


# Wrapper function for easy integration
def process_routes_with_all_fallback(poi_data: Union[List[dict], pd.DataFrame], 
                                    max_driving_distance: float = None,
                                    max_walking_distance: float = None,
                                    max_concurrent: int = 3,
                                    output_prefix: str = "poi_routes"):
    """
    High-level function to process routes between POI pairs with comprehensive fallback
    
    Args:
        poi_data: List of POI dictionaries or DataFrame
        max_driving_distance: Maximum distance for driving routes (defaults to 150km)
        max_walking_distance: Maximum distance for walking routes (defaults to 2km)
        max_concurrent: Maximum concurrent requests
        output_prefix: Prefix for output files
    
    Returns:
        List of processed route results
    """
    
    # Convert DataFrame to list of dicts if needed
    if isinstance(poi_data, pd.DataFrame):
        poi_list = poi_data.to_dict('records')
    else:
        poi_list = poi_data
    
    # Create the route fetcher
    fetcher = create_route_fetcher(debug=True)
    
    # Process all routes with comprehensive fallback
    routes = fetcher.fetch_all_routes_with_fallback(poi_list, max_driving_distance, 
                                                  max_walking_distance, max_concurrent)
    
    # Save results to files
    if routes and output_prefix:
        fetcher.save_routes(routes, output_prefix)
    
    return routes


# Async wrapper for Jupyter notebooks
async def process_routes_async(poi_data: Union[List[dict], pd.DataFrame], 
                             max_driving_distance: float = None,
                             max_walking_distance: float = None,
                             max_concurrent: int = 3,
                             output_prefix: str = "poi_routes_async"):
    """
    Async wrapper function for processing routes in Jupyter notebooks
    
    Args:
        poi_data: List of POI dictionaries or DataFrame
        max_driving_distance: Maximum distance for driving routes (defaults to 150km)
        max_walking_distance: Maximum distance for walking routes (defaults to 2km)
        max_concurrent: Maximum concurrent requests
        output_prefix: Prefix for output files
    
    Returns:
        List of processed route results
    """
    
    # Convert DataFrame to list of dicts if needed
    if isinstance(poi_data, pd.DataFrame):
        poi_list = poi_data.to_dict('records')
    else:
        poi_list = poi_data
    
    # Create the route fetcher
    fetcher = create_route_fetcher(debug=True)
    
    # Process all routes asynchronously
    routes = await fetcher.fetch_all_routes_async_robust(poi_list, max_driving_distance,
                                                       max_walking_distance, max_concurrent)
    
    # Save results to files
    if routes and output_prefix:
        fetcher.save_routes(routes, output_prefix)
    
    return routes


# Test function
def test_route_fetcher(poi_data: List[dict], test_single_route: bool = True):
    """
    Test the route fetcher with sample data
    """
    print(f"\n{'='*60}")
    print(f"TESTING ROUTE FETCHER")
    print(f"{'='*60}")
    
    fetcher = create_route_fetcher(debug=True)
    
    # Test with a subset of data
    test_data = poi_data[:10] if len(poi_data) > 10 else poi_data
    
    if test_single_route and len(test_data) >= 2:
        # Test single route
        print("\n1. Testing single route:")
        origin = test_data[0]
        destination = test_data[1]
        
        print(f"   Origin: {origin.get('searched_name', 'Unknown')}")
        print(f"   Destination: {destination.get('searched_name', 'Unknown')}")
        
        for mode in ['driving', 'walking']:
            result = fetcher.fetch_distance_matrix_sync([origin], [destination], mode)
            routes = fetcher.process_distance_matrix_result(result, [origin], [destination], mode)
            
            if routes and routes[0].get('status') == 'OK':
                route = routes[0]
                print(f"   {mode.capitalize()}: {route.get('distance_text')} in {route.get('duration_text')}")
            else:
                print(f"   {mode.capitalize()}: Failed")
    
    # Test batch processing
    print("\n2. Testing batch processing with new distance limits:")
    print(f"   Driving max distance: {fetcher.default_max_distances['driving']}km")
    print(f"   Walking max distance: {fetcher.default_max_distances['walking']}km")
    
    routes = fetcher.fetch_all_routes_with_fallback(test_data)
    
    if routes:
        df = fetcher.save_routes(routes, "test_routes")
        print(f"\nTest completed. Found {len(routes)} routes.")
        return df
    else:
        print("No routes found in test")
        return None