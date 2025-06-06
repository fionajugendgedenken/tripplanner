import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
from itertools import combinations
from typing import List, Dict, Tuple, Optional, Union
import json
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from time import sleep
import random
from collections import defaultdict

# Import environment configuration
try:
    from environment_config import ENV_CONFIG, get_requests_kwargs, get_aiohttp_kwargs
    print(f"Flight Fetcher - Environment configuration loaded: {ENV_CONFIG.environment}")
except ImportError:
    print("Warning: environment_config.py not found. Using default settings for Flight Fetcher.")
    ENV_CONFIG = None
    
    def get_requests_kwargs():
        return {'timeout': (10, 30), 'verify': True}
    
    def get_aiohttp_kwargs():
        import aiohttp
        return {'timeout': aiohttp.ClientTimeout(total=60), 'trust_env': False}

# Import API keys directly
try:
    from keys import AMADEUS_API_KEY, AMADEUS_API_SECRET
    print("Amadeus API credentials loaded successfully")
except ImportError:
    print("Warning: keys.py not found. Please ensure AMADEUS_API_KEY and AMADEUS_API_SECRET are available.")
    AMADEUS_API_KEY = None
    AMADEUS_API_SECRET = None


@dataclass
class FlightInfo:
    """Data class to store flight information"""
    origin: str
    destination: str
    departure_date: str
    departure_time: str
    arrival_date: str
    arrival_time: str
    duration: str
    price: float
    currency: str
    airline: str
    flight_number: str


class AmadeusFlightSearch:
    """
    A class to search for flights between multiple airports using Amadeus API
    with concurrent requests for improved performance.
    Environment-aware version that adapts to corporate/personal environments.
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None, debug: bool = True):
        """
        Initialize the AmadeusFlightSearch with API credentials.
        
        Args:
            api_key: Amadeus API key (optional, will use imported key)
            api_secret: Amadeus API secret (optional, will use imported secret)
            debug: Enable debug logging
        """
        # Use imported API credentials if not provided
        self.api_key = api_key or AMADEUS_API_KEY
        self.api_secret = api_secret or AMADEUS_API_SECRET
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Amadeus API credentials are required")
        
        self.debug = debug
        
        # Environment-aware configuration
        if ENV_CONFIG:
            self.rate_limit_settings = ENV_CONFIG.rate_limit_settings
            self.is_corporate = ENV_CONFIG.is_corporate
            self.environment = ENV_CONFIG.environment
        else:
            # Fallback configuration
            self.rate_limit_settings = {
                'min_request_interval': 0.5,
                'max_concurrent': 5,
                'retry_attempts': 3,
                'backoff_factor': 1.5
            }
            self.is_corporate = False
            self.environment = "unknown"
        
        self.token = None
        self.token_expiry = None
        self.base_url = "https://test.api.amadeus.com"
        
        # Thread locks for sync operations
        self._token_lock = threading.Lock()
        self._request_lock = threading.Lock()
        
        # Async locks for async operations (will be initialized when needed)
        self._async_token_lock = None
        self._async_request_lock = None
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = self.rate_limit_settings['min_request_interval']
        self._request_counter = 0
        self._rate_limit_hits = 0
        
        # Set up session configuration
        self._setup_session_config()
        
        self.log(f"AmadeusFlightSearch initialized for {self.environment} environment")
        self.log(f"Rate limiting: {self._min_request_interval}s between requests")
    
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
            print(f"[{datetime.now().strftime('%H:%M:%S')}] AmadeusFlightSearch: {message}")
        
    def _get_access_token(self) -> str:
        """
        Get access token from Amadeus API. Refreshes if expired.
        Thread-safe implementation with environment-aware requests.
        
        Returns:
            Access token string
        """
        with self._token_lock:
            # Check if token is still valid
            if self.token and self.token_expiry and datetime.now() < self.token_expiry:
                return self.token
                
            # Get new token
            token_url = f"{self.base_url}/v1/security/oauth2/token"
            
            data = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.api_secret
            }
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            # Use environment-aware requests
            if ENV_CONFIG:
                resp = requests.post(token_url, data=data, headers=headers, **ENV_CONFIG.get_requests_kwargs())
            else:
                resp = requests.post(token_url, data=data, headers=headers, timeout=30)
            
            resp.raise_for_status()
            
            response_data = resp.json()
            self.token = response_data.get("access_token")
            expires_in = response_data.get("expires_in", 1799)  # Default 30 minutes
            
            # Set expiry time with 5-minute buffer
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)
            
            self.log(f"New access token obtained. Expires at: {self.token_expiry}")
            return self.token
    
    async def _get_access_token_async(self, session: aiohttp.ClientSession) -> str:
        """
        Async version of getting access token - uses asyncio.Lock
        """
        await self._ensure_async_locks()
        
        async with self._async_token_lock:
            # Check if token is still valid
            if self.token and self.token_expiry and datetime.now() < self.token_expiry:
                return self.token
                
            # Get new token
            token_url = f"{self.base_url}/v1/security/oauth2/token"
            
            data = {
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.api_secret
            }
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            try:
                if session.closed:
                    return None
                
                async with session.post(token_url, data=data, headers=headers) as resp:
                    if resp.status == 200:
                        response_data = await resp.json()
                        self.token = response_data.get("access_token")
                        expires_in = response_data.get("expires_in", 1799)
                        
                        # Set expiry time with 5-minute buffer
                        self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)
                        
                        self.log(f"New access token obtained (async). Expires at: {self.token_expiry}")
                        return self.token
                    else:
                        return None
            except Exception as e:
                self.log(f"Error getting token: {e}")
                return None
    
    def _apply_rate_limit(self):
        """Apply rate limiting with adaptive delay"""
        with self._request_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            # Adaptive delay: increase interval if we're hitting rate limits
            adaptive_interval = self._min_request_interval
            if self._rate_limit_hits > 0:
                # Add extra delay for each rate limit hit
                adaptive_interval += (self._rate_limit_hits * 0.1)
            
            if time_since_last < adaptive_interval:
                sleep_time = adaptive_interval - time_since_last
                sleep(sleep_time)
            
            self._last_request_time = time.time()
            self._request_counter += 1
    
    async def _ensure_async_locks(self):
        """Initialize async locks if not already done"""
        if self._async_token_lock is None:
            self._async_token_lock = asyncio.Lock()
        if self._async_request_lock is None:
            self._async_request_lock = asyncio.Lock()
    
    async def _apply_rate_limit_async(self):
        """Apply rate limiting with adaptive delay (async) - uses asyncio.Lock"""
        await self._ensure_async_locks()
        
        async with self._async_request_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            # Adaptive delay: increase interval if we're hitting rate limits
            adaptive_interval = self._min_request_interval
            if self._rate_limit_hits > 0:
                adaptive_interval += (self._rate_limit_hits * 0.1)
            
            if time_since_last < adaptive_interval:
                sleep_time = adaptive_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self._last_request_time = time.time()
            self._request_counter += 1
    
    def _search_flights_for_route(self, origin_code: str, dest_code: str, 
                                  departure_date: str, retry_count: int = None) -> List[FlightInfo]:
        """
        Search flights for specific airport codes and date with retry logic (sync).
        """
        # Use environment-specific retry count if not provided
        if retry_count is None:
            retry_count = self.rate_limit_settings.get('retry_attempts', 3)
        
        # Get location names for the airports
        origin_location = getattr(self, '_airport_to_location', {}).get(origin_code, origin_code)
        dest_location = getattr(self, '_airport_to_location', {}).get(dest_code, dest_code)
        
        for attempt in range(retry_count):
            try:
                # Apply rate limiting
                self._apply_rate_limit()
                
                # Get token
                token = self._get_access_token()
                
                # Prepare API request
                url = f"{self.base_url}/v2/shopping/flight-offers"
                
                headers = {
                    "Authorization": f"Bearer {token}"
                }
                
                params = {
                    "originLocationCode": origin_code,
                    "destinationLocationCode": dest_code,
                    "departureDate": departure_date,
                    "adults": 1,
                    "nonStop": "false",
                    "max": 5  # Limit results
                }
                
                # Use environment-aware requests
                if ENV_CONFIG:
                    resp = requests.get(url, headers=headers, params=params, **ENV_CONFIG.get_requests_kwargs())
                else:
                    resp = requests.get(url, headers=headers, params=params, timeout=30)
                
                if resp.status_code == 429:  # Too Many Requests
                    with self._request_lock:
                        self._rate_limit_hits += 1
                    
                    if attempt < retry_count - 1:
                        backoff_factor = self.rate_limit_settings.get('backoff_factor', 2.0)
                        wait_time = (backoff_factor ** attempt) + random.uniform(0, 1)
                        self.log(f"Rate limited. Waiting {wait_time:.2f}s before retry {attempt + 1}/{retry_count}")
                        sleep(wait_time)
                        continue
                    else:
                        self.log(f"Rate limit exceeded for {origin_code} to {dest_code} on {departure_date}")
                        return []
                
                resp.raise_for_status()
                
                data = resp.json()
                flights = []
                
                for offer in data.get("data", []):
                    # Extract flight information
                    for itinerary in offer.get("itineraries", []):
                        segments = itinerary.get("segments", [])
                        if not segments:
                            continue
                        
                        # Get first and last segment for multi-leg flights
                        first_segment = segments[0]
                        last_segment = segments[-1]
                        
                        flight_info = FlightInfo(
                            origin=f"{origin_location} ({origin_code})",
                            destination=f"{dest_location} ({dest_code})",
                            departure_date=first_segment.get("departure", {}).get("at", "")[:10],
                            departure_time=first_segment.get("departure", {}).get("at", "")[11:16],
                            arrival_date=last_segment.get("arrival", {}).get("at", "")[:10],
                            arrival_time=last_segment.get("arrival", {}).get("at", "")[11:16],
                            duration=itinerary.get("duration", ""),
                            price=float(offer.get("price", {}).get("total", 0)),
                            currency=offer.get("price", {}).get("currency", "EUR"),
                            airline=first_segment.get("carrierCode", ""),
                            flight_number=f"{first_segment.get('carrierCode', '')}{first_segment.get('number', '')}"
                        )
                        flights.append(flight_info)
                
                return flights
                
            except requests.exceptions.RequestException as e:
                if attempt < retry_count - 1:
                    backoff_factor = self.rate_limit_settings.get('backoff_factor', 2.0)
                    wait_time = (backoff_factor ** attempt) + random.uniform(0, 1)
                    self.log(f"Error on attempt {attempt + 1}/{retry_count}: {e}. Retrying in {wait_time:.2f}s...")
                    sleep(wait_time)
                else:
                    self.log(f"Error searching flights from {origin_code} to {dest_code}: {e}")
                    return []
        
        return []
    
    async def _search_flights_for_route_async(self, session: aiohttp.ClientSession, origin_code: str, 
                                            dest_code: str, departure_date: str, retry_count: int = None) -> List[FlightInfo]:
        """
        Search flights for specific airport codes and date with retry logic (async).
        """
        # Use environment-specific retry count if not provided
        if retry_count is None:
            retry_count = self.rate_limit_settings.get('retry_attempts', 3)
        
        # Get location names for the airports
        origin_location = getattr(self, '_airport_to_location', {}).get(origin_code, origin_code)
        dest_location = getattr(self, '_airport_to_location', {}).get(dest_code, dest_code)
        
        for attempt in range(retry_count):
            try:
                # Apply rate limiting
                await self._apply_rate_limit_async()
                
                # Get token (with improved logging inside the method)
                token = await self._get_access_token_async(session)
                if not token:
                    return []
                
                # Prepare API request
                url = f"{self.base_url}/v2/shopping/flight-offers"
                
                headers = {
                    "Authorization": f"Bearer {token}"
                }
                
                params = {
                    "originLocationCode": origin_code,
                    "destinationLocationCode": dest_code,
                    "departureDate": departure_date,
                    "adults": 1,
                    "nonStop": "false",
                    "max": 5  # Limit results
                }
                
                if session.closed:
                    return []
                
                async with session.get(url, headers=headers, params=params) as resp:
                    
                    if resp.status == 429:  # Too Many Requests
                        async with self._async_request_lock:
                            self._rate_limit_hits += 1
                        
                        if attempt < retry_count - 1:
                            backoff_factor = self.rate_limit_settings.get('backoff_factor', 2.0)
                            wait_time = (backoff_factor ** attempt) + random.uniform(0, 1)
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            return []
                    
                    if resp.status == 200:
                        data = await resp.json()
                        flights = []
                        
                        for offer in data.get("data", []):
                            # Extract flight information
                            for itinerary in offer.get("itineraries", []):
                                segments = itinerary.get("segments", [])
                                if not segments:
                                    continue
                                
                                # Get first and last segment for multi-leg flights
                                first_segment = segments[0]
                                last_segment = segments[-1]
                                
                                flight_info = FlightInfo(
                                    origin=f"{origin_location} ({origin_code})",
                                    destination=f"{dest_location} ({dest_code})",
                                    departure_date=first_segment.get("departure", {}).get("at", "")[:10],
                                    departure_time=first_segment.get("departure", {}).get("at", "")[11:16],
                                    arrival_date=last_segment.get("arrival", {}).get("at", "")[:10],
                                    arrival_time=last_segment.get("arrival", {}).get("at", "")[11:16],
                                    duration=itinerary.get("duration", ""),
                                    price=float(offer.get("price", {}).get("total", 0)),
                                    currency=offer.get("price", {}).get("currency", "EUR"),
                                    airline=first_segment.get("carrierCode", ""),
                                    flight_number=f"{first_segment.get('carrierCode', '')}{first_segment.get('number', '')}"
                                )
                                flights.append(flight_info)
                        
                        return flights
                    else:
                        # Only log non-200 responses that aren't retryable
                        if resp.status >= 500:
                            self.log(f"API error {resp.status} for {origin_code} -> {dest_code}")
                        return []
                        
            except asyncio.TimeoutError:
                if attempt < retry_count - 1:
                    await asyncio.sleep(2)
                else:
                    self.log(f"Timeout after {retry_count} attempts: {origin_code} -> {dest_code}")
            except Exception as e:
                if attempt < retry_count - 1:
                    backoff_factor = self.rate_limit_settings.get('backoff_factor', 2.0)
                    wait_time = (backoff_factor ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                else:
                    self.log(f"Final error for {origin_code} -> {dest_code}: {e}")
                    return []
        
        return []
    
    async def search_flights_async_robust(self, location_airports: Dict[str, List[str]], 
                                        start_date: str, end_date: str, 
                                        max_concurrent: int = 3) -> Dict[str, List[FlightInfo]]:
        """
        Asynchronously search for flights with robust session management
        """
        # Adjust concurrency for corporate environments
        if self.is_corporate:
            max_concurrent = min(max_concurrent, 2)
        
        # Create reverse mapping for airport to location
        self._airport_to_location = {}
        total_airports = 0
        for location, airports in location_airports.items():
            for airport in airports:
                self._airport_to_location[airport] = location
                total_airports += 1
        
        self.log(f"Starting flight search for {len(location_airports)} locations with {total_airports} airports...")
        self.log(f"Environment: {self.environment}")
        self.log(f"Max concurrent: {max_concurrent}")
        
        # Generate valid airport pairs (excluding same-location pairs)
        airport_pairs = []
        locations = list(location_airports.keys())
        
        for i, origin_loc in enumerate(locations):
            for j, dest_loc in enumerate(locations):
                if i == j:  # Skip same location pairs only
                    continue
                
                # Get all airports for each location
                origin_airports = location_airports[origin_loc]
                dest_airports = location_airports[dest_loc]
                
                # Create pairs between airports of different locations
                for origin_airport in origin_airports:
                    for dest_airport in dest_airports:
                        airport_pairs.append((origin_airport, dest_airport))
        
        self.log(f"Generated {len(airport_pairs)} valid airport pairs")
        
        # Generate date range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        date_list = []
        current = start
        while current <= end:
            date_list.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        self.log(f"Date range: {len(date_list)} days")
        
        # Create all search tasks
        search_tasks = []
        for origin, destination in airport_pairs:
            for date in date_list:
                search_tasks.append((origin, destination, date))
        
        self.log(f"Total searches to perform: {len(search_tasks)}")
        
        # Process searches
        results = defaultdict(list)
        failed_searches = []
        
        session_kwargs = self.create_session_kwargs()
        
        try:
            async with aiohttp.ClientSession(**session_kwargs) as session:
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def search_with_semaphore(task):
                    async with semaphore:
                        origin_code, dest_code, date = task
                        
                        try:
                            flights = await self._search_flights_for_route_async(session, origin_code, dest_code, date)
                            
                            if flights:
                                origin_loc = self._airport_to_location.get(origin_code, origin_code)
                                dest_loc = self._airport_to_location.get(dest_code, dest_code)
                                route_key = f"{origin_loc} -> {dest_loc}"
                                return route_key, flights
                            else:
                                return None, task
                        except Exception as e:
                            self.log(f"Search error {origin_code} -> {dest_code}: {e}")
                            return None, task
                
                # Process in batches to avoid overwhelming the system
                batch_size = max_concurrent * 10
                total_batches = (len(search_tasks) - 1) // batch_size + 1
                
                for batch_num, batch_start in enumerate(range(0, len(search_tasks), batch_size), 1):
                    batch = search_tasks[batch_start:batch_start + batch_size]
                    
                    self.log(f"Processing batch {batch_num}/{total_batches} ({len(batch)} searches)")
                    
                    batch_tasks = [search_with_semaphore(task) for task in batch]
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    successful_in_batch = 0
                    failed_in_batch = 0
                    
                    for result in batch_results:
                        if isinstance(result, Exception):
                            self.log(f"Batch error: {result}")
                            failed_in_batch += 1
                        elif result[0]:  # Successful search
                            route_key, flights = result
                            results[route_key].extend(flights)
                            successful_in_batch += 1
                        else:  # Failed search
                            _, task = result
                            failed_searches.append(task)
                            failed_in_batch += 1
                    
                    # Progress reporting
                    completed_searches = batch_start + len(batch)
                    overall_progress = (completed_searches / len(search_tasks)) * 100
                    
                    self.log(f"Batch {batch_num} complete: {successful_in_batch} successful, {failed_in_batch} failed")
                    self.log(f"Overall progress: {completed_searches}/{len(search_tasks)} ({overall_progress:.1f}%)")
                    
                    # Cool-down between batches
                    if batch_start + batch_size < len(search_tasks) and self._rate_limit_hits > 0:
                        cooldown = min(5, self._rate_limit_hits * 0.5)
                        self.log(f"Cooling down for {cooldown:.1f}s between batches...")
                        await asyncio.sleep(cooldown)
                        # Reset rate limit counter for next batch
                        async with self._async_request_lock:
                            self._rate_limit_hits = max(0, self._rate_limit_hits - 5)
        
        except Exception as e:
            self.log(f"Async flight search error: {e}")
            # Return what we have so far
        
        # Report statistics
        self.log(f"\nSearch Statistics:")
        self.log(f"- Environment: {self.environment}")
        self.log(f"- Total requests made: {self._request_counter}")
        self.log(f"- Rate limit hits: {self._rate_limit_hits}")
        self.log(f"- Failed searches: {len(failed_searches)}")
        
        if failed_searches:
            self.log(f"\n{len(failed_searches)} searches failed. Consider re-running with lower concurrency")
        
        return dict(results)
    
    def search_flights_sync(self, location_airports: Dict[str, List[str]], 
                           start_date: str, end_date: str) -> Dict[str, List[FlightInfo]]:
        """
        Synchronous version of flight search (fallback)
        """
        # Create reverse mapping for airport to location
        self._airport_to_location = {}
        total_airports = 0
        for location, airports in location_airports.items():
            for airport in airports:
                self._airport_to_location[airport] = location
                total_airports += 1
        
        self.log(f"Starting sync flight search for {len(location_airports)} locations...")
        
        # Generate valid airport pairs
        airport_pairs = []
        locations = list(location_airports.keys())
        
        for i, origin_loc in enumerate(locations):
            for j, dest_loc in enumerate(locations):
                if i == j:  # Skip same location pairs only
                    continue
                
                origin_airports = location_airports[origin_loc]
                dest_airports = location_airports[dest_loc]
                
                for origin_airport in origin_airports:
                    for dest_airport in dest_airports:
                        airport_pairs.append((origin_airport, dest_airport))
        
        self.log(f"Generated {len(airport_pairs)} valid airport pairs")
        
        # Generate date range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        date_list = []
        current = start
        while current <= end:
            date_list.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        # Process searches
        results = defaultdict(list)
        total_searches = len(airport_pairs) * len(date_list)
        completed = 0
        successful = 0
        
        self.log(f"Total searches to perform: {total_searches}")
        
        for origin_code, dest_code in airport_pairs:
            for date in date_list:
                flights = self._search_flights_for_route(origin_code, dest_code, date)
                
                if flights:
                    origin_loc = self._airport_to_location.get(origin_code, origin_code)
                    dest_loc = self._airport_to_location.get(dest_code, dest_code)
                    route_key = f"{origin_loc} -> {dest_loc}"
                    results[route_key].extend(flights)
                    successful += 1
                
                completed += 1
                
                # Progress reporting with percentages
                if completed % 10 == 0 or completed == total_searches:
                    progress_pct = (completed / total_searches) * 100
                    success_rate = (successful / completed) * 100 if completed > 0 else 0
                    self.log(f"Progress: {completed}/{total_searches} ({progress_pct:.1f}%) - {successful} successful ({success_rate:.1f}%)")
                
                # Rate limiting between requests
                time.sleep(self.rate_limit_settings['min_request_interval'])
        
        return dict(results)
    
    def search_flights_with_fallback(self, location_airports: Dict[str, List[str]], 
                                   start_date: str, end_date: str, 
                                   max_concurrent: int = 3) -> Dict[str, List[FlightInfo]]:
        """
        Search flights with async-first logic - quits on async failure for debugging
        """
        self.log(f"Starting async-first flight search")
        self.log(f"Environment: {'Corporate' if self.is_corporate else 'Personal'}")
        
        try:
            # Check if we're in an event loop (like Jupyter)
            try:
                loop = asyncio.get_running_loop()
                self.log("Event loop detected - attempting async execution in Jupyter")
                
                # Try using nest_asyncio for Jupyter compatibility
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    self.log("nest_asyncio applied - enabling async in Jupyter")
                    
                    # Now we can use asyncio.run even in Jupyter
                    return asyncio.run(self.search_flights_async_robust(
                        location_airports, start_date, end_date, max_concurrent
                    ))
                    
                except ImportError as e:
                    self.log("❌ nest_asyncio not available - install with: pip install nest-asyncio")
                    
                    # Original sync fallback (commented out for debugging)
                    # self.log("Falling back to sync processing")
                    # return self.search_flights_sync(location_airports, start_date, end_date)
                    
                    # New: Quit for debugging
                    self.log("❌ ASYNC FAILED: Quitting for debugging (no sync fallback)")
                    raise ImportError("nest_asyncio required for async execution in Jupyter. Install with: pip install nest-asyncio") from e
                    
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                self.log("No event loop detected - using asyncio.run() for async execution")
                return asyncio.run(self.search_flights_async_robust(
                    location_airports, start_date, end_date, max_concurrent
                ))
            
        except Exception as e:
            self.log(f"❌ ASYNC EXECUTION FAILED: {e}")
            
            # Original sync fallback (commented out for debugging)
            # self.log("Final fallback to synchronous processing...")
            # return self.search_flights_sync(location_airports, start_date, end_date)
            
            # New: Quit for debugging
            self.log("❌ QUITTING FOR DEBUGGING (no sync fallback)")
            self.log(f"❌ Exception type: {type(e).__name__}")
            import traceback
            self.log("❌ Full traceback:")
            traceback.print_exc()
            
            # Quit instead of falling back to sync
            raise RuntimeError(f"Async flight search failed: {e}") from e
    
    async def search_flights_async_direct(self, location_airports: Dict[str, List[str]], 
                                        start_date: str, end_date: str, 
                                        max_concurrent: int = 3) -> Dict[str, List[FlightInfo]]:
        """
        Direct async method for use in async contexts (like Jupyter with await)
        """
        return await self.search_flights_async_robust(location_airports, start_date, end_date, max_concurrent)
    
    def save_results(self, results: Dict[str, List[FlightInfo]], 
                    output_prefix: str = "flight_results") -> pd.DataFrame:
        """
        Save flight search results to files
        """
        if not results:
            self.log("No results to save")
            return None
        
        # Convert to flat list for DataFrame
        flat_results = []
        for route, flights in results.items():
            for flight in flights:
                flat_results.append({
                    'route': route,
                    'origin': flight.origin,
                    'destination': flight.destination,
                    'departure_date': flight.departure_date,
                    'departure_time': flight.departure_time,
                    'arrival_date': flight.arrival_date,
                    'arrival_time': flight.arrival_time,
                    'duration': flight.duration,
                    'price': flight.price,
                    'currency': flight.currency,
                    'airline': flight.airline,
                    'flight_number': flight.flight_number
                })
        
        df = pd.DataFrame(flat_results)
        
        # Save to CSV
        csv_filename = f"{output_prefix}.csv"
        df.to_csv(csv_filename, index=False)
        self.log(f"Saved flights to CSV: {csv_filename}")
        
        # Save to JSON (newline-delimited for consistency)
        json_filename = f"{output_prefix}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            for result in flat_results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        self.log(f"Saved flights to JSON: {json_filename}")
        
        # Save cheapest flights separately
        self.save_cheapest_flights(results, f"{output_prefix}_cheapest")
        
        # Create summary
        self.create_flight_summary(df, results, output_prefix)
        
        return df
    
    def save_cheapest_flights(self, results: Dict[str, List[FlightInfo]], 
                            output_prefix: str, max_per_route: int = 3):
        """
        Save only the cheapest flights for each route
        """
        cheapest_results = {}
        
        for route, flights in results.items():
            if flights:
                # Sort by price and take the cheapest ones
                sorted_flights = sorted(flights, key=lambda x: x.price)[:max_per_route]
                cheapest_results[route] = [
                    {
                        'departure': f"{f.departure_date} {f.departure_time}",
                        'arrival': f"{f.arrival_date} {f.arrival_time}",
                        'duration': f.duration,
                        'price': f.price,
                        'currency': f.currency,
                        'airline': f.airline,
                        'flight': f.flight_number
                    }
                    for f in sorted_flights
                ]
        
        # Save cheapest flights
        json_filename = f"{output_prefix}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(cheapest_results, f, ensure_ascii=False, indent=2)
        self.log(f"Saved cheapest flights to: {json_filename}")
    
    def create_flight_summary(self, df: pd.DataFrame, results: Dict[str, List[FlightInfo]], 
                            output_prefix: str):
        """
        Create and save summary statistics for flights
        """
        summary = {
            'environment': self.environment,
            'search_date': datetime.now().isoformat(),
            'total_routes': len(results),
            'total_flights': len(df) if df is not None else 0,
            'total_requests': self._request_counter,
            'rate_limit_hits': self._rate_limit_hits,
        }
        
        if df is not None and len(df) > 0:
            # Price statistics
            summary['price_stats'] = {
                'min': float(df['price'].min()),
                'max': float(df['price'].max()),
                'avg': float(df['price'].mean()),
                'median': float(df['price'].median())
            }
            
            # Routes with flights
            routes_with_flights = []
            for route, flights in results.items():
                if flights:
                    route_prices = [f.price for f in flights]
                    routes_with_flights.append({
                        'route': route,
                        'flight_count': len(flights),
                        'min_price': min(route_prices),
                        'avg_price': sum(route_prices) / len(route_prices)
                    })
            
            summary['routes'] = sorted(routes_with_flights, key=lambda x: x['min_price'])[:10]  # Top 10 cheapest
        
        # Save summary
        summary_filename = f"{output_prefix}_summary.json"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.log(f"Saved flight summary to: {summary_filename}")
        
        # Print summary
        self.log(f"\n{'='*40}")
        self.log(f"FLIGHT SEARCH SUMMARY")
        self.log(f"{'='*40}")
        self.log(f"Environment: {summary['environment']}")
        self.log(f"Total routes searched: {summary['total_routes']}")
        self.log(f"Total flights found: {summary['total_flights']}")
        self.log(f"API requests made: {summary['total_requests']}")
        
        if 'price_stats' in summary:
            self.log(f"\nPrice Statistics:")
            self.log(f"  Min: ${summary['price_stats']['min']:.2f}")
            self.log(f"  Max: ${summary['price_stats']['max']:.2f}")
            self.log(f"  Avg: ${summary['price_stats']['avg']:.2f}")
            self.log(f"  Median: ${summary['price_stats']['median']:.2f}")
    
    def print_results(self, results: Dict[str, List[FlightInfo]], max_per_route: int = 3) -> None:
        """
        Print flight search results in a formatted manner.
        """
        for route, flights in results.items():
            print(f"\n{'='*60}")
            print(f"Route: {route}")
            print(f"{'='*60}")
            
            if not flights:
                print("No flights found for this route.")
                continue
            
            # Sort by price
            flights.sort(key=lambda x: x.price)
            
            # Group by date for better readability
            flights_by_date = defaultdict(list)
            for flight in flights:
                flights_by_date[flight.departure_date].append(flight)
            
            for date in sorted(flights_by_date.keys()):
                print(f"\n📅 {date}:")
                date_flights = sorted(flights_by_date[date], key=lambda x: x.price)[:max_per_route]
                for flight in date_flights:
                    print(f"  {flight.departure_time} → {flight.arrival_time} | "
                          f"Duration: {flight.duration} | "
                          f"{flight.airline}{flight.flight_number} | "
                          f"💰 {flight.price:.2f} {flight.currency}")


# Convenience function for easy integration
def create_flight_searcher(api_key: str = None, api_secret: str = None, debug: bool = True):
    """
    Factory function to create the flight searcher
    """
    return AmadeusFlightSearch(api_key, api_secret, debug)


# Wrapper function for easy integration
def process_flights_with_all_fallback(location_airports: Dict[str, List[str]], 
                                    start_date: str, end_date: str,
                                    max_concurrent: int = 3,
                                    output_prefix: str = "flight_results"):
    """
    High-level function to search flights with async-first fallback
    
    Args:
        location_airports: Dictionary mapping location names to their airport codes
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_concurrent: Maximum concurrent requests
        output_prefix: Prefix for output files
    
    Returns:
        Dictionary of flight results
    """
    
    # Create the flight searcher
    searcher = create_flight_searcher(debug=True)
    
    # Search for flights with async-first fallback
    results = searcher.search_flights_with_fallback(
        location_airports, start_date, end_date, max_concurrent
    )
    
    # Save results to files
    if results and output_prefix:
        searcher.save_results(results, output_prefix)
    
    # Print summary
    searcher.print_results(results, max_per_route=3)
    
    return results


# Async wrapper for Jupyter notebooks and async contexts
async def process_flights_async(location_airports: Dict[str, List[str]], 
                              start_date: str, end_date: str,
                              max_concurrent: int = 3,
                              output_prefix: str = "flight_results_async"):
    """
    Async wrapper function for searching flights in Jupyter notebooks and async contexts
    
    Args:
        location_airports: Dictionary mapping location names to their airport codes
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_concurrent: Maximum concurrent requests
        output_prefix: Prefix for output files
    
    Returns:
        Dictionary of flight results
    """
    
    # Create the flight searcher
    searcher = create_flight_searcher(debug=True)
    
    # Search for flights asynchronously
    results = await searcher.search_flights_async_direct(
        location_airports, start_date, end_date, max_concurrent
    )
    
    # Save results to files
    if results and output_prefix:
        searcher.save_results(results, output_prefix)
    
    # Print summary
    searcher.print_results(results, max_per_route=3)
    
    return results


# Convenience wrapper that automatically chooses sync or async
def process_flights_smart(location_airports: Dict[str, List[str]], 
                         start_date: str, end_date: str,
                         max_concurrent: int = 3,
                         output_prefix: str = "flight_results_smart"):
    """
    Smart wrapper that automatically detects environment and uses appropriate method
    For Jupyter notebooks, use process_flights_async() with await instead
    
    Args:
        location_airports: Dictionary mapping location names to their airport codes
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_concurrent: Maximum concurrent requests
        output_prefix: Prefix for output files
    
    Returns:
        Dictionary of flight results
    """
    
    try:
        # Check if we're in an async context
        loop = asyncio.get_running_loop()
        print("⚠️  Detected existing event loop (Jupyter notebook?)")
        print("   For best performance in Jupyter, use: await process_flights_async(...)")
        print("   Falling back to sync processing...")
        
        # In event loop, use sync fallback
        return process_flights_with_all_fallback(location_airports, start_date, end_date, max_concurrent, output_prefix)
        
    except RuntimeError:
        # No event loop, safe to use async
        searcher = create_flight_searcher(debug=True)
        results = asyncio.run(searcher.search_flights_async_robust(
            location_airports, start_date, end_date, max_concurrent
        ))
        
        # Save results to files
        if results and output_prefix:
            searcher.save_results(results, output_prefix)
        
        # Print summary
        searcher.print_results(results, max_per_route=3)
        
        return results


# Test function
def test_flight_searcher(location_airports: Dict[str, List[str]], test_single_route: bool = True):
    """
    Test the flight searcher with sample data
    """
    print(f"\n{'='*60}")
    print(f"TESTING FLIGHT SEARCHER")
    print(f"{'='*60}")
    
    searcher = create_flight_searcher(debug=True)
    
    # Test token acquisition
    print("\n1. Testing token acquisition:")
    try:
        token = searcher._get_access_token()
        print(f"   ✓ Token acquired successfully")
        print(f"   Token expires at: {searcher.token_expiry}")
    except Exception as e:
        print(f"   ✗ Token acquisition failed: {e}")
        return None
    
    # Test single route if requested
    if test_single_route and len(location_airports) >= 2:
        print("\n2. Testing single route:")
        locations = list(location_airports.keys())
        origin_loc = locations[0]
        dest_loc = locations[1]
        origin_code = location_airports[origin_loc][0]
        dest_code = location_airports[dest_loc][0]
        test_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        
        print(f"   {origin_loc} ({origin_code}) -> {dest_loc} ({dest_code})")
        print(f"   Date: {test_date}")
        
        flights = searcher._search_flights_for_route(origin_code, dest_code, test_date)
        
        if flights:
            print(f"   ✓ Found {len(flights)} flights")
            cheapest = min(flights, key=lambda x: x.price)
            print(f"   Cheapest: ${cheapest.price:.2f} on {cheapest.airline}")
        else:
            print(f"   ✗ No flights found")
    
    # Test batch processing with limited scope
    print("\n3. Testing batch processing (limited):")
    test_airports = dict(list(location_airports.items())[:2])  # Only first 2 locations
    test_start = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    test_end = (datetime.now() + timedelta(days=32)).strftime("%Y-%m-%d")  # Only 3 days
    
    results = searcher.search_flights_with_fallback(test_airports, test_start, test_end, max_concurrent=2)
    
    if results:
        searcher.save_results(results, "test_flights")
        print(f"\nTest completed. Found flights for {len(results)} routes.")
        return results
    else:
        print("No flights found in test")
        return None