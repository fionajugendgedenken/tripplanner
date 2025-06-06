import requests
import asyncio
import aiohttp
import json
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
import statistics

# Import environment configuration
try:
    from environment_config import ENV_CONFIG, get_requests_kwargs, get_aiohttp_kwargs
    print(f"Environment configuration loaded: {ENV_CONFIG.environment}")
except ImportError:
    print("Warning: environment_config.py not found. Using default settings.")
    ENV_CONFIG = None
    
    def get_requests_kwargs():
        return {'timeout': (10, 30), 'verify': True}
    
    def get_aiohttp_kwargs():
        import aiohttp
        return {'timeout': aiohttp.ClientTimeout(total=60), 'trust_env': False}

# Import API keys directly
try:
    from keys import GOOGLE_API_KEY, SERPAPI_KEY, RAPID_API_KEY
    print("API keys loaded successfully")
except ImportError:
    print("Warning: keys.py not found. Please ensure API keys are available.")
    GOOGLE_API_KEY = None
    SERPAPI_KEY = None
    RAPID_API_KEY = None


class POIDataFetcherWithHotels:
    """
    Enhanced POI fetcher with SerpAPI integration for ticket pricing and Hotels.com API for hotel costs
    Includes robust fallback logic and POI filtering
    """
    
    def __init__(self, poi_list: Dict[str, Dict[str, Dict[str, Any]]], debug: bool = True):
        # Use imported API keys
        self.google_api_key = GOOGLE_API_KEY
        self.serpapi_key = SERPAPI_KEY
        self.rapid_api_key = RAPID_API_KEY
        
        self.poi_list = poi_list
        self.debug = debug
        self.base_url = "https://places.googleapis.com/v1/places:searchText"
        self.serpapi_url = "https://serpapi.com/search.json"
        
        # Hotels.com API endpoints
        self.hotels_location_url = "https://hotels-com-provider.p.rapidapi.com/v2/regions"
        self.hotels_offers_url = "https://hotels-com-provider.p.rapidapi.com/v2/hotels/offers"
        
        # Hotel search date range (2025-06-14 to 2025-06-22)
        self.hotel_start_date = "2025-06-14"
        self.hotel_end_date = "2025-06-22"
        self.hotel_dates = self._generate_date_range()
        
        # Get environment-specific settings
        if ENV_CONFIG:
            self.rate_limit_settings = ENV_CONFIG.rate_limit_settings
            self.is_corporate = ENV_CONFIG.is_corporate
        else:
            self.rate_limit_settings = {
                'min_request_interval': 0.2,
                'max_concurrent': 5,
                'retry_attempts': 3,
                'backoff_factor': 1.5
            }
            self.is_corporate = False
        
        # Transportation hub settings
        self.transportation_hubs = ['Public Airport', 'Public Ferry Terminal', 'Public Train Station']
        self.valid_transport_types = {
            'Public Airport': ['airport', 'international_airport'],
            'Public Ferry Terminal': ['ferry_terminal'],
            'Public Train Station': ['transit_station','train_station']
        }
        self.duplicate_distance_threshold = 2.0
        
        # Food-related types to exclude
        self.food_types = {
            'restaurant', 'food', 'meal_takeaway', 'meal_delivery', 
            'dessert_shop', 'ice_cream_shop', 'bakery', 'cafe', 
            'bar', 'night_club', 'confectionery', 'food_store',
            'breakfast_restaurant', 'fine_dining_restaurant', 'greek_restaurant',
            'mediterranean_restaurant', 'barbecue_restaurant', 'dessert_restaurant'
        }
        
        # Hotel-related types to identify hotels
        self.hotel_types = {
            'hotel', 'lodging', 'resort', 'motel', 'inn', 'bed_and_breakfast',
            'guest_house', 'hostel', 'apartment_hotel', 'boutique_hotel'
        }
        
        # Duration settings
        self.restaurant_price_durations = {
            'PRICE_LEVEL_INEXPENSIVE': 30,
            'PRICE_LEVEL_MODERATE': 60,
            'PRICE_LEVEL_EXPENSIVE': 120,
            'PRICE_LEVEL_VERY_EXPENSIVE': 180
        }
        self.default_duration = 30
        
        # Set up session configuration based on environment
        self._setup_session_config()
    
    def _generate_date_range(self) -> List[str]:
        """Generate list of dates from start to end date"""
        start = datetime.strptime(self.hotel_start_date, "%Y-%m-%d")
        end = datetime.strptime(self.hotel_end_date, "%Y-%m-%d")
        
        dates = []
        current = start
        while current < end:  # Don't include the checkout date
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        return dates
    
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
    
    def log(self, message: str):
        """Simple logging for debugging"""
        if self.debug:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    
    def create_session_kwargs(self):
        """Create aiohttp session kwargs based on environment"""
        # Create timeout WITHOUT using context manager approach
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
    
    def is_hotel(self, place_data: dict, searched_name: str) -> bool:
        """
        Determine if this POI is a hotel based on types and name
        """
        primary_type = place_data.get('primaryType', '').strip()
        types = set(place_data.get('types', []))
        name_lower = place_data.get('displayName', {}).get('text', '').lower()
        searched_lower = searched_name.lower()
        
        # Check if it's a hotel based on types
        is_hotel_type = (primary_type in self.hotel_types or 
                        bool(types.intersection(self.hotel_types)))
        
        # Check if name suggests it's a hotel
        hotel_keywords = ['hotel', 'resort', 'suites', 'inn', 'lodge', 'spa']
        is_hotel_name = any(keyword in searched_lower or keyword in name_lower 
                           for keyword in hotel_keywords)
        
        return is_hotel_type or is_hotel_name
    
    def should_get_ticket_info(self, place_data: dict, searched_name: str) -> bool:
        """
        Determine if we should fetch ticket information for this POI
        Excludes transportation hubs, restaurants/food places, hotels, and places without primary_type
        """
        # Skip if no primary type
        primary_type = place_data.get('primaryType', '').strip()
        if not primary_type:
            self.log(f"  Skipping {searched_name}: No primary type")
            return False
        
        # Skip transportation hubs
        if self.is_transportation_hub(searched_name):
            self.log(f"  Skipping {searched_name}: Transportation hub")
            return False
        
        # Skip food-related places
        types = set(place_data.get('types', []))
        if types.intersection(self.food_types) or primary_type in self.food_types:
            self.log(f"  Skipping {searched_name}: Food/restaurant related")
            return False
        
        # Skip hotels/lodging
        if self.is_hotel(place_data, searched_name):
            self.log(f"  Skipping {searched_name}: Hotel/lodging")
            return False
        
        self.log(f"  ✓ Will fetch tickets for {searched_name} (type: {primary_type})")
        return True
    
    def should_get_hotel_info(self, place_data: dict, searched_name: str) -> bool:
        """
        Determine if we should fetch hotel cost information for this POI
        """
        if self.is_hotel(place_data, searched_name):
            self.log(f"  ✓ Will fetch hotel costs for {searched_name}")
            return True
        return False
    
    def compare_hotel_match(self, poi_name: str, poi_address: str, hotel_data: dict) -> bool:
        """
        Compare POI name and address with hotel data to determine if they match
        """
        hotel_short_name = hotel_data.get('regionNames', {}).get('shortName', '').lower()
        hotel_address = hotel_data.get('hotelAddress', {})
        hotel_street = hotel_address.get('street', '').lower()
        hotel_city = hotel_address.get('city', '').lower()
        
        poi_name_lower = poi_name.lower()
        poi_address_lower = poi_address.lower()
        
        # Check name similarity
        name_match = (poi_name_lower in hotel_short_name or 
                     hotel_short_name in poi_name_lower or
                     any(word in hotel_short_name for word in poi_name_lower.split() if len(word) > 3))
        
        # Check address similarity
        address_match = (hotel_street in poi_address_lower or 
                        hotel_city in poi_address_lower)
        
        return name_match or address_match
    
    async def get_hotel_location_data_async(self, session: aiohttp.ClientSession, poi_name: str, location: str) -> Optional[dict]:
        """
        Search for hotel location data using Hotels.com API (async)
        """
        params = {
            "query": f"{poi_name}, {location}",
            "domain": "US",
            "locale": "en_US"
        }
        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }
        
        try:
            if session.closed:
                self.log(f"  ✗ Session closed for hotel location search: {poi_name}")
                return None
            
            async with session.get(self.hotels_location_url, headers=headers, params=params) as response:
                if response.status == 200:
                    location_data = await response.json()
                    
                    # Find best match (gaiaHotelResult type)
                    for item in location_data.get('data', []):
                        if item.get('@type') == 'gaiaHotelResult':
                            self.log(f"  ✓ Found hotel location data for {poi_name}")
                            return item
                    
                    self.log(f"  - No gaiaHotelResult found for {poi_name}")
                    return None
                else:
                    self.log(f"  ✗ Hotel location search HTTP {response.status} for {poi_name}")
                    return None
                    
        except Exception as e:
            self.log(f"  ✗ Hotel location search error for {poi_name}: {e}")
            return None
    
    def get_hotel_location_data_sync(self, poi_name: str, location: str) -> Optional[dict]:
        """
        Search for hotel location data using Hotels.com API (sync)
        """
        params = {
            "query": f"{poi_name}, {location}",
            "domain": "US",
            "locale": "en_US"
        }
        headers = {
            "X-RapidAPI-Key": self.rapid_api_key,
            "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
        }
        
        try:
            if ENV_CONFIG:
                response = requests.get(self.hotels_location_url, headers=headers, params=params, **ENV_CONFIG.get_requests_kwargs())
            else:
                response = requests.get(self.hotels_location_url, headers=headers, params=params, timeout=30)
            
            response.raise_for_status()
            location_data = response.json()
            
            # Find best match (gaiaHotelResult type)
            for item in location_data.get('data', []):
                if item.get('@type') == 'gaiaHotelResult':
                    self.log(f"  ✓ Found hotel location data for {poi_name}")
                    return item
            
            self.log(f"  - No gaiaHotelResult found for {poi_name}")
            return None
            
        except Exception as e:
            self.log(f"  ✗ Hotel location search error for {poi_name}: {e}")
            return None
    
    async def get_hotel_costs_for_date_async(self, session: aiohttp.ClientSession, hotel_id: str, checkin_date: str, poi_name: str) -> Optional[dict]:
        """
        Get hotel cost for a specific date with minimum stay handling (async)
        Returns dict with cost info or None if unavailable
        """
        max_stay_attempt = 7  # Maximum nights to try for minimum stay
        checkout_date = (datetime.strptime(checkin_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        
        for stay_length in range(1, max_stay_attempt + 1):
            checkout_date = (datetime.strptime(checkin_date, "%Y-%m-%d") + timedelta(days=stay_length)).strftime("%Y-%m-%d")
            
            # Don't go beyond our end date
            if checkout_date > self.hotel_end_date:
                break
            
            params = {
                "domain": "US",
                "adults_number": "1",
                "hotel_id": hotel_id,
                "locale": "en_US",
                "checkin_date": checkin_date,
                "checkout_date": checkout_date,
            }
            headers = {
                "X-RapidAPI-Key": self.rapid_api_key,
                "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
            }
            
            try:
                if session.closed:
                    return None
                
                async with session.get(self.hotels_offers_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        offer_data = await response.json()
                        
                        # Check if sold out
                        if offer_data.get('soldOut', False):
                            return {'cost': None, 'soldOut': True, 'minimum_stay': stay_length}
                        
                        # Check for minimum stay error - simplified logic
                        error_message = offer_data.get('errorMessage')
                        if error_message and stay_length < max_stay_attempt:
                            # If there's any error message and we're not sold out, try longer stay
                            self.log(f"    Error message detected for {poi_name}, trying {stay_length + 1} nights...")
                            continue
                        
                        # Get price from stickyBar
                        price_info = offer_data.get('stickyBar', {}).get('displayPrice')
                        if price_info:
                            # Extract total price
                            import re
                            price_match = re.search(r'[\$]?(\d+(?:\.\d{2})?)', str(price_info).replace(',', ''))
                            if price_match:
                                total_price = float(price_match.group(1))
                                avg_per_night = round(total_price / stay_length, 2)
                                return {
                                    'cost': avg_per_night,
                                    'soldOut': False,
                                    'minimum_stay': stay_length,
                                    'total_price': total_price
                                }
                        
                        # If we get here, there's no price but no error - treat as unavailable
                        return {'cost': None, 'soldOut': False, 'minimum_stay': stay_length, 'no_price': True}
                    
                    else:
                        return None
                        
            except Exception as e:
                self.log(f"  ✗ Hotel cost error for {poi_name} on {checkin_date} ({stay_length} nights): {e}")
                if stay_length == 1:  # Only log error on first attempt
                    return None
        
        # If we've exhausted all stay lengths, return unavailable
        return {'cost': None, 'soldOut': False, 'minimum_stay': max_stay_attempt, 'max_stay_exceeded': True}
    
    def get_hotel_costs_for_date_sync(self, hotel_id: str, checkin_date: str, poi_name: str) -> Optional[dict]:
        """
        Get hotel cost for a specific date with minimum stay handling (sync)
        Returns dict with cost info or None if unavailable
        """
        max_stay_attempt = 7  # Maximum nights to try for minimum stay
        
        for stay_length in range(1, max_stay_attempt + 1):
            checkout_date = (datetime.strptime(checkin_date, "%Y-%m-%d") + timedelta(days=stay_length)).strftime("%Y-%m-%d")
            
            # Don't go beyond our end date
            if checkout_date > self.hotel_end_date:
                break
            
            params = {
                "domain": "US",
                "adults_number": "1",
                "hotel_id": hotel_id,
                "locale": "en_US",
                "checkin_date": checkin_date,
                "checkout_date": checkout_date,
            }
            headers = {
                "X-RapidAPI-Key": self.rapid_api_key,
                "X-RapidAPI-Host": "hotels-com-provider.p.rapidapi.com"
            }
            
            try:
                if ENV_CONFIG:
                    response = requests.get(self.hotels_offers_url, headers=headers, params=params, **ENV_CONFIG.get_requests_kwargs())
                else:
                    response = requests.get(self.hotels_offers_url, headers=headers, params=params, timeout=30)
                
                response.raise_for_status()
                offer_data = response.json()
                
                # Check if sold out
                if offer_data.get('soldOut', False):
                    return {'cost': None, 'soldOut': True, 'minimum_stay': stay_length}
                
                # Check for minimum stay error - simplified logic
                error_message = offer_data.get('errorMessage')
                if error_message and stay_length < max_stay_attempt:
                    # If there's any error message and we're not sold out, try longer stay
                    self.log(f"    Error message detected for {poi_name}, trying {stay_length + 1} nights...")
                    continue
                
                # Get price from stickyBar
                price_info = offer_data.get('stickyBar', {}).get('displayPrice')
                if price_info:
                    # Extract total price
                    import re
                    price_match = re.search(r'[\$]?(\d+(?:\.\d{2})?)', str(price_info).replace(',', ''))
                    if price_match:
                        total_price = float(price_match.group(1))
                        avg_per_night = round(total_price / stay_length, 2)
                        return {
                            'cost': avg_per_night,
                            'soldOut': False,
                            'minimum_stay': stay_length,
                            'total_price': total_price
                        }
                
                # If we get here, there's no price but no error - treat as unavailable
                return {'cost': None, 'soldOut': False, 'minimum_stay': stay_length, 'no_price': True}
                
            except Exception as e:
                self.log(f"  ✗ Hotel cost error for {poi_name} on {checkin_date} ({stay_length} nights): {e}")
                if stay_length == 1:  # Only log error on first attempt
                    return None
        
        # If we've exhausted all stay lengths, return unavailable
        return {'cost': None, 'soldOut': False, 'minimum_stay': max_stay_attempt, 'max_stay_exceeded': True}
    
    async def get_hotel_costs_async(self, session: aiohttp.ClientSession, hotel_data: dict, poi_name: str, poi_address: str) -> Optional[dict]:
        """
        Get hotel costs for all dates if hotel matches (async)
        """
        # Check if this hotel matches our POI
        if not self.compare_hotel_match(poi_name, poi_address, hotel_data):
            self.log(f"  - Hotel data doesn't match POI {poi_name}")
            return None
        
        hotel_id = hotel_data.get('hotelId')
        if not hotel_id:
            return None
        
        self.log(f"  ✓ Hotel match found for {poi_name}, fetching costs...")
        
        # Get costs for each date
        costs = []
        cost_details = []
        
        for date in self.hotel_dates:
            cost_info = await self.get_hotel_costs_for_date_async(session, hotel_id, date, poi_name)
            
            if cost_info:
                costs.append(cost_info.get('cost'))  # This will be None if sold out
                cost_details.append(cost_info)
            else:
                costs.append(None)
                cost_details.append({'cost': None, 'soldOut': True, 'minimum_stay': 1})
            
            # Small delay between date requests
            await asyncio.sleep(0.1)
        
        return {
            'hotelId': hotel_data.get('hotelId'),
            'cityId': hotel_data.get('cityId'),
            'hotelAddress': hotel_data.get('hotelAddress'),
            'shortName': hotel_data.get('regionNames', {}).get('shortName'),
            'estimated_costs': costs,
            'cost_details': cost_details,  # Detailed info about minimum stay, sold out, etc.
            'date_range': self.hotel_dates
        }
    
    def get_hotel_costs_sync(self, hotel_data: dict, poi_name: str, poi_address: str) -> Optional[dict]:
        """
        Get hotel costs for all dates if hotel matches (sync)
        """
        # Check if this hotel matches our POI
        if not self.compare_hotel_match(poi_name, poi_address, hotel_data):
            self.log(f"  - Hotel data doesn't match POI {poi_name}")
            return None
        
        hotel_id = hotel_data.get('hotelId')
        if not hotel_id:
            return None
        
        self.log(f"  ✓ Hotel match found for {poi_name}, fetching costs...")
        
        # Get costs for each date
        costs = []
        cost_details = []
        
        for date in self.hotel_dates:
            cost_info = self.get_hotel_costs_for_date_sync(hotel_id, date, poi_name)
            
            if cost_info:
                costs.append(cost_info.get('cost'))  # This will be None if sold out
                cost_details.append(cost_info)
            else:
                costs.append(None)
                cost_details.append({'cost': None, 'soldOut': True, 'minimum_stay': 1})
            
            # Small delay between date requests
            time.sleep(0.1)
        
        return {
            'hotelId': hotel_data.get('hotelId'),
            'cityId': hotel_data.get('cityId'),
            'hotelAddress': hotel_data.get('hotelAddress'),
            'shortName': hotel_data.get('regionNames', {}).get('shortName'),
            'estimated_costs': costs,
            'cost_details': cost_details,  # Detailed info about minimum stay, sold out, etc.
            'date_range': self.hotel_dates
        }
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula"""
        import math
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return 6371 * c  # Earth radius in km
    
    def filter_valid_transportation_hubs(self, places: List[dict], poi_name: str) -> List[dict]:
        """Filter transportation hub results"""
        if not self.is_transportation_hub(poi_name):
            return places
        
        valid_types = self.valid_transport_types.get(poi_name, [])
        filtered_places = [p for p in places if p.get('primaryType', '') in valid_types]
        
        if not filtered_places:
            return []
        
        # Remove duplicates based on proximity
        deduplicated_places = []
        for place in filtered_places:
            location = place.get('location', {})
            lat = location.get('latitude', 0)
            lon = location.get('longitude', 0)
            
            is_duplicate = False
            for selected_place in deduplicated_places:
                selected_location = selected_place.get('location', {})
                selected_lat = selected_location.get('latitude', 0)
                selected_lon = selected_location.get('longitude', 0)
                
                distance = self.calculate_distance(lat, lon, selected_lat, selected_lon)
                if distance < self.duplicate_distance_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated_places.append(place)
        
        return deduplicated_places[:10]  # Limit to 10 results
    
    def is_transportation_hub(self, poi_name: str) -> bool:
        """Check if POI is a transportation hub"""
        return poi_name in self.transportation_hubs
    
    def get_transportation_search_query(self, poi_name: str, location: str) -> str:
        """Get search query for transportation hubs"""
        if poi_name == 'Public Airport':
            return f'airport {location}'
        elif poi_name == 'Public Ferry Terminal':
            return f'ferry terminal port {location}'
        elif poi_name == 'Public Train Station':
            return f'train station railway {location}'
        else:
            return f'{poi_name}, {location}'
    
    def calculate_visit_duration(self, place_data: dict, searched_name: str, poi_info: Dict[str, Any]) -> float:
        """Calculate visit duration"""
        if poi_info.get('duration') is not None:
            return poi_info['duration']
        
        if self.is_transportation_hub(searched_name):
            return 15
        
        # Check if it's a restaurant and use price level
        types = place_data.get('types', [])
        primary_type = place_data.get('primaryType', '')
        price_level = place_data.get('priceLevel', '')
        name_lower = place_data.get('displayName', {}).get('text', '').lower()
        searched_lower = searched_name.lower()
        
        is_restaurant = ('restaurant' in types or 
                        primary_type == 'restaurant' or 
                        'restaurant' in searched_lower or
                        'restaurant' in name_lower)
        
        if is_restaurant:
            return self.restaurant_price_durations.get(price_level, 60)
        
        return self.default_duration
    
    def extract_locality(self, address_components: List[dict]) -> str:
        """Extract locality from address components"""
        for component in address_components:
            if 'locality' in component.get('types', []):
                return component.get('longText', '')
        return ''
    
    def process_serpapi_admission_data(self, admission_data: List[dict]) -> dict:
        """
        Process SerpAPI admission data to extract ticket pricing information
        Returns processed ticket info with official tickets, tours/activities, and estimated cost
        
        Expected format: List of providers, each with 'title', 'options', and optional 'icon'
        """
        try:
            if not admission_data:
                return {
                    'official_tickets': [],
                    'tours_activities': {'summary': {}, 'providers': {}},
                    'estimated_cost': None
                }
            
            official_tickets = []
            tours_activities_raw = []
            
            for provider in admission_data:
                provider_name = provider.get('title', 'Unknown')
                options = provider.get('options', [])
                
                # Check if this is an official provider
                is_official = (
                    provider.get('icon', '').find('official_admission') != -1 or
                    any(option.get('official_site', False) for option in options) or
                    any(keyword in provider_name.lower() for keyword in ['official', 'museum', 'site'])
                )
                
                # Process each option for this provider
                provider_prices = []
                for option in options:
                    price_usd = option.get('extracted_price')
                    if price_usd is None:
                        # Try to extract from price string if extracted_price not available
                        price_str = option.get('price', '')
                        if price_str:
                            import re
                            price_match = re.search(r'[\$]?(\d+(?:\.\d{2})?)', price_str.replace(',', ''))
                            if price_match:
                                price_usd = float(price_match.group(1))
                    
                    if price_usd is not None:
                        provider_prices.append(price_usd)
                        
                        if is_official:
                            # For official tickets, we'll take the first/main option
                            if not any(ticket['provider_name'] == provider_name for ticket in official_tickets):
                                official_tickets.append({
                                    'provider_name': provider_name,
                                    'price_usd': price_usd
                                })
                
                # For non-official providers, add all their prices to tours/activities
                if not is_official and provider_prices:
                    for price in provider_prices:
                        tours_activities_raw.append({
                            'provider_name': provider_name,
                            'price_usd': price
                        })
            
            # Process tours and activities
            tours_activities = self._process_tours_activities(tours_activities_raw)
            
            # Calculate estimated cost
            estimated_cost = self._calculate_estimated_cost(official_tickets, tours_activities_raw)
            
            return {
                'official_tickets': official_tickets,
                'tours_activities': tours_activities,
                'estimated_cost': estimated_cost
            }
            
        except Exception as e:
            self.log(f"Error in process_serpapi_admission_data: {str(e)}")
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}")
            return {
                'official_tickets': [],
                'tours_activities': {'summary': {}, 'providers': {}},
                'estimated_cost': None
            }
    
    def _process_tours_activities(self, tours_raw: List[dict]) -> dict:
        """Process tours and activities data"""
        if not tours_raw:
            return {'summary': {}, 'providers': {}}
        
        # Group by provider and calculate stats
        provider_stats = {}
        all_prices = []
        
        for item in tours_raw:
            provider = item['provider_name']
            price = item['price_usd']
            
            if price is not None:
                all_prices.append(price)
                if provider not in provider_stats:
                    provider_stats[provider] = []
                provider_stats[provider].append(price)
        
        # Calculate provider statistics
        providers = {}
        for provider, prices in provider_stats.items():
            if prices:
                providers[provider] = {
                    'min': round(min(prices), 2),
                    'max': round(max(prices), 2),
                    'average': round(statistics.mean(prices), 2),
                    'option_count': len(prices)
                }
        
        # Generate summary
        summary = {}
        if providers:
            # Find provider with minimum price
            min_provider_data = min(providers.items(), key=lambda x: x[1]['min'])
            # Find provider with maximum price  
            max_provider_data = max(providers.items(), key=lambda x: x[1]['max'])
            # Find provider with most tour options
            most_tours_provider_data = max(providers.items(), key=lambda x: x[1]['option_count'])
            
            summary = {
                'min_price_provider': {
                    'name': min_provider_data[0],
                    'price': min_provider_data[1]['min']
                },
                'max_price_provider': {
                    'name': max_provider_data[0],
                    'price': max_provider_data[1]['max']
                },
                'most_tours_provider': {
                    'name': most_tours_provider_data[0],
                    'option_count': most_tours_provider_data[1]['option_count']
                }
            }
        
        return {
            'summary': summary,
            'providers': providers
        }
    
    def _calculate_estimated_cost(self, official_tickets: List[dict], tours_raw: List[dict]) -> Optional[float]:
        """Calculate estimated cost based on available data"""
        # Prefer official ticket price
        if official_tickets:
            official_prices = [t['price_usd'] for t in official_tickets if t['price_usd'] is not None]
            if official_prices:
                return round(statistics.mean(official_prices), 2)
        
        # Fall back to average of all tour prices
        if tours_raw:
            tour_prices = [t['price_usd'] for t in tours_raw if t['price_usd'] is not None]
            if tour_prices:
                return round(statistics.mean(tour_prices), 2)
        
        return None
    
    async def search_single_place_async(self, session: aiohttp.ClientSession, place_name: str, location: str) -> List[dict]:
        """Search for a single place using async HTTP request"""
        is_transport_hub = self.is_transportation_hub(place_name)
        
        if is_transport_hub:
            search_query = self.get_transportation_search_query(place_name, location)
            self.log(f"Async searching transportation hub: {search_query}")
        else:
            search_query = f'{place_name}, {location}'
            self.log(f"Async searching: {search_query}")
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.google_api_key,
            'X-Goog-FieldMask': (
                'places.displayName,places.formattedAddress,places.location,'
                'places.rating,places.types,places.priceLevel,places.regularOpeningHours,'
                'places.addressComponents,places.primaryType,places.id'
            )
        }
        
        data = {'textQuery': search_query}
        
        try:
            # Check session state
            if session.closed:
                self.log(f"  ✗ Session is closed for {place_name}")
                return []
            
            # Make the actual HTTP request
            async with session.post(self.base_url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    places = result.get('places', [])
                    
                    if places:
                        if is_transport_hub:
                            filtered_places = self.filter_valid_transportation_hubs(places, place_name)
                            if filtered_places:
                                self.log(f"  ✓ Found {len(filtered_places)} valid {place_name.lower()}(s)")
                                return filtered_places
                            else:
                                self.log(f"  ✗ No valid {place_name.lower()} found")
                                return []
                        else:
                            self.log(f"  ✓ Found: {places[0].get('displayName', {}).get('text', 'Unknown')}")
                            return [places[0]]
                    else:
                        self.log(f"  ✗ No results found for {place_name}")
                        return []
                else:
                    error_text = await response.text()
                    self.log(f"  ✗ HTTP Error {response.status} for {place_name}: {error_text[:100]}")
                    return []
                    
        except asyncio.TimeoutError:
            self.log(f"  ✗ Timeout error for {place_name}")
            return []
        except Exception as e:
            self.log(f"  ✗ Exception for {place_name}: {type(e).__name__}: {str(e)}")
            return []
    
    async def get_serpapi_ticket_info_async(self, session: aiohttp.ClientSession, place_id: str, place_name: str) -> dict:
        """Fetch ticket information from SerpAPI using place_id"""
        params = {
            "engine": "google_maps",
            "place_id": place_id,
            "api_key": self.serpapi_key,
            "hl": "en"
        }
        
        try:
            if session.closed:
                self.log(f"  ✗ Session closed for SerpAPI call: {place_name}")
                return {}
            
            async with session.get(self.serpapi_url, params=params) as response:
                if response.status == 200:
                    search_data = await response.json()
                    admission_data = search_data.get('place_results', {}).get('admission', [])
                    
                    if admission_data:
                        self.log(f"  ✓ Found {len(admission_data)} ticket options for {place_name}")
                        return self.process_serpapi_admission_data(admission_data)
                    else:
                        self.log(f"  - No ticket info found for {place_name}")
                        return {}
                else:
                    self.log(f"  ✗ SerpAPI HTTP {response.status} for {place_name}")
                    return {}
                    
        except Exception as e:
            self.log(f"  ✗ SerpAPI error for {place_name}: {e}")
            return {}
    
    def get_serpapi_ticket_info_sync(self, place_id: str, place_name: str) -> dict:
        """Synchronous version of SerpAPI ticket info fetch"""
        params = {
            "engine": "google_maps",
            "place_id": place_id,
            "api_key": self.serpapi_key,
            "hl": "en"
        }
        
        try:
            if ENV_CONFIG:
                response = requests.get(self.serpapi_url, params=params, **ENV_CONFIG.get_requests_kwargs())
            else:
                response = requests.get(self.serpapi_url, params=params, timeout=30)
            
            response.raise_for_status()
            search_data = response.json()
            admission_data = search_data.get('place_results', {}).get('admission', [])
            
            if admission_data:
                self.log(f"  ✓ Found {len(admission_data)} ticket options for {place_name}")
                return self.process_serpapi_admission_data(admission_data)
            else:
                self.log(f"  - No ticket info found for {place_name}")
                return {}
                
        except Exception as e:
            self.log(f"  ✗ SerpAPI error for {place_name}: {e}")
            return {}
    
    def process_place_data(self, place_data: dict, searched_name: str, searched_location: str, poi_info: Dict[str, Any], result_index: int = 0, ticket_info: dict = None, hotel_info: dict = None) -> dict:
        """Process place data into structured format with ticket and hotel information"""
        location = place_data.get('location', {})
        address_components = place_data.get('addressComponents', [])
        opening_hours = place_data.get('regularOpeningHours', {})
        
        opening_hours_str = ''
        if opening_hours:
            weekday_descriptions = opening_hours.get('weekdayDescriptions', [])
            opening_hours_str = '; '.join(weekday_descriptions)
        
        duration = self.calculate_visit_duration(place_data, searched_name, poi_info)
        
        display_searched_name = searched_name
        if self.is_transportation_hub(searched_name) and result_index > 0:
            display_searched_name = f"{searched_name} ({result_index + 1})"
        
        base_result = {
            'searched_name': display_searched_name,
            'searched_location': searched_location,
            'actual_name': place_data.get('displayName', {}).get('text', ''),
            'place_id': place_data.get('id', ''),
            'address': place_data.get('formattedAddress', ''),
            'latitude': location.get('latitude', 0),
            'longitude': location.get('longitude', 0),
            'opening_hours': opening_hours_str,
            'rating': place_data.get('rating', 0),
            'types': ', '.join(place_data.get('types', [])),
            'primary_type': place_data.get('primaryType', ''),
            'price_level': place_data.get('priceLevel', ''),
            'locality': self.extract_locality(address_components),
            'estimated_duration_minutes': round(duration, 0),
            'preference_score': poi_info.get('preference', 0),
            'duration_source': 'POI_LIST' if poi_info.get('duration') is not None else 'API/Default',
            'is_transportation_hub': self.is_transportation_hub(searched_name),
            'is_hotel': self.is_hotel(place_data, searched_name),
            'result_index': result_index
        }
        
        # Add ticket information if available
        if ticket_info:
            base_result.update({
                'ticket_info': ticket_info,
                'has_ticket_pricing': True
            })
        else:
            base_result.update({
                'ticket_info': None,
                'has_ticket_pricing': False
            })
        
        # Add hotel information if available
        if hotel_info:
            base_result.update({
                'hotel_info': hotel_info,
                'has_hotel_pricing': True
            })
        else:
            base_result.update({
                'hotel_info': None,
                'has_hotel_pricing': False
            })
        
        return base_result
    
    async def process_single_poi_with_all_data_async(self, poi_name: str, location: str, max_concurrent: int = 2) -> List[dict]:
        """
        Process a single POI with Places API, SerpAPI, and Hotels.com API calls
        Uses controlled concurrency for the API calls
        """
        self.log(f"  Processing POI: {poi_name}")
        
        session_kwargs = self.create_session_kwargs()
        
        try:
            async with aiohttp.ClientSession(**session_kwargs) as session:
                # Step 1: Get Places API data
                places_data = await self.search_single_place_async(session, poi_name, location)
                
                if not places_data:
                    return []
                
                poi_info = self.poi_list[location][poi_name]
                results = []
                
                # Step 2: Process each place and get additional info if needed
                for idx, place_data in enumerate(places_data):
                    ticket_info = None
                    hotel_info = None
                    
                    # Check if we should get ticket info
                    if self.should_get_ticket_info(place_data, poi_name):
                        place_id = place_data.get('id', '')
                        if place_id:
                            # Small delay between Places API and SerpAPI call
                            await asyncio.sleep(0.1)
                            ticket_info = await self.get_serpapi_ticket_info_async(session, place_id, poi_name)
                    
                    # Check if we should get hotel info
                    if self.should_get_hotel_info(place_data, poi_name):
                        # Small delay before hotel API calls
                        await asyncio.sleep(0.1)
                        
                        # Get hotel location data
                        hotel_location_data = await self.get_hotel_location_data_async(session, poi_name, location)
                        
                        if hotel_location_data:
                            # Small delay between hotel location and cost calls
                            await asyncio.sleep(0.1)
                            poi_address = place_data.get('formattedAddress', '')
                            hotel_info = await self.get_hotel_costs_async(session, hotel_location_data, poi_name, poi_address)
                    
                    # Process the place data with all info
                    processed = self.process_place_data(place_data, poi_name, location, poi_info, idx, ticket_info, hotel_info)
                    results.append(processed)
                
                return results
                
        except Exception as e:
            self.log(f"  Error processing POI {poi_name}: {e}")
            return []
    
    async def process_poi_batch_async(self, poi_names: List[str], location: str, max_concurrent: int = 3) -> List[dict]:
        """Process a batch of POIs with concurrent processing"""
        self.log(f"Processing async batch: {poi_names}")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_poi_with_semaphore(poi_name: str):
            async with semaphore:
                return await self.process_single_poi_with_all_data_async(poi_name, location)
        
        try:
            tasks = [process_poi_with_semaphore(poi_name) for poi_name in poi_names]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    self.log(f"  Batch error: {result}")
                else:
                    results.extend(result)
            
            return results
            
        except Exception as e:
            self.log(f"Async batch processing error: {e}")
            return []
    
    async def process_poi_batch_sequential_async(self, poi_names: List[str], location: str) -> List[dict]:
        """Process POIs sequentially with async calls (fallback method)"""
        self.log(f"Processing sequential async batch: {poi_names}")
        
        results = []
        for poi_name in poi_names:
            try:
                poi_results = await self.process_single_poi_with_all_data_async(poi_name, location, max_concurrent=1)
                results.extend(poi_results)
                
                # Rate limiting between POIs
                await asyncio.sleep(self.rate_limit_settings['min_request_interval'])
                
            except Exception as e:
                self.log(f"  Sequential async error for {poi_name}: {e}")
        
        return results
    
    def process_single_poi_sync(self, poi_name: str, location: str) -> List[dict]:
        """Synchronous processing of a single POI (final fallback)"""
        self.log(f"  Sync processing POI: {poi_name}")
        
        try:
            # Step 1: Get Places API data
            places_data = self._search_place_sync(poi_name, location)
            
            if not places_data:
                return []
            
            poi_info = self.poi_list[location][poi_name]
            results = []
            
            # Step 2: Process each place and get additional info if needed
            for idx, place_data in enumerate(places_data):
                ticket_info = None
                hotel_info = None
                
                # Check if we should get ticket info
                if self.should_get_ticket_info(place_data, poi_name):
                    place_id = place_data.get('id', '')
                    if place_id:
                        # Small delay between API calls
                        time.sleep(0.1)
                        ticket_info = self.get_serpapi_ticket_info_sync(place_id, poi_name)
                
                # Check if we should get hotel info
                if self.should_get_hotel_info(place_data, poi_name):
                    # Small delay before hotel API calls
                    time.sleep(0.1)
                    
                    # Get hotel location data
                    hotel_location_data = self.get_hotel_location_data_sync(poi_name, location)
                    
                    if hotel_location_data:
                        # Small delay between hotel location and cost calls
                        time.sleep(0.1)
                        poi_address = place_data.get('formattedAddress', '')
                        hotel_info = self.get_hotel_costs_sync(hotel_location_data, poi_name, poi_address)
                
                # Process the place data with all info
                processed = self.process_place_data(place_data, poi_name, location, poi_info, idx, ticket_info, hotel_info)
                results.append(processed)
            
            return results
            
        except Exception as e:
            self.log(f"  Sync error for POI {poi_name}: {e}")
            return []
    
    def _search_place_sync(self, place_name: str, location: str) -> List[dict]:
        """Synchronous search using requests library"""
        is_transport_hub = self.is_transportation_hub(place_name)
        
        if is_transport_hub:
            search_query = self.get_transportation_search_query(place_name, location)
        else:
            search_query = f'{place_name}, {location}'
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.google_api_key,
            'X-Goog-FieldMask': (
                'places.displayName,places.formattedAddress,places.location,'
                'places.rating,places.types,places.priceLevel,places.regularOpeningHours,'
                'places.addressComponents,places.primaryType,places.id'
            )
        }
        
        data = {'textQuery': search_query}
        
        try:
            # Use environment-aware requests
            if ENV_CONFIG:
                response = requests.post(self.base_url, headers=headers, json=data, **ENV_CONFIG.get_requests_kwargs())
            else:
                response = requests.post(self.base_url, headers=headers, json=data, timeout=(10, 30))
            
            if response.status_code == 200:
                result = response.json()
                places = result.get('places', [])
                
                if places:
                    if is_transport_hub:
                        return self.filter_valid_transportation_hubs(places, place_name)
                    else:
                        return [places[0]]
                else:
                    return []
            else:
                self.log(f"Sync HTTP error {response.status_code} for {place_name}")
                return []
                
        except Exception as e:
            self.log(f"Sync exception for {place_name}: {e}")
            return []
    
    async def fetch_all_async_robust(self, max_concurrent: int = 3) -> List[dict]:
        """Fetch all POIs with robust session management and full fallback logic"""
        all_results = []
        
        # Adjust concurrency for corporate environments
        if self.is_corporate:
            max_concurrent = min(max_concurrent, 2)
        
        for location, pois_dict in self.poi_list.items():
            self.log(f"\n{'='*60}")
            self.log(f"Processing location: {location} ({len(pois_dict)} POIs)")
            self.log(f"{'='*60}")
            
            poi_names = list(pois_dict.keys())
            
            # Process in small batches to avoid overwhelming the system
            batch_size = 2 if self.is_corporate else 3
            
            for i in range(0, len(poi_names), batch_size):
                batch = poi_names[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(poi_names) - 1) // batch_size + 1
                
                self.log(f"\nBatch {batch_num}/{total_batches}: Processing {len(batch)} POIs")
                
                # Try concurrent processing first
                try:
                    batch_results = await self.process_poi_batch_async(batch, location, max_concurrent)
                    if batch_results:
                        all_results.extend(batch_results)
                        self.log(f"Batch {batch_num} completed (concurrent): {len(batch_results)} results")
                    else:
                        raise Exception("No results from concurrent processing")
                        
                except Exception as e:
                    self.log(f"Concurrent batch {batch_num} failed: {e}")
                    self.log(f"Falling back to sequential async for batch {batch_num}")
                    
                    # Fallback to sequential async processing
                    try:
                        batch_results = await self.process_poi_batch_sequential_async(batch, location)
                        if batch_results:
                            all_results.extend(batch_results)
                            self.log(f"Batch {batch_num} completed (sequential async): {len(batch_results)} results")
                        else:
                            raise Exception("No results from sequential async")
                            
                    except Exception as e2:
                        self.log(f"Sequential async batch {batch_num} failed: {e2}")
                        self.log(f"Falling back to sync processing for batch {batch_num}")
                        
                        # Final fallback to sync processing
                        for poi_name in batch:
                            try:
                                poi_results = self.process_single_poi_sync(poi_name, location)
                                all_results.extend(poi_results)
                                # Rate limiting between sync calls
                                time.sleep(self.rate_limit_settings['min_request_interval'])
                                
                            except Exception as e3:
                                self.log(f"Sync processing failed for {poi_name}: {e3}")
                        
                        self.log(f"Batch {batch_num} completed (sync fallback)")
                
                # Delay between batches
                if i + batch_size < len(poi_names):
                    delay = self.rate_limit_settings['min_request_interval'] * 2
                    if self.is_corporate:
                        delay *= 2  # Longer delays for corporate
                    
                    self.log(f"  Waiting {delay}s before next batch...")
                    await asyncio.sleep(delay)
        
        self.log(f"\n{'='*60}")
        self.log(f"Processing completed! Total results: {len(all_results)}")
        self.log(f"Results with ticket info: {sum(1 for r in all_results if r.get('has_ticket_pricing', False))}")
        self.log(f"Results with hotel info: {sum(1 for r in all_results if r.get('has_hotel_pricing', False))}")
        self.log(f"{'='*60}")
        
        return all_results
    
    def fetch_all_with_comprehensive_fallback(self, max_concurrent: int = 3) -> List[dict]:
        """
        Public method with comprehensive fallback logic
        1. Try full async processing
        2. Fall back to sync processing if async fails completely
        """
        self.log(f"Starting comprehensive POI fetch with ticket and hotel integration")
        self.log(f"Environment: {'Corporate' if self.is_corporate else 'Personal'}")
        self.log(f"Max concurrent: {max_concurrent}")
        self.log(f"Hotel date range: {self.hotel_start_date} to {self.hotel_end_date}")
        
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                self.log("Event loop detected, using sync processing")
                return self._fetch_all_sync_complete()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self.fetch_all_async_robust(max_concurrent))
            
        except Exception as e:
            self.log(f"Async processing failed: {e}")
            self.log("Falling back to complete synchronous processing...")
            
            # Complete fallback to sync processing
            return self._fetch_all_sync_complete()
    
    def _fetch_all_sync_complete(self) -> List[dict]:
        """Complete synchronous fallback implementation"""
        all_results = []
        
        for location, pois_dict in self.poi_list.items():
            self.log(f"Sync processing location: {location}")
            
            for poi_name, poi_info in pois_dict.items():
                try:
                    poi_results = self.process_single_poi_sync(poi_name, location)
                    all_results.extend(poi_results)
                    
                    # Rate limiting between POIs
                    time.sleep(self.rate_limit_settings['min_request_interval'])
                    
                except Exception as e:
                    self.log(f"Complete sync error for {poi_name}: {e}")
        
        self.log(f"Sync processing completed: {len(all_results)} total results")
        return all_results
    
    def save_results_to_json(self, results: List[dict], filename: str = "poi_results_with_hotels.json"):
        """Save results to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for result in results:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
            
            self.log(f"Results saved to {filename}")
            
            # Print summary statistics
            total_pois = len(results)
            pois_with_tickets = sum(1 for r in results if r.get('has_ticket_pricing', False))
            pois_with_hotels = sum(1 for r in results if r.get('has_hotel_pricing', False))
            transportation_hubs = sum(1 for r in results if r.get('is_transportation_hub', False))
            hotels = sum(1 for r in results if r.get('is_hotel', False))
            
            self.log(f"\nSummary:")
            self.log(f"  Total POIs processed: {total_pois}")
            self.log(f"  POIs with ticket pricing: {pois_with_tickets}")
            self.log(f"  POIs with hotel pricing: {pois_with_hotels}")
            self.log(f"  Transportation hubs: {transportation_hubs}")
            self.log(f"  Hotels identified: {hotels}")
            self.log(f"  Regular POIs: {total_pois - transportation_hubs - hotels}")
            
        except Exception as e:
            self.log(f"Error saving results: {e}")


# Convenience function for easy integration
def create_enhanced_poi_fetcher(poi_list: Dict[str, Dict[str, Dict[str, Any]]], debug: bool = True):
    """
    Factory function to create the enhanced POI fetcher with ticket and hotel integration
    """
    return POIDataFetcherWithHotels(poi_list, debug)


# Example usage function
def process_poi_with_all_data(poi_list: Dict[str, Dict[str, Dict[str, Any]]], 
                            max_concurrent: int = 3, output_file: str = "poi_results_with_hotels.json"):
    """
    High-level function to process POIs with ticket and hotel information
    
    Args:
        poi_list: POI list dictionary
        max_concurrent: Maximum concurrent requests
        output_file: Output JSON file name
    
    Returns:
        List of processed POI results with ticket and hotel information
    """
    
    # Create the enhanced fetcher
    fetcher = create_enhanced_poi_fetcher(poi_list, debug=True)
    
    # Process all POIs with comprehensive fallback
    results = fetcher.fetch_all_with_comprehensive_fallback(max_concurrent)
    
    # Save results to file
    if output_file:
        fetcher.save_results_to_json(results, output_file)
    
    return results

