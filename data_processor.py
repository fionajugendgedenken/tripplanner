"""
Enhanced Data Processor for Trip Planning
Handles filtering and preprocessing of transportation, POI, and route data
"""

import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class EnhancedDataProcessor:
    """
    Processes and filters all data sources for trip planning
    """
    
    def __init__(self, start_date: str = "2025-06-14", end_date: str = "2025-06-22"):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Create date mapping for opening hours
        self.date_mapping = self._create_date_mapping()
        
        # Currency conversion rates to USD (approximate rates)
        self.currency_rates = {
            'USD': 1.0,
            'EUR': 1.08,
            'GBP': 1.25,
            'JPY': 0.0067,
            'CAD': 0.74,
            'AUD': 0.65,
            'CHF': 1.10,
            'CNY': 0.14,
            'SEK': 0.092,
            'NOK': 0.091,
            'DKK': 0.145,
            'PLN': 0.23,
            'CZK': 0.043,
            'HUF': 0.0026,
            'RON': 0.21,
            'BGN': 0.55,
            'HRK': 0.14,
            'RUB': 0.010,
            'TRY': 0.029,
            'BRL': 0.17,
            'MXN': 0.049,
            'INR': 0.012,
            'KRW': 0.00074,
            'SGD': 0.74,
            'HKD': 0.13,
            'NZD': 0.59,
            'ZAR': 0.053,
            'THB': 0.028,
            'MYR': 0.22,
            'PHP': 0.017,
            'IDR': 0.000063,
            'VND': 0.000040
        }
        
    def _create_date_mapping(self) -> Dict[str, str]:
        """Map day names to actual dates in trip period"""
        date_mapping = {}
        current_date = self.start_date
        
        while current_date <= self.end_date:
            day_name = current_date.strftime('%A')  # Monday, Tuesday, etc.
            date_str = current_date.strftime('%Y-%m-%d')
            date_mapping[date_str] = day_name
            current_date += timedelta(days=1)
            
        return date_mapping
    
    def _convert_to_usd(self, price: float, currency: str) -> float:
        """Convert price from given currency to USD"""
        currency = currency.upper().strip()
        
        if currency in self.currency_rates:
            usd_price = price * self.currency_rates[currency]
            return round(usd_price, 2)
        else:
            # If currency not found, assume it's already USD or use 1:1 ratio
            print(f"    Warning: Unknown currency '{currency}', treating as USD")
            return price
    
    def _map_days_to_dates(self, days_list: List[str]) -> List[str]:
        """Map day names (Monday, Tuesday, etc.) to actual trip dates"""
        applicable_dates = []
        
        # Create reverse mapping from day names to dates
        day_to_dates = {}
        for date_str, day_name in self.date_mapping.items():
            if day_name not in day_to_dates:
                day_to_dates[day_name] = []
            day_to_dates[day_name].append(date_str)
        
        # Find all dates that match the specified days
        for day_name in days_list:
            day_name = day_name.strip().title()  # Normalize to "Monday", "Tuesday", etc.
            if day_name in day_to_dates:
                applicable_dates.extend(day_to_dates[day_name])
        
        return sorted(applicable_dates)
    
    def load_and_filter_transportation(self, flights_file: str, ferries_file: str) -> Dict[str, Dict[str, Dict[str, List[Dict]]]]:
        """
        Load and filter flights/ferries with >6.5h and overnight filtering + 1h buffers
        Returns nested dict: {departure_date: {origin: {destination: [transport_options]}}}
        """
        print("Loading and filtering transportation data...")
        
        transportation_by_date_origin_dest = {}
        
        # Process flights
        print("  Processing flights...")
        with open(flights_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        flight = json.loads(line)
                        
                        # Parse duration
                        duration_minutes = self._parse_duration(flight.get('duration', 'PT1H'))
                        
                        # Filter >6.5 hours
                        if duration_minutes > 6.5 * 60:
                            continue
                            
                        # Filter overnight flights
                        if self._is_overnight_transport(flight.get('departure_time', ''), 
                                                     flight.get('arrival_time', '')):
                            continue
                        
                        # Convert price to USD if needed
                        price = flight.get('price', 0)
                        currency = flight.get('currency', 'USD')
                        usd_price = self._convert_to_usd(price, currency)
                        
                        # Add buffers and check time constraints
                        processed_flight = self._add_transport_buffers({
                            'origin': flight['origin'].split(' (')[0],
                            'destination': flight['destination'].split(' (')[0],
                            'mode': 'flight',
                            'base_duration_minutes': duration_minutes,
                            'cost': usd_price,
                            'original_price': price,
                            'original_currency': currency,
                            'departure_time': flight['departure_time'],
                            'arrival_time': flight['arrival_time'],
                            'departure_date': flight.get('departure_date', ''),
                            'raw_data': flight  # Keep original data for reference
                        })
                        
                        # Only add if within time constraints
                        if processed_flight:
                            self._add_to_nested_transport(transportation_by_date_origin_dest, processed_flight)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"    Warning: Skipping invalid flight data: {e}")
        
        # Process ferries
        print("  Processing ferries...")
        with open(ferries_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        ferry = json.loads(line)
                        
                        duration_hours = ferry.get('duration', 0)
                        
                        # Filter >6.5 hours  
                        if duration_hours > 6.5:
                            continue
                            
                        # Filter overnight ferries
                        if self._is_overnight_transport(ferry.get('departure_time', ''), 
                                                     ferry.get('arrival_time', '')):
                            continue
                        
                        # Map days to actual dates
                        days = ferry.get('days', [])
                        if isinstance(days, str):
                            days = [days]  # Convert single day to list
                        
                        applicable_dates = self._map_days_to_dates(days)
                        
                        if not applicable_dates:
                            # If no specific days given, assume all trip days
                            applicable_dates = [self.start_date.strftime('%Y-%m-%d')]
                        
                        # Create ferry entry for each applicable date
                        for departure_date in applicable_dates:
                            # Add buffers and check time constraints
                            processed_ferry = self._add_transport_buffers({
                                'origin': ferry['origin_location'],
                                'destination': ferry['destination_location'],
                                'mode': 'ferry',
                                'base_duration_minutes': int(duration_hours * 60),
                                'cost': ferry['price'],
                                'departure_time': ferry['departure_time'],
                                'arrival_time': ferry['arrival_time'],
                                'departure_date': departure_date,
                                'applicable_days': days,
                                'raw_data': ferry  # Keep original data for reference
                            })
                            
                            # Only add if within time constraints
                            if processed_ferry:
                                self._add_to_nested_transport(transportation_by_date_origin_dest, processed_ferry)
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"    Warning: Skipping invalid ferry data: {e}")
        
        # Print summary
        total_options = sum(
            len(transport_options_list)
            for date_dict in transportation_by_date_origin_dest.values()
            for origin_dict in date_dict.values()
            for transport_options_list in origin_dict.values()
        )
        
        print(f"  Organized {total_options} transportation options")
        print(f"  Available dates: {len(transportation_by_date_origin_dest)}")
        
        # Print structure summary
        for date, origins in list(transportation_by_date_origin_dest.items())[:3]:  # Show first 3 dates
            print(f"    {date}: {len(origins)} origins")
            for origin, destinations in list(origins.items())[:2]:  # Show first 2 origins per date
                dest_count = sum(len(options) for options in destinations.values())
                print(f"      {origin}: {len(destinations)} destinations, {dest_count} total options")
        
        return transportation_by_date_origin_dest
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO duration string to minutes"""
        if not duration_str.startswith('PT'):
            return 60
        
        duration_str = duration_str[2:]  # Remove 'PT'
        hours = 0
        minutes = 0
        
        if 'H' in duration_str:
            hours_part, remainder = duration_str.split('H', 1)
            hours = int(hours_part)
            duration_str = remainder
        
        if 'M' in duration_str:
            minutes_part = duration_str.split('M')[0]
            minutes = int(minutes_part)
        
        return hours * 60 + minutes
    
    def _is_overnight_transport(self, departure_time: str, arrival_time: str) -> bool:
        """Check if transport is overnight (depart after 10pm or arrive before 6am)"""
        try:
            # Extract time part (handle "HH:MM" format)
            dep_time = departure_time.split(':')
            arr_time = arrival_time.split(':')
            
            dep_hour = int(dep_time[0])
            arr_hour = int(arr_time[0])
            
            # Overnight if departure after 10pm or arrival before 6am
            return dep_hour >= 22 or arr_hour <= 6
            
        except (ValueError, IndexError):
            return False  # If can't parse, assume not overnight
    
    def _add_transport_buffers(self, transport: Dict) -> Dict:
        """Add check-in and delay buffers to transportation times"""
        try:
            # Parse times
            dep_time = datetime.strptime(transport['departure_time'], '%H:%M')
            arr_time = datetime.strptime(transport['arrival_time'], '%H:%M')
            
            # Add 1-hour buffer before departure for check-in/security
            buffered_dep = dep_time - timedelta(hours=1)
            
            # Add 1-hour buffer after arrival for baggage/delays
            buffered_arr = arr_time + timedelta(hours=1)
            
            # Check if buffered times are within allowed window (9am-10pm)
            day_start = dep_time.replace(hour=9, minute=0)
            day_end = dep_time.replace(hour=22, minute=0)
            
            # If buffered departure is before 9am or buffered arrival is after 10pm, mark as invalid
            if buffered_dep < day_start or buffered_arr > day_end:
                return None
            
            # Calculate total duration including buffers
            base_duration = transport.get('base_duration_minutes', 0)
            total_duration = base_duration + 120  # Add 2 hours (1h before + 1h after)
            
            return {
                'origin': transport['origin'],
                'destination': transport['destination'],
                'mode': transport['mode'],
                'departure_time': transport['departure_time'],
                'arrival_time': transport['arrival_time'],
                'buffered_departure': buffered_dep.strftime('%H:%M'),
                'buffered_arrival': buffered_arr.strftime('%H:%M'),
                'base_duration_minutes': base_duration,
                'total_duration_minutes': total_duration,
                'cost': transport['cost'],
                'departure_date': transport['departure_date'],
                'raw_data': transport.get('raw_data', {})
            }
        except Exception as e:
            print(f"    Warning: Error adding transport buffers: {e}")
            return None
    
    def _add_to_nested_transport(self, transport_dict: Dict, transport: Dict):
        """Add a transport option to the nested dictionary structure"""
        departure_date = transport.get('departure_date', 'unknown')
        origin = transport['origin']
        destination = transport['destination']
        
        # Initialize nested structure if needed
        if departure_date not in transport_dict:
            transport_dict[departure_date] = {}
        
        if origin not in transport_dict[departure_date]:
            transport_dict[departure_date][origin] = {}
        
        if destination not in transport_dict[departure_date][origin]:
            transport_dict[departure_date][origin][destination] = []
        
        # Add the transport option
        transport_dict[departure_date][origin][destination].append(transport)
    
    def load_and_process_poi_data(self, poi_results_file: str) -> Dict[str, Dict[str, Dict]]:
        """
        Load POI data and organize by location for easier trip planning
        Returns: {location: {poi_name: poi_data}}
        """
        print("Loading and processing POI data...")
        
        poi_data_by_location = {}
        
        with open(poi_results_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                    
                try:
                    poi = json.loads(line)
                    poi_name = poi.get('searched_name', '')
                    location = poi.get('searched_location', '')
                    
                    if not poi_name or not location:
                        continue
                    
                    # Initialize location if not exists
                    if location not in poi_data_by_location:
                        poi_data_by_location[location] = {}
                    
                    # Process opening hours into daily format
                    daily_hours = self._process_opening_hours(poi.get('opening_hours', []))
                    
                    # Determine POI type based on new rules
                    price_level = poi.get('price_level', '')
                    is_restaurant = bool(price_level)  # If there's a price_level, it's a restaurant
                    is_hotel = poi.get('is_hotel', False)
                    is_transportation_hub = poi.get('is_transportation_hub', False)
                    is_poi = not is_transportation_hub  # is_poi if it's not transportation_hub
                    
                    # Process hotel availability based on estimated_costs
                    hotel_availability = {}
                    if is_hotel and poi.get('hotel_info') and poi['hotel_info'].get('estimated_costs'):
                        # Hotel is available if estimated_costs is not null
                        estimated_costs = poi['hotel_info']['estimated_costs']
                        # Create availability mapping for each day (assuming 8 days for the trip)
                        for i, cost in enumerate(estimated_costs):
                            day_key = f"day_{i+1}"
                            hotel_availability[day_key] = cost is not None
                    
                    # Calculate estimated cost based on type
                    estimated_cost = None
                    
                    if poi.get('has_ticket_pricing', False) and poi.get('ticket_info'):
                        # For POIs with ticket pricing, get from ticket_info
                        estimated_cost = poi['ticket_info'].get('estimated_cost')
                    elif is_restaurant and price_level:
                        # For restaurants, use price_level to calculate cost
                        price_mapping = {
                            'PRICE_LEVEL_INEXPENSIVE': 20,
                            'PRICE_LEVEL_MODERATE': 40, 
                            'PRICE_LEVEL_EXPENSIVE': 80,
                            'PRICE_LEVEL_VERY_EXPENSIVE': 150
                        }
                        estimated_cost = price_mapping.get(price_level, 40)  # Default to moderate if unknown
                    elif is_hotel and poi.get('hotel_info') and poi['hotel_info'].get('estimated_costs'):
                        # For hotels, cost should be a list from hotel_info.estimated_costs
                        estimated_cost = poi['hotel_info']['estimated_costs']
                    
                    poi_data_by_location[location][poi_name] = {
                        'name': poi_name,
                        'location': location,
                        'types': poi.get('types', []),
                        'is_hotel': is_hotel,
                        'is_restaurant': is_restaurant,
                        'is_transportation_hub': is_transportation_hub,
                        'is_poi': is_poi,
                        'daily_opening_hours': daily_hours,
                        'hotel_availability': hotel_availability,
                        'estimated_cost': estimated_cost,
                        'price_level': price_level,
                        'latitude': poi.get('latitude', 0),
                        'longitude': poi.get('longitude', 0),
                        'place_id': poi.get('place_id', ''),
                        'preference_score': poi.get('preference_score', 1),
                        'estimated_duration_minutes': poi.get('estimated_duration_minutes', 60)
                    }
                    
                except json.JSONDecodeError as e:
                    print(f"    Warning: Invalid JSON on line {line_num}: {e}")
                except Exception as e:
                    print(f"    Warning: Error processing POI on line {line_num}: {e}")
        
        # Print summary by location
        total_pois = sum(len(pois) for pois in poi_data_by_location.values())
        print(f"  Processed {total_pois} POIs across {len(poi_data_by_location)} locations:")
        for location, pois in poi_data_by_location.items():
            restaurants = sum(1 for poi in pois.values() if poi['is_restaurant'])
            hotels = sum(1 for poi in pois.values() if poi['is_hotel'])
            transport_hubs = sum(1 for poi in pois.values() if poi['is_transportation_hub'])
            regular_pois = sum(1 for poi in pois.values() if poi['is_poi'])
            print(f"    {location}: {len(pois)} total (R:{restaurants}, H:{hotels}, T:{transport_hubs}, P:{regular_pois})")
        
        return poi_data_by_location
    
    def _process_opening_hours(self, opening_hours_data) -> Dict[str, Dict]:
        """
        Convert opening hours to daily format mapped to actual trip dates
        """
        daily_hours = {}
        
        # Initialize all trip dates with default hours
        current_date = self.start_date
        while current_date <= self.end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            daily_hours[date_str] = {'open': '09:00', 'close': '22:00'}  # Default
            current_date += timedelta(days=1)
        
        if not opening_hours_data:
            return daily_hours
        
        # Process actual opening hours data
        if isinstance(opening_hours_data, list):
            for entry in opening_hours_data:
                if isinstance(entry, str):
                    self._parse_opening_hours_entry(entry, daily_hours)
        elif isinstance(opening_hours_data, str):
            self._parse_opening_hours_entry(opening_hours_data, daily_hours)
        
        return daily_hours
    
    def _parse_opening_hours_entry(self, entry: str, daily_hours: Dict):
        """Parse a single opening hours entry and update daily_hours"""
        entry = entry.strip()
        original_entry = entry
        entry_lower = entry.lower()
        
        # Handle 24-hour operation first
        if '24 hours' in entry_lower or 'open 24 hours' in entry_lower:
            # Open 24 hours
            for date_str in daily_hours:
                daily_hours[date_str] = {'open': '00:00', 'close': '23:59'}
            return
        
        # Split by semicolon or comma to handle individual day entries
        day_entries = []
        if ';' in entry:
            day_entries = [part.strip() for part in entry.split(';') if part.strip()]
        elif ',' in entry:
            day_entries = [part.strip() for part in entry.split(',') if part.strip()]
        else:
            day_entries = [entry]
        
        # Process each day entry individually
        for day_entry in day_entries:
            self._parse_single_day_hours(day_entry, daily_hours)
    
    def _parse_single_day_hours(self, day_entry: str, daily_hours: Dict):
        """Parse opening hours for a single day"""
        day_entry = day_entry.strip()
        day_entry_lower = day_entry.lower()
        
        # Map day names to standardized format
        days_map = {
            'monday': 'Monday', 'tuesday': 'Tuesday', 'wednesday': 'Wednesday',
            'thursday': 'Thursday', 'friday': 'Friday', 'saturday': 'Saturday', 
            'sunday': 'Sunday'
        }
        
        # Find which day this entry is about
        target_day = None
        for day_key, day_name in days_map.items():
            if day_key in day_entry_lower:
                target_day = day_name
                break
        
        if not target_day:
            return  # Can't determine which day this is about
        
        # Check if this day is closed
        if 'closed' in day_entry_lower:
            # Apply closed status to matching dates
            for date_str, mapped_day in self.date_mapping.items():
                if mapped_day == target_day:
                    daily_hours[date_str] = {'open': None, 'close': None}
        else:
            # Try to extract time range for this day
            times = self._extract_time_range(day_entry)
            if times:
                # Apply opening hours to matching dates
                for date_str, mapped_day in self.date_mapping.items():
                    if mapped_day == target_day:
                        daily_hours[date_str] = times
    
    def _extract_time_range(self, text: str) -> Optional[Dict[str, str]]:
        """Extract opening and closing times from text"""
        # Look for time patterns like "9:00 AM – 10:00 PM"
        import re
        
        # Pattern to match time ranges
        time_pattern = r'(\d{1,2}:?\d{0,2})\s*(am|pm)?\s*[–-]\s*(\d{1,2}:?\d{0,2})\s*(am|pm)?'
        match = re.search(time_pattern, text, re.IGNORECASE)
        
        if match:
            start_time = match.group(1)
            start_period = match.group(2)
            end_time = match.group(3)
            end_period = match.group(4)
            
            # Convert to 24-hour format
            open_time = self._convert_to_24hour(start_time, start_period)
            close_time = self._convert_to_24hour(end_time, end_period)
            
            if open_time and close_time:
                return {'open': open_time, 'close': close_time}
        
        return None
    
    def _convert_to_24hour(self, time_str: str, period: str = None) -> Optional[str]:
        """Convert time string to 24-hour format"""
        try:
            if ':' not in time_str:
                time_str += ':00'
            
            if period:
                time_obj = datetime.strptime(f"{time_str} {period.upper()}", '%I:%M %p')
            else:
                # Assume 24-hour format if no period
                time_obj = datetime.strptime(time_str, '%H:%M')
            
            return time_obj.strftime('%H:%M')
            
        except ValueError:
            return None
    
    def _process_hotel_availability(self, sold_out_data: Dict) -> Dict[str, bool]:
        """Convert soldOut data to daily availability for trip dates"""
        availability = {}
        
        # Initialize all trip dates as available
        current_date = self.start_date
        while current_date <= self.end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            availability[date_str] = True  # Available by default
            current_date += timedelta(days=1)
        
        # Process sold out data
        if isinstance(sold_out_data, dict):
            for date_key, is_sold_out in sold_out_data.items():
                # Try to parse date key and map to our format
                try:
                    if isinstance(date_key, str) and len(date_key) >= 8:
                        # Try different date formats
                        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                            try:
                                parsed_date = datetime.strptime(date_key, fmt)
                                if self.start_date <= parsed_date <= self.end_date:
                                    date_str = parsed_date.strftime('%Y-%m-%d')
                                    availability[date_str] = not bool(is_sold_out)
                                break
                            except ValueError:
                                continue
                except Exception:
                    continue
        
        return availability
    
    def load_and_process_poi_routes(self, routes_file: str) -> Dict[str, Dict[str, Dict]]:
        """
        Load POI routes and select preferred mode (walking vs driving) with costs
        Returns dict organized by: {origin_name: {destination_name: route_data}}
        Note: Use organize_routes_by_location() to group by location after loading POI data
        """
        print("Loading and processing POI routes...")
        
        # Check file extension to determine format
        if routes_file.endswith('.csv'):
            return self._load_routes_from_csv_nested(routes_file)
        else:
            return self._load_routes_from_json_nested(routes_file)
    
    def _load_routes_from_json_nested(self, routes_file: str) -> Dict[str, Dict[str, Dict[str, Dict]]]:
        """Load routes from JSON file"""
        route_pairs = defaultdict(list)
        
        with open(routes_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        route = json.loads(line)
                        if route.get('status') == 'OK':
                            origin = route.get('origin_name', '')
                            destination = route.get('destination_name', '')
                            mode = route.get('transport_mode', '')
                            
                            if origin and destination and mode in ['walking', 'driving']:
                                pair_key = (origin, destination)
                                route_pairs[pair_key].append(route)
                    except json.JSONDecodeError:
                        continue
        
        return self._process_route_pairs_nested(route_pairs)
    
    def _load_routes_from_csv_nested(self, routes_file: str) -> Dict[str, Dict[str, Dict[str, Dict]]]:
        """Load routes from CSV file"""
        route_pairs = defaultdict(list)
        
        with open(routes_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                origin = row.get('origin_name', '')
                destination = row.get('destination_name', '')
                mode = row.get('transport_mode', '')
                
                if origin and destination and mode in ['walking', 'driving']:
                    duration_minutes = float(row.get('duration_minutes', 0))
                    
                    route = {
                        'origin_name': origin,
                        'destination_name': destination,
                        'transport_mode': mode,
                        'duration_minutes': duration_minutes,
                        'distance_km': float(row.get('distance_km', 0)),
                        'status': 'OK'
                    }
                    
                    pair_key = (origin, destination)
                    route_pairs[pair_key].append(route)
        
        return self._process_route_pairs_nested(route_pairs)
    
    def _process_route_pairs_nested(self, route_pairs: Dict) -> Dict[str, Dict[str, Dict]]:
        """Process route pairs and select preferred mode, organize by origin->destination first"""
        processed_routes = {}
        total_processed = 0
        
        for (origin, destination), routes in route_pairs.items():
            walking_route = None
            driving_route = None
            
            for route in routes:
                if route['transport_mode'] == 'walking':
                    walking_route = route
                elif route['transport_mode'] == 'driving':
                    driving_route = route
            
            # Apply selection rules
            preferred_route = self._select_preferred_route(walking_route, driving_route)
            
            if preferred_route:
                # Add cost calculation
                if preferred_route['transport_mode'] == 'driving':
                    # $1 per minute of driving
                    preferred_route['estimated_cost'] = preferred_route.get('duration_minutes', 0) * 1.0
                else:
                    preferred_route['estimated_cost'] = 0.0  # Walking is free
                
                # Organize by origin -> destination
                if origin not in processed_routes:
                    processed_routes[origin] = {}
                
                processed_routes[origin][destination] = preferred_route
                total_processed += 1
        
        print(f"  Selected {total_processed} preferred routes from {len(route_pairs)} pairs")
        return processed_routes
    
    def _select_preferred_route(self, walking_route: Optional[Dict], 
                              driving_route: Optional[Dict]) -> Optional[Dict]:
        """
        Select preferred route based on rules:
        1. If walking <= 15 mins: always walk
        2. If walking <= 1.3 * driving time: choose walking
        3. Otherwise: choose driving
        """
        if not walking_route and not driving_route:
            return None
        if not walking_route:
            return driving_route
        if not driving_route:
            return walking_route
        
        walking_time = walking_route.get('duration_minutes', float('inf'))
        driving_time = driving_route.get('duration_minutes', float('inf'))
        
        # Rule 1: If walking <= 15 mins, always walk
        if walking_time <= 15:
            return walking_route
        
        # Rule 2: If walking <= 1.3 * driving time, choose walking
        if walking_time <= 1.3 * driving_time:
            return walking_route
        
        # Rule 3: Otherwise, choose driving
        return driving_route
    
    def get_poi_data_by_location(self, poi_data_by_location: Dict[str, Dict[str, Dict]], 
                                location: str) -> Dict[str, Dict]:
        """Get all POIs for a specific location"""
        return poi_data_by_location.get(location, {})
    
    def get_flat_poi_data(self, poi_data_by_location: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict]:
        """Convert location-organized POI data back to flat structure if needed"""
        flat_data = {}
        for location, pois in poi_data_by_location.items():
            flat_data.update(pois)
        return flat_data
    
    def get_locations_list(self, poi_data_by_location: Dict[str, Dict[str, Dict]]) -> List[str]:
        """Get list of all available locations"""
        return list(poi_data_by_location.keys())
    
    def get_visited_unvisited_pois(self, poi_data_by_location: Dict[str, Dict[str, Dict]], 
                                  location: str, visited_poi_names: Set[str]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """
        Split POIs in a location into visited and unvisited based on POI names
        Returns: (visited_pois, unvisited_pois)
        """
        location_pois = self.get_poi_data_by_location(poi_data_by_location, location)
        
        visited_pois = {}
        unvisited_pois = {}
        
        for poi_name, poi_data in location_pois.items():
            if poi_name in visited_poi_names:
                visited_pois[poi_name] = poi_data
            else:
                unvisited_pois[poi_name] = poi_data
        
        return visited_pois, unvisited_pois
    
    def get_transportation_options(self, transport_data: Dict[str, Dict[str, Dict[str, List[Dict]]]], 
                                  departure_date: str, origin: str, destination: str) -> List[Dict]:
        """Get transportation options for specific date, origin, and destination"""
        return transport_data.get(departure_date, {}).get(origin, {}).get(destination, [])
    
    def get_available_destinations(self, transport_data: Dict[str, Dict[str, Dict[str, List[Dict]]]], 
                                  departure_date: str, origin: str) -> List[str]:
        """Get list of available destinations from origin on a specific date"""
        return list(transport_data.get(departure_date, {}).get(origin, {}).keys())
    
    def get_available_origins(self, transport_data: Dict[str, Dict[str, Dict[str, List[Dict]]]], 
                             departure_date: str) -> List[str]:
        """Get list of available origins on a specific date"""
        return list(transport_data.get(departure_date, {}).keys())
    
    def get_available_dates(self, transport_data: Dict[str, Dict[str, Dict[str, List[Dict]]]]) -> List[str]:
        """Get list of all available departure dates"""
        return list(transport_data.keys())
    
    def get_flat_transportation_data(self, transport_data: Dict[str, Dict[str, Dict[str, List[Dict]]]]) -> List[Dict]:
        """Convert nested transportation data back to flat list if needed"""
        flat_list = []
        for date, origins in transport_data.items():
            for origin, destinations in origins.items():
                for destination, options in destinations.items():
                    flat_list.extend(options)
        return flat_list
    
    def find_transportation_routes(self, transport_data: Dict[str, Dict[str, Dict[str, List[Dict]]]], 
                                  origin: str, destination: str, 
                                  date_range: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        Find all transportation options between origin and destination across multiple dates
        Returns: {departure_date: [transport_options]}
        """
        routes_by_date = {}
        
        dates_to_check = date_range if date_range else self.get_available_dates(transport_data)
        
        for date in dates_to_check:
            options = self.get_transportation_options(transport_data, date, origin, destination)
            if options:
                routes_by_date[date] = options
        
        return routes_by_date
    
    def organize_routes_by_location(self, routes_by_origin: Dict[str, Dict[str, Dict]], 
                                   poi_data_by_location: Dict[str, Dict[str, Dict]]) -> Dict[str, Dict[str, Dict[str, Dict]]]:
        """
        Reorganize routes by location using POI location data
        Returns: {origin_location: {origin_name: {destination_name: route_data}}}
        """
        print("Organizing routes by location...")
        
        # Create POI name to location mapping
        poi_to_location = {}
        for location, pois in poi_data_by_location.items():
            for poi_name in pois.keys():
                poi_to_location[poi_name] = location
        
        routes_by_location = {}
        total_organized = 0
        skipped_inter_city = 0
        
        for origin, destinations in routes_by_origin.items():
            origin_location = poi_to_location.get(origin)
            
            if not origin_location:
                continue  # Skip if origin location not found
            
            for destination, route_data in destinations.items():
                destination_location = poi_to_location.get(destination)
                
                # Only include routes within the same location (intra-city)
                if destination_location == origin_location:
                    # Initialize nested structure: location -> origin -> destination
                    if origin_location not in routes_by_location:
                        routes_by_location[origin_location] = {}
                    
                    if origin not in routes_by_location[origin_location]:
                        routes_by_location[origin_location][origin] = {}
                    
                    routes_by_location[origin_location][origin][destination] = route_data
                    total_organized += 1
                else:
                    skipped_inter_city += 1
        
        print(f"  Organized {total_organized} intra-city routes across {len(routes_by_location)} locations")
        print(f"  Skipped {skipped_inter_city} inter-city routes")
        
        for location, origin_dict in routes_by_location.items():
            route_count = sum(len(destinations) for destinations in origin_dict.values())
            origin_count = len(origin_dict)
            print(f"    {location}: {route_count} routes from {origin_count} origins")
        
        return routes_by_location
    
    def get_route_data(self, routes_by_location: Dict[str, Dict[str, Dict[str, Dict]]], 
                      location: str, origin: str, destination: str) -> Optional[Dict]:
        """Get route data for specific location, origin, and destination"""
        return routes_by_location.get(location, {}).get(origin, {}).get(destination)
    
    def get_routes_from_origin(self, routes_by_location: Dict[str, Dict[str, Dict[str, Dict]]], 
                              location: str, origin: str) -> Dict[str, Dict]:
        """Get all routes from a specific origin in a location"""
        return routes_by_location.get(location, {}).get(origin, {})
    
    def get_available_destinations_from_origin(self, routes_by_location: Dict[str, Dict[str, Dict[str, Dict]]], 
                                              location: str, origin: str) -> List[str]:
        """Get list of destinations reachable from origin in a location"""
        return list(routes_by_location.get(location, {}).get(origin, {}).keys())
    
    def get_origins_in_location(self, routes_by_location: Dict[str, Dict[str, Dict[str, Dict]]], 
                               location: str) -> List[str]:
        """Get list of all origins (POIs) that have routes in a location"""
        return list(routes_by_location.get(location, {}).keys())
    
    def get_route_locations(self, routes_by_location: Dict[str, Dict[str, Dict[str, Dict]]]) -> List[str]:
        """Get list of all locations that have route data"""
        return list(routes_by_location.keys())


# Example usage and testing
if __name__ == "__main__":
    processor = EnhancedDataProcessor()
    
    # Test transportation loading (now organized by date->origin->destination)
    try:
        transportation_data = processor.load_and_filter_transportation(
            'location_flights.json', 'ferry_results.json'
        )
        
        # Show structure overview
        total_options = len(processor.get_flat_transportation_data(transportation_data))
        available_dates = processor.get_available_dates(transportation_data)
        print(f"Loaded {total_options} transportation options across {len(available_dates)} dates")
        
        # Example usage of utility functions
        if available_dates:
            test_date = available_dates[0]
            origins = processor.get_available_origins(transportation_data, test_date)
            print(f"On {test_date}, can depart from: {origins}")
            
            if origins:
                test_origin = origins[0]
                destinations = processor.get_available_destinations(transportation_data, test_date, test_origin)
                print(f"From {test_origin} on {test_date}, can go to: {destinations}")
                
                if destinations:
                    test_destination = destinations[0]
                    options = processor.get_transportation_options(transportation_data, test_date, test_origin, test_destination)
                    print(f"Options from {test_origin} to {test_destination} on {test_date}: {len(options)}")
                    
                    if options:
                        sample = options[0]
                        print(f"Sample: {sample['mode']}, {sample['total_duration_minutes']}min, ${sample['cost']}")
        
        # Test route finding across dates
        if len(available_dates) >= 2 and transportation_data:
            # Find routes between any two cities across all dates
            all_origins = set()
            all_destinations = set()
            for date_data in transportation_data.values():
                for origin, dest_data in date_data.items():
                    all_origins.add(origin)
                    all_destinations.update(dest_data.keys())
            
            if all_origins and all_destinations:
                test_origin = list(all_origins)[0]
                test_destination = list(all_destinations)[0]
                routes_by_date = processor.find_transportation_routes(
                    transportation_data, test_origin, test_destination
                )
                print(f"Routes from {test_origin} to {test_destination} available on {len(routes_by_date)} dates")
        
    except FileNotFoundError as e:
        print(f"Transportation files not found: {e}")
    except UnicodeDecodeError as e:
        print(f"Encoding error in transportation files: {e}")
    
    # Test POI data loading (now organized by location)
    try:
        poi_data_by_location = processor.load_and_process_poi_data('poi_results.json')
        total_pois = sum(len(pois) for pois in poi_data_by_location.values())
        print(f"Loaded {total_pois} POIs across {len(poi_data_by_location)} locations")
        
        # Show sample from each location
        if poi_data_by_location:
            for location, pois in list(poi_data_by_location.items())[:2]:  # Show first 2 locations
                sample_name = list(pois.keys())[0]
                sample = pois[sample_name]
                print(f"Sample POI from {location}: {sample_name}")
                print(f"  Type: Hotel={sample['is_hotel']}, Restaurant={sample['is_restaurant']}, POI={sample['is_poi']}")
                print(f"  Cost: {sample['estimated_cost']}, Opening hours: {len(sample['daily_opening_hours'])} days")
    except FileNotFoundError as e:
        print(f"POI files not found: {e}")
    except UnicodeDecodeError as e:
        print(f"Encoding error in POI files: {e}")
    
    # Test route loading
    try:
        routes = processor.load_and_process_poi_routes('poi_routes.json')
        print(f"Loaded {len(routes)} routes")
    except FileNotFoundError as e:
        print(f"Route files not found: {e}")
    except UnicodeDecodeError as e:
        print(f"Encoding error in route files: {e}")
        print("All files should be saved with UTF-8 encoding to handle Greek characters") 