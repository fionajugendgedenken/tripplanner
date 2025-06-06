"""
Hotel Selection Module
Implements greedy algorithm for selecting hotels based on preference, distance to POIs, and cost
"""

from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class HotelSelector:
    """
    Greedy hotel selection based on multi-criteria scoring
    """
    
    def __init__(self, poi_data: Dict, poi_details: Dict, routes: Dict[str, Dict[str, Dict[str, Dict]]],
                 poi_desirability: float = 1.0, cost_sensitivity: float = 0.1,
                 transportation_averseness: float = 0.5):
        self.poi_data = poi_data
        self.poi_details = poi_details
        self.routes = routes  # Nested routes: {location: {origin: {destination: route_data}}}
        self.poi_desirability = poi_desirability
        self.cost_sensitivity = cost_sensitivity
        self.transportation_averseness = transportation_averseness
        
        print(f"Initialized HotelSelector with routes for {len(routes)} locations")
    
    def select_best_hotel(self, city: str, date: str, global_visited_hotels: Set[str],
                         current_hotel: Optional[str] = None) -> Optional[Dict]:
        """
        Select best available hotel in city using greedy scoring
        
        Args:
            city: City name
            date: Date string (YYYY-MM-DD) for availability check
            global_visited_hotels: Hotels already visited globally (for variety)
            current_hotel: Current hotel if staying in same city (preference for same)
            
        Returns:
            Best hotel info or None if no hotels available
        """
        
        # Get available hotels in city
        available_hotels = self._get_available_hotels(city, date, global_visited_hotels)
        
        if not available_hotels:
            return None
        
        # If we have a current hotel and it's available, consider preference for staying
        if current_hotel and current_hotel in available_hotels:
            # Give bonus to staying in same hotel (reduce moving cost)
            available_hotels[current_hotel]['same_hotel_bonus'] = 10.0
        
        # Score all available hotels
        hotel_scores = {}
        for hotel_name, hotel_info in available_hotels.items():
            score = self._calculate_hotel_score(city, hotel_name, hotel_info)
            hotel_scores[hotel_name] = score
        
        # Select hotel with highest score
        if not hotel_scores:
            return None
        
        best_hotel_name = max(hotel_scores.keys(), key=lambda h: hotel_scores[h])
        best_hotel_info = available_hotels[best_hotel_name]
        
        # Add score to hotel info
        best_hotel_info['selection_score'] = hotel_scores[best_hotel_name]
        
        return {
            'name': best_hotel_name,
            'city': city,
            'date': date,
            'info': best_hotel_info
        }
    
    def _get_available_hotels(self, city: str, date: str, 
                            global_visited_hotels: Set[str]) -> Dict[str, Dict]:
        """Get all available hotels in city for given date"""
        # print(f"      🔍 [HotelSelector] Looking for hotels in {city} on {date}")
        
        if city not in self.poi_data.get('POI_LIST', {}):
            print(f"      ❌ [HotelSelector] City {city} not found in POI_LIST")
            return {}

        available_hotels = {}
        city_pois = self.poi_data['POI_LIST'][city]
        # print(f"      📊 [HotelSelector] Found {len(city_pois)} POIs in {city}")
        
        hotel_count = 0
        for poi_name, poi_info in city_pois.items():
            # Check if it's a hotel
            is_hotel = self._is_hotel(poi_name, poi_info)
            if not is_hotel:
                continue
                
            hotel_count += 1
            # print(f"      🏨 [HotelSelector] Checking hotel: {poi_name}")
            
            # Create enhanced poi_info with name for cost lookup
            enhanced_poi_info = poi_info.copy()
            enhanced_poi_info['name'] = poi_name  # Add name for lookup in poi_details
            
            # Get real hotel cost for the date
            hotel_cost_info = self._get_hotel_cost_for_date(enhanced_poi_info, date)
            if hotel_cost_info is None:
                print(f"      ❌ [HotelSelector] {poi_name} - No cost info for {date}")
                continue  # Hotel not available on this date
            
            # print(f"      ✅ [HotelSelector] {poi_name} - Available for ${hotel_cost_info['cost']:.2f}")
            
            # Include both visited and unvisited hotels
            # (User said: don't exclude unless leaving city)
            
            # Combine basic and detailed info
            hotel_info = {
                'name': poi_name,
                'preference': poi_info.get('preference', 0),
                'base_cost': hotel_cost_info['cost'],
                'is_available': hotel_cost_info['available'],
                'is_visited': poi_name in global_visited_hotels,
                'same_hotel_bonus': 0.0  # Will be set if current hotel
            }
            
            # Add detailed info if available
            if poi_name in self.poi_details:
                details = self.poi_details[poi_name]
                hotel_info.update({
                    'types': details.get('types', []),
                    'latitude': details.get('latitude', 0),
                    'longitude': details.get('longitude', 0)
                })
            
            available_hotels[poi_name] = hotel_info
        
        # print(f"      📊 [HotelSelector] Found {hotel_count} potential hotels, {len(available_hotels)} available")
        return available_hotels
    
    def _get_hotel_cost_for_date(self, poi_info: Dict, date: str) -> Optional[Dict]:
        """Get real hotel cost for specific date from poi_details"""
        # print(f"        🔍 [HotelSelector] Getting cost for date {date}")
        # print(f"        📊 [HotelSelector] POI info keys: {list(poi_info.keys())}")
        
        # The cost data is in poi_details, not in poi_info from POI_LIST
        # We need to find this hotel in poi_details
        hotel_name = poi_info.get('name')  # This should be set when we create the poi_info
        if not hotel_name:
            print(f"        ❌ [HotelSelector] No hotel name found in poi_info")
            return None
        
        # print(f"        🔍 [HotelSelector] Looking for hotel details: {hotel_name}")
        
        # Find hotel details (check both nested and flat structure)
        hotel_details = None
        
        # Check nested structure: poi_details[city][hotel_name]
        for city_name, city_details in self.poi_details.items():
            if isinstance(city_details, dict) and hotel_name in city_details:
                hotel_details = city_details[hotel_name]
                # print(f"        📊 [HotelSelector] Found hotel in nested structure under {city_name}")
                break
        
        # Check flat structure: poi_details[hotel_name]
        if not hotel_details and hotel_name in self.poi_details:
            hotel_details = self.poi_details[hotel_name]
            # print(f"        📊 [HotelSelector] Found hotel in flat structure")
        
        if not hotel_details:
            print(f"        ❌ [HotelSelector] Hotel {hotel_name} not found in poi_details")
            return None
        
        # print(f"        📊 [HotelSelector] Hotel details keys: {list(hotel_details.keys())}")
        
        # Check for estimated_cost (note: singular, not plural)
        if 'estimated_cost' not in hotel_details:
            print(f"        ❌ [HotelSelector] No 'estimated_cost' in hotel details")
            return None
        
        estimated_costs = hotel_details['estimated_cost']
        if not isinstance(estimated_costs, list):
            print(f"        ❌ [HotelSelector] estimated_cost is not a list: {type(estimated_costs)}")
            return None
        
        # print(f"        📊 [HotelSelector] Found estimated_cost list with {len(estimated_costs)} entries: {estimated_costs}")
        
        # Map date to index in estimated_costs list
        try:
            from USER_PROFILE import START_DATE
            start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
            target_date = datetime.strptime(date, '%Y-%m-%d')
            
            # Calculate day index (0-based)
            day_index = (target_date - start_date).days
            
            # print(f"        📅 [HotelSelector] START_DATE={START_DATE}, target_date={date}, day_index={day_index}")
            
            # Check if index is valid
            if day_index < 0 or day_index >= len(estimated_costs):
                print(f"        ❌ [HotelSelector] day_index {day_index} out of range [0, {len(estimated_costs)-1}]")
                return None
            
            # Get cost for this day
            cost = estimated_costs[day_index]
            
            # print(f"        💰 [HotelSelector] Cost for day {day_index}: {cost}")
            
            # Hotel is available if cost is not None
            if cost is None:
                print(f"        ❌ [HotelSelector] Cost is None - hotel not available")
                return None
            
            # Additional check using hotel_availability if present
            hotel_availability = hotel_details.get('hotel_availability', {})
            if hotel_availability:
                day_key = f"day_{day_index + 1}"  # hotel_availability uses 1-based indexing
                is_available = hotel_availability.get(day_key, True)
                # print(f"        📊 [HotelSelector] Hotel availability check: {day_key} = {is_available}")
                
                if not is_available:
                    print(f"        ❌ [HotelSelector] Hotel not available according to hotel_availability")
                    return None
            
            result = {
                'cost': float(cost),
                'available': True,
                'day_index': day_index
            }
            # print(f"        ✅ [HotelSelector] Returning cost info: {result}")
            return result
            
        except (ValueError, IndexError) as e:
            print(f"        ❌ [HotelSelector] Error processing date/cost: {e}")
            return None
    
    def _is_hotel(self, poi_name: str, poi_info: Dict) -> bool:
        """Determine if POI is a hotel"""
        # Check basic criteria (duration is None and high preference)
        if poi_info.get('duration') is None and poi_info.get('preference', 0) >= 50:
            return True
        
        # Check detailed info if available
        if poi_name in self.poi_details:
            return self.poi_details[poi_name].get('is_hotel', False)
        
        # Check name patterns
        hotel_keywords = ['hotel', 'suites', 'resort', 'lodge', 'villa', 'hyatt', 'amanzoe']
        poi_name_lower = poi_name.lower()
        return any(keyword in poi_name_lower for keyword in hotel_keywords)
    
    def _is_hotel_available(self, hotel_name: str, date: str) -> bool:
        """Check if hotel is available on given date using real cost data"""
        if hotel_name not in self.poi_data.get('POI_LIST', {}):
            return False
        
        # Find the hotel in poi_data
        for city_name, city_pois in self.poi_data['POI_LIST'].items():
            if hotel_name in city_pois:
                poi_info = city_pois[hotel_name]
                cost_info = self._get_hotel_cost_for_date(poi_info, date)
                return cost_info is not None and cost_info['available']
        
        # Fallback to detailed info check
        if hotel_name not in self.poi_details:
            return True  # Assume available if no detailed info
        
        hotel_availability = self.poi_details[hotel_name].get('hotel_availability', {})
        return hotel_availability.get(date, True)  # Default to available
    
    def _calculate_hotel_score(self, city: str, hotel_name: str, hotel_info: Dict) -> float:
        """
        Calculate hotel score based on:
        preference_score - β * avg_distance_to_POIs - β * cost + same_hotel_bonus
        """
        
        # Base preference score
        preference_score = self.poi_desirability * hotel_info.get('preference', 0)
        
        # Hotel cost penalty
        cost_penalty = self.cost_sensitivity * hotel_info.get('base_cost', 0)
        
        # Average distance to other POIs penalty
        avg_distance_penalty = self.transportation_averseness * self._calculate_avg_distance_to_pois(
            city, hotel_name
        )
        
        # Same hotel bonus (if staying in same hotel)
        same_hotel_bonus = hotel_info.get('same_hotel_bonus', 0)
        
        # Variety bonus (slight preference for unvisited hotels, but not exclusion)
        variety_bonus = 5.0 if not hotel_info.get('is_visited', False) else 0.0
        
        # Total score
        score = (
            preference_score - 
            cost_penalty - 
            avg_distance_penalty + 
            same_hotel_bonus +
            variety_bonus
        )
        
        return score
    
    def _calculate_avg_distance_to_pois(self, city: str, hotel_name: str) -> float:
        """Calculate average travel time from hotel to all other POIs in city"""
        if city not in self.poi_data.get('POI_LIST', {}):
            return 0.0
        
        total_distance = 0.0
        poi_count = 0
        
        for poi_name, poi_info in self.poi_data['POI_LIST'][city].items():
            # Skip the hotel itself
            if poi_name == hotel_name:
                continue
            
            # Skip other hotels (we care about attractions/restaurants)
            if self._is_hotel(poi_name, poi_info):
                continue
            
            # Skip transportation hubs
            if 'Public' in poi_name:
                continue
            
            # Get travel time to this POI
            travel_time = self._get_travel_time(hotel_name, poi_name, city)
            
            # Skip POIs with no route (too far apart or unconnected)
            if travel_time is None:
                continue
                
            total_distance += travel_time
            poi_count += 1
        
        # Return average travel time in minutes
        if poi_count == 0:
            return 0.0
        
        avg_distance = total_distance / poi_count
        
        # Convert to hours for penalty calculation
        return avg_distance / 60.0
    
    def _get_travel_time(self, origin: str, destination: str, city: str = None) -> Optional[float]:
        """Get travel time between POIs in minutes. Returns None if no route exists."""
        if origin == destination:
            return 0.0
        
        # Direct lookup using known city for efficiency
        if city and city in self.routes:
            city_routes = self.routes[city]
            if origin in city_routes and destination in city_routes[origin]:
                return city_routes[origin][destination].get('duration_minutes')
        
        # Fallback: search all locations if city not specified or not found
        for location, location_routes in self.routes.items():
            if origin in location_routes and destination in location_routes[origin]:
                return location_routes[origin][destination].get('duration_minutes')
        
        # No route found - POIs are too far apart or unconnected
        return None
    
    def get_top_hotels_in_city(self, city: str, date: str, top_k: int = 3) -> List[Dict]:
        """
        Get top K hotels in city for comparison
        """
        available_hotels = self._get_available_hotels(city, date, set())
        
        if not available_hotels:
            return []
        
        # Score all hotels
        hotel_scores = []
        for hotel_name, hotel_info in available_hotels.items():
            score = self._calculate_hotel_score(city, hotel_name, hotel_info)
            hotel_scores.append({
                'name': hotel_name,
                'score': score,
                'info': hotel_info
            })
        
        # Sort by score and return top K
        hotel_scores.sort(key=lambda h: h['score'], reverse=True)
        return hotel_scores[:top_k]


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    try:
        poi_data = {'POI_LIST': {
            'Athens, Greece': {
                'Grand Hyatt Athens': {'duration': None, 'preference': 50},
                'Mona Athens': {'duration': None, 'preference': 50},
                'A for Athens': {'duration': None, 'preference': 5},
                'Acropolis of Athens': {'duration': 135, 'preference': 50},
                'Delta Restaurant': {'duration': None, 'preference': 5}
            }
        }}
        
        poi_details = {
            'Grand Hyatt Athens': {
                'is_hotel': True,
                'hotel_availability': {'2025-06-14': True, '2025-06-15': True}
            },
            'Mona Athens': {
                'is_hotel': True,
                'hotel_availability': {'2025-06-14': True, '2025-06-15': False}
            }
        }
        
        routes = {
            'Athens, Greece': {
                'Grand Hyatt Athens': {
                    'Acropolis of Athens': {
                        'duration_minutes': 25,
                        'estimated_cost': 15.0
                    }
                },
                'Mona Athens': {
                    'Acropolis of Athens': {
                        'duration_minutes': 15,
                        'estimated_cost': 10.0
                    }
                }
            }
        }
        
        # Create selector
        selector = HotelSelector(poi_data, poi_details, routes, poi_desirability=1.0, cost_sensitivity=0.1, transportation_averseness=0.5)
        
        # Test hotel selection
        best_hotel = selector.select_best_hotel(
            city='Athens, Greece',
            date='2025-06-14',
            global_visited_hotels=set()
        )
        
        if best_hotel:
            print("Best Hotel Selection:")
            print(f"  Name: {best_hotel['name']}")
            print(f"  Score: {best_hotel['info']['selection_score']:.2f}")
            print(f"  Preference: {best_hotel['info']['preference']}")
            print(f"  Cost: ${best_hotel['info']['base_cost']:.2f}")
        else:
            print("No hotels available")
        
        # Test top hotels
        print("\nTop Hotels in Athens:")
        top_hotels = selector.get_top_hotels_in_city('Athens, Greece', '2025-06-14', top_k=3)
        for i, hotel in enumerate(top_hotels, 1):
            print(f"  {i}. {hotel['name']}: Score {hotel['score']:.2f}")
        
    except Exception as e:
        print(f"Test failed: {e}") 