"""
Intra-City Subproblem Solver
Uses beam search to solve TSP-like problems within cities with time windows,
opening hours, and restaurant constraints.
"""

import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class IntracityState:
    """State for beam search in intra-city optimization"""
    current_poi: str
    current_time: datetime
    visited_pois: Set[str]
    visited_restaurants_today: int
    total_preference_score: float
    total_travel_time: float  # minutes
    total_cost: float
    path: List[Tuple[str, datetime, datetime]]  # (poi, start_time, end_time)
    
    def copy(self) -> 'IntracityState':
        """Create a copy of this state"""
        return IntracityState(
            current_poi=self.current_poi,
            current_time=self.current_time,
            visited_pois=self.visited_pois.copy(),
            visited_restaurants_today=self.visited_restaurants_today,
            total_preference_score=self.total_preference_score,
            total_travel_time=self.total_travel_time,
            total_cost=self.total_cost,
            path=self.path.copy()
        )
    
    def __lt__(self, other):
        """For priority queue comparison - higher objective score is better
        Note: This method may not be used directly since beam search uses explicit objective score calculation"""
        # This is a simplified comparison - actual beam search uses the full objective function
        return self.total_preference_score > other.total_preference_score


class IntracitySubproblemSolver:
    """
    Solves intra-city optimization using beam search with constraints
    """
    
    def __init__(self, poi_data: Dict, poi_details: Dict, routes: Dict[str, Dict[str, Dict[str, Dict]]], 
                 poi_desirability: float = 1.0, cost_sensitivity: float = 0.1,
                 transportation_averseness: float = 0.5):
        self.poi_data = poi_data  # From POI_LIST.json with preferences
        self.poi_details = poi_details  # From poi_results_*.json with opening hours - nested by location
        self.routes = routes  # Nested routes: {location: {origin: {destination: route_data}}}
        self.poi_desirability = poi_desirability
        self.cost_sensitivity = cost_sensitivity
        self.transportation_averseness = transportation_averseness
        
        print(f"Initialized IntracitySubproblemSolver with routes for {len(routes)} locations")
    
    def solve_subproblem(self, city: str, start_poi: str, start_time: datetime, 
                        end_time: datetime, global_visited_pois: Set[str],
                        beam_width: int = 5, date: str = None, end_poi: str = None) -> Dict:
        """
        Solve intra-city subproblem using beam search
        
        Args:
            city: City name
            start_poi: Starting POI (usually a hotel)
            start_time: Start time for the period
            end_time: End time for the period  
            global_visited_pois: POIs already visited in THIS trajectory (across all cities)
                                 Note: In beam search, each beam state has its own trajectory
                                 and thus its own set of visited POIs
            beam_width: Beam search width
            date: Date string for opening hours check (if None, extract from start_time)
            end_poi: Optional ending POI (for hotel-to-hotel round trips)
            
        Returns:
            Best solution with objective score and path
        """
        
        print(f"    🔍 [SubproblemSolver] Starting solve_subproblem")
        print(f"    📍 City: {city}")
        print(f"    🏨 Start POI: {start_poi}")
        print(f"    🎯 End POI: {end_poi if end_poi else 'Any/Open'}")
        print(f"    🕐 Time: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
        print(f"    📅 Date: {date}")
        print(f"    🎯 Beam width: {beam_width}")
        print(f"    🚫 Global visited POIs: {len(global_visited_pois)}")
        
        # Extract date if not provided
        if date is None:
            date = start_time.strftime('%Y-%m-%d')
            print(f"    📅 Extracted date: {date}")
        
        # Automatically determine if restaurant is required (>5 hours)
        duration_hours = (end_time - start_time).total_seconds() / 3600
        require_restaurant = duration_hours > 5.0
        
        print(f"    ⏱️  Duration: {duration_hours:.1f} hours")
        print(f"    🍽️  Restaurant required: {require_restaurant}")
        
        # Get available POIs in this city
        city_pois = self._get_city_pois(city, global_visited_pois)
        
        if not city_pois:
            print(f"    ❌ [SubproblemSolver] No POIs available in {city}")
            return {
                'objective_score': 0.0,
                'path': [],
                'visited_pois': set(),
                'total_cost': 0.0,
                'total_travel_time': 0.0,
                'feasible': True
            }
        
        print(f"    📊 [SubproblemSolver] Found {len(city_pois)} available POIs")
        
        # Initialize beam search
        initial_state = IntracityState(
            current_poi=start_poi,
            current_time=start_time,
            visited_pois=set(),  # Don't automatically mark start_poi as visited
            visited_restaurants_today=0,
            total_preference_score=0.0,
            total_travel_time=0.0,
            total_cost=0.0,
            path=[]
        )
        
        # Add starting POI score if it's a valid POI and restaurant (for tracking purposes)
        if start_poi in city_pois:
            poi_info = city_pois[start_poi]
            if poi_info.get('is_restaurant', False):
                initial_state.visited_restaurants_today += 1
        
        print(f"    🚀 [SubproblemSolver] Starting beam search with initial score: {initial_state.total_preference_score}")
        
        # Run beam search
        best_solution = self._beam_search(
            initial_state, city_pois, end_time, beam_width, require_restaurant, city, date, end_poi
        )
        
        print(f"    🏆 [SubproblemSolver] Beam search completed")
        print(f"    📊 Final objective score: {best_solution['objective_score']:.2f}")
        print(f"    🏛️  POIs visited: {len(best_solution['visited_pois'])}")
        print(f"    🍽️  Restaurants: {best_solution.get('visited_restaurants', 0)}")
        print(f"    ✅ Feasible: {best_solution['feasible']}")
        
        # Print detailed path information
        if best_solution['path']:
            print(f"    📍 Path details:")
            for i, visit in enumerate(best_solution['path'], 1):
                poi_name = visit['poi_name']
                arrival = visit['arrival_time']
                departure = visit['departure_time']
                duration = visit['visit_duration_minutes']
                preference = visit['preference_score']
                travel_cost = visit['travel_cost']
                poi_cost = visit['poi_cost']
                travel_mode = visit['travel_mode']
                
                # Format times
                if isinstance(arrival, str):
                    arrival_str = arrival
                else:
                    arrival_str = arrival.strftime('%H:%M')
                
                if isinstance(departure, str):
                    departure_str = departure
                else:
                    departure_str = departure.strftime('%H:%M')
                
                print(f"       {i:2d}. {arrival_str}-{departure_str}: {poi_name}")
                print(f"           ⏱️  Duration: {duration}min, ⭐ Pref: {preference}, 💰 Cost: ${travel_cost + poi_cost:.2f}")
                print(f"           🚌 Travel: {travel_mode}")
        else:
            print(f"    📍 No activities planned")
        
        print(f"    💰 Total costs: ${best_solution['total_cost']:.2f}")
        print(f"    🚌 Total travel time: {best_solution['total_travel_time']:.1f} min")
        
        return best_solution
    
    def _get_city_pois(self, city: str, global_visited_pois: Set[str]) -> Dict[str, Dict]:
        """Get available POIs in city excluding globally visited ones"""
        print(f"      🔍 [SubproblemSolver] Getting POIs for {city}")
        print(f"      🚫 [SubproblemSolver] Global visited POIs: {global_visited_pois}")
        
        if city not in self.poi_details:
            print(f"      ❌ [SubproblemSolver] City {city} not found in poi_details")
            return {}
        
        city_pois = {}
        city_poi_details = self.poi_details[city]
        print(f"      📊 [SubproblemSolver] Found {len(city_poi_details)} POIs in {city}")
        
        processed_count = 0
        skipped_visited = 0
        skipped_transport = 0
        added_count = 0
        
        for poi_name, poi_details in city_poi_details.items():
            processed_count += 1
            
            # Skip if already visited globally
            if poi_name in global_visited_pois:
                skipped_visited += 1
                continue
            
            # Skip transportation hubs
            if poi_details.get('is_transportation_hub', False):
                skipped_transport += 1
                continue
            
            # Use POI details directly
            city_pois[poi_name] = poi_details
            added_count += 1
            
            poi_type = "🏨 Hotel" if poi_details.get('is_hotel', False) else ("🍽️ Restaurant" if poi_details.get('is_restaurant', False) else "🏛️ Attraction")
            # print(f"        ✅ [SubproblemSolver] {poi_name}: {poi_type}, pref={poi_details.get('preference_score', 0)}")
        
        print(f"      📊 [SubproblemSolver] Summary: {processed_count} processed, {skipped_visited} visited, {skipped_transport} transport, {added_count} available")
        return city_pois
    
    def _beam_search(self, initial_state: IntracityState, city_pois: Dict,
                    end_time: datetime, beam_width: int, require_restaurant: bool, city: str, date: str, end_poi: str) -> Dict:
        """
        Beam search implementation for intra-city optimization
        """
        # Priority queue for beam states: (negative_objective_score, state)
        initial_objective = self._calculate_objective_score(initial_state)
        beam = [(-initial_objective, initial_state)]
        best_solution = None
        best_score = float('-inf')
        
        iterations = 0
        max_iterations = 200  # Prevent infinite loops
        
        while beam and iterations < max_iterations:
            iterations += 1
            
            # Get current beam states
            current_beam = []
            while beam and len(current_beam) < beam_width:
                _, state = heapq.heappop(beam)
                current_beam.append(state)
            
            # Generate successors for each state in beam
            next_beam = []
            
            for state in current_beam:
                # Check if this is a terminal state (good enough solution)
                if self._is_good_terminal_state(state, end_time, require_restaurant, end_poi):
                    state_objective = self._calculate_objective_score(state)
                    if state_objective > best_score:
                        best_score = state_objective
                        best_solution = self._state_to_solution(state, end_poi)
                
                # Generate successor states
                successors = self._generate_successors(state, city_pois, end_time, city, date, end_poi)
                
                for successor in successors:
                    # Feasibility check
                    if self._is_feasible_state(successor, end_time):
                        # Use full objective score for beam search priority
                        successor_objective = self._calculate_objective_score(successor)
                        heapq.heappush(next_beam, (-successor_objective, successor))
            
            # Keep only top beam_width states for next iteration
            beam = []
            seen_states = set()
            
            while next_beam and len(beam) < beam_width:
                score, state = heapq.heappop(next_beam)
                
                # Avoid duplicate states (same current POI + visited set)
                state_key = (state.current_poi, frozenset(state.visited_pois))
                if state_key not in seen_states:
                    seen_states.add(state_key)
                    beam.append((score, state))
        
        # If no solution found, return empty solution
        if best_solution is None:
            return {
                'objective_score': 0.0,
                'path': [],
                'visited_pois': set(),
                'total_cost': 0.0,
                'total_travel_time': 0.0,
                'feasible': False
            }
        
        return best_solution
    
    def _generate_successors(self, state: IntracityState, city_pois: Dict,
                           end_time: datetime, city: str, date: str, end_poi: str) -> List[IntracityState]:
        """Generate all valid successor states from current state"""
        successors = []
        
        # First check if current POI is an unvisited attraction (e.g., hotel with tourist value)
        if (state.current_poi in city_pois and 
            state.current_poi not in state.visited_pois):  # Allow visiting current POI if it has tourist value
            
            current_poi_info = city_pois[state.current_poi]
            preference_score = current_poi_info.get('preference_score', 0)
            
            # Only create successor if the POI has tourist value (preference_score > 0)
            if preference_score > 0:
                
                # Calculate visit duration for current POI
                visit_duration = current_poi_info.get('estimated_duration_minutes')
                if visit_duration is None:
                    visit_duration = current_poi_info.get('duration')
                if visit_duration is None:  # Restaurant
                    visit_duration = 90  # Default 1.5 hours for restaurants
                else:
                    visit_duration = int(visit_duration)
                
                departure_time = state.current_time + timedelta(minutes=visit_duration)
                
                # Check if visiting current POI plus travel to end_poi is feasible
                feasible = True
                if end_poi and end_poi != state.current_poi:
                    travel_to_end_time = self._get_travel_time(state.current_poi, end_poi, city)
                    if travel_to_end_time is not None:
                        final_end_time = departure_time + timedelta(minutes=travel_to_end_time)
                        if final_end_time > end_time:
                            feasible = False
                else:
                    if departure_time > end_time:
                        feasible = False
                
                # Check opening hours
                if feasible:
                    is_open = self._is_poi_open_during_visit(current_poi_info, state.current_time, departure_time, date)
                    
                    if is_open:
                        # Get POI cost
                        poi_cost = self._get_poi_estimated_cost(current_poi_info)
                        
                        # Create successor for visiting current POI
                        successor = state.copy()
                        successor.current_time = departure_time
                        successor.visited_pois.add(state.current_poi)
                        successor.total_preference_score += current_poi_info.get('preference_score', 0)
                        successor.total_cost += poi_cost  # Only POI cost, no travel cost (already here)
                        
                        # Track restaurant visits
                        if current_poi_info.get('is_restaurant', False):
                            successor.visited_restaurants_today += 1
                        
                        # Add to path
                        visit_info = {
                            'poi_name': state.current_poi,
                            'arrival_time': state.current_time,
                            'departure_time': departure_time,
                            'visit_duration_minutes': visit_duration,
                            'preference_score': current_poi_info.get('preference_score', 0),
                            'travel_time_minutes': 0,  # No travel time (already here)
                            'travel_cost': 0.0,  # No travel cost
                            'poi_cost': poi_cost,
                            'travel_mode': 'already_here',
                            'is_restaurant': current_poi_info.get('is_restaurant', False),
                            'is_hotel': current_poi_info.get('is_hotel', False),
                            'poi_type': 'restaurant' if current_poi_info.get('is_restaurant', False) else ('hotel' if current_poi_info.get('is_hotel', False) else 'attraction')
                        }
                        successor.path.append(visit_info)
                        
                        successors.append(successor)
        
        # Generate successors for other POIs
        for poi_name, poi_info in city_pois.items():
            # Skip if already visited
            if poi_name in state.visited_pois:
                continue
            
            # Skip hotels in path planning (hotels are endpoints)
            if poi_info.get('is_hotel', False):
                continue
            
            # Check if route exists between current POI and target POI
            travel_time = self._get_travel_time(state.current_poi, poi_name, city)
            travel_cost = self._get_travel_cost(state.current_poi, poi_name, city)
            travel_mode = self._get_travel_mode(state.current_poi, poi_name, city)
            
            # Screen out POIs with no route (too far apart or unconnected)
            if travel_time is None or travel_cost is None:
                continue
            
            # Calculate arrival time
            arrival_time = state.current_time + timedelta(minutes=travel_time)
            
            # Calculate visit duration
            visit_duration = poi_info.get('estimated_duration_minutes')  # Try detailed info first
            if visit_duration is None:
                visit_duration = poi_info.get('duration')  # Try basic info
            if visit_duration is None:  # Restaurant (duration=None in basic data)
                visit_duration = 90  # Default 1.5 hours for restaurants
            else:
                visit_duration = int(visit_duration)  # Ensure it's an integer
            
            departure_time = arrival_time + timedelta(minutes=visit_duration)
            
            # Check if we have enough time for this visit plus travel to end_poi (if specified)
            if end_poi and end_poi != poi_name:
                travel_to_end_time = self._get_travel_time(poi_name, end_poi, city)
                if travel_to_end_time is not None:
                    final_end_time = departure_time + timedelta(minutes=travel_to_end_time)
                    if final_end_time > end_time:
                        continue  # Skip this POI - no time to reach end_poi after visiting
            else:
                # Check if we have enough time before end_time
                if departure_time > end_time:
                    continue
            
            # Enhanced opening hours check: POI must be open during ENTIRE visit duration
            if not self._is_poi_open_during_visit(poi_info, arrival_time, departure_time, date):
                continue
            
            # Get POI estimated cost (from detailed POI info if available)
            poi_cost = self._get_poi_estimated_cost(poi_info)
            
            # Create successor state
            successor = state.copy()
            successor.current_poi = poi_name
            successor.current_time = departure_time
            successor.visited_pois.add(poi_name)
            successor.total_preference_score += poi_info.get('preference_score', 0)
            successor.total_travel_time += travel_time
            successor.total_cost += travel_cost + poi_cost  # Add both travel cost AND POI cost
            
            # Track restaurant visits
            if poi_info.get('is_restaurant', False):
                successor.visited_restaurants_today += 1
            
            # Add to path with detailed information
            visit_info = {
                'poi_name': poi_name,
                'arrival_time': arrival_time,
                'departure_time': departure_time,
                'visit_duration_minutes': visit_duration,
                'preference_score': poi_info.get('preference_score', 0),
                'travel_time_minutes': travel_time,
                'travel_cost': travel_cost,
                'poi_cost': poi_cost,  # Add POI cost to visit info
                'travel_mode': travel_mode,
                'is_restaurant': poi_info.get('is_restaurant', False),
                'is_hotel': poi_info.get('is_hotel', False),
                'poi_type': 'restaurant' if poi_info.get('is_restaurant', False) else ('hotel' if poi_info.get('is_hotel', False) else 'attraction')
            }
            successor.path.append(visit_info)
            
            successors.append(successor)
        
        return successors
    
    def _is_good_terminal_state(self, state: IntracityState, end_time: datetime,
                               require_restaurant: bool, end_poi: str = None) -> bool:
        """Check if state represents a good terminal solution"""
        
        # If end_poi is specified, we need to check if we can reach it
        if end_poi:
            # Calculate travel time to end_poi
            travel_time_to_end = self._get_travel_time(state.current_poi, end_poi)
            if travel_time_to_end is None:
                return False  # Can't reach end_poi
            
            # Check if we have enough time to reach end_poi
            arrival_at_end = state.current_time + timedelta(minutes=travel_time_to_end)
            if arrival_at_end > end_time:
                return False  # Not enough time to reach end_poi
            
            # If we're already at end_poi, check other constraints
            if state.current_poi == end_poi:
                time_remaining = (end_time - state.current_time).total_seconds() / 60
                # Hotels have restaurants, so if we're at the end_poi (hotel), restaurant requirement is satisfied
                if require_restaurant and state.visited_restaurants_today == 0:
                    # All lodging has restaurants, so being at the hotel satisfies restaurant requirement
                    pass  # Restaurant requirement satisfied by being at hotel
                return True
            
            # We can reach end_poi but need to check if we should go there now
            time_remaining_after_end = (end_time - arrival_at_end).total_seconds() / 60
            if time_remaining_after_end >= 0:  # We can reach end_poi in time
                # Check restaurant requirement
                if require_restaurant and state.visited_restaurants_today == 0:
                    return False
                return True
            
            return False
        
        # Original logic for open ending
        time_remaining = (end_time - state.current_time).total_seconds() / 60
        if time_remaining < 30:  # Less than 30 minutes left
            # Check restaurant requirement
            if require_restaurant and state.visited_restaurants_today == 0:
                return False
            return True
        
        return False
    
    def _is_feasible_state(self, state: IntracityState, end_time: datetime) -> bool:
        """Check if state is feasible (basic constraints)"""
        # Must not exceed time limit
        if state.current_time > end_time:
            return False
        
        # Must have reasonable preference score (pruning)
        if state.total_preference_score < 0:
            return False
        
        return True
    
    def _is_poi_open_during_visit(self, poi_info: Dict, arrival_time: datetime, 
                                 departure_time: datetime, date: str) -> bool:
        """Check if POI is open during the entire visit duration"""
        # Hotels are always "open"
        if poi_info.get('is_hotel', False):
            return True
        
        # Check daily opening hours if available
        daily_hours = poi_info.get('daily_opening_hours', {})
        date_str = date
        
        if date_str in daily_hours:
            hours = daily_hours[date_str]
            open_time_str = hours.get('open')
            close_time_str = hours.get('close')
            
            if open_time_str is None or close_time_str is None:
                return False  # Closed
            
            try:
                open_time = datetime.strptime(open_time_str, '%H:%M').time()
                close_time = datetime.strptime(close_time_str, '%H:%M').time()
                arrival_time_only = arrival_time.time()
                departure_time_only = departure_time.time()
                
                # Handle overnight hours (close_time < open_time)
                if close_time < open_time:
                    # Open overnight (e.g., 18:00 - 02:00)
                    # Must arrive after opening OR before closing
                    # Must depart after opening OR before closing
                    arrival_valid = arrival_time_only >= open_time or arrival_time_only <= close_time
                    departure_valid = departure_time_only >= open_time or departure_time_only <= close_time
                    return arrival_valid and departure_valid
                else:
                    # Normal hours (e.g., 09:00 - 17:00)
                    # Must arrive after opening AND depart before closing
                    return (open_time <= arrival_time_only <= close_time and 
                           open_time <= departure_time_only <= close_time)
                    
            except ValueError:
                pass
        
        # Default: open 9am-10pm, check both arrival and departure times
        arrival_hour = arrival_time.hour
        departure_hour = departure_time.hour
        return (9 <= arrival_hour <= 22 and 9 <= departure_hour <= 22)
    
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
    
    def _get_travel_cost(self, origin: str, destination: str, city: str = None) -> Optional[float]:
        """Get travel cost between POIs. Returns None if no route exists."""
        if origin == destination:
            return 0.0
        
        # Direct lookup using known city for efficiency
        if city and city in self.routes:
            city_routes = self.routes[city]
            if origin in city_routes and destination in city_routes[origin]:
                return city_routes[origin][destination].get('estimated_cost', 0.0)
        
        # Fallback: search all locations if city not specified or not found
        for location, location_routes in self.routes.items():
            if origin in location_routes and destination in location_routes[origin]:
                return location_routes[origin][destination].get('estimated_cost', 0.0)
        
        # No route found - POIs are too far apart or unconnected
        return None
    
    def _get_travel_mode(self, origin: str, destination: str, city: str = None) -> Optional[str]:
        """Get travel mode between POIs. Returns None if no route exists."""
        if origin == destination:
            return 'same_location'
        
        # Direct lookup using known city for efficiency
        if city and city in self.routes:
            city_routes = self.routes[city]
            if origin in city_routes and destination in city_routes[origin]:
                return city_routes[origin][destination].get('transport_mode', 'unknown')
        
        # Fallback: search all locations if city not specified or not found
        for location, location_routes in self.routes.items():
            if origin in location_routes and destination in location_routes[origin]:
                return location_routes[origin][destination].get('transport_mode', 'unknown')
        
        # No route found
        return None
    
    def _get_poi_estimated_cost(self, poi_info: Dict) -> float:
        """Get POI estimated cost from detailed POI info"""
        # First check if estimated_cost is directly available in enhanced poi_info
        if 'estimated_cost' in poi_info and poi_info['estimated_cost'] is not None:
            estimated_cost = poi_info['estimated_cost']
            
            # If it's a list (hotel lodging prices), return fixed cost for visiting as attraction
            if isinstance(estimated_cost, list):
                return 50.0  # Fixed cost for visiting hotel as tourist attraction
            else:
                return float(estimated_cost)
        
        # If not found, return 0.0 (free attractions)
        return 0.0
    
    def _calculate_objective_score(self, state: IntracityState) -> float:
        """Calculate objective score using user profile weights"""
        return (
            self.poi_desirability * state.total_preference_score - 
            self.transportation_averseness * (state.total_travel_time / 60.0) -  # Convert to hours
            self.cost_sensitivity * state.total_cost
        )
    
    def _state_to_solution(self, state: IntracityState, end_poi: str = None) -> Dict:
        """Convert state to solution format"""
        # Calculate objective score with proper weighting (use consistent helper method)
        objective_score = self._calculate_objective_score(state)
        
        # Create solution with current path
        solution = {
            'objective_score': objective_score,
            'preference_score': state.total_preference_score,
            'path': state.path.copy(),
            'visited_pois': state.visited_pois.copy(),
            'total_cost': state.total_cost,
            'total_travel_time': state.total_travel_time,
            'visited_restaurants': state.visited_restaurants_today,
            'feasible': True
        }
        
        # If end_poi is specified and we're not already there, add final travel
        if end_poi and state.current_poi != end_poi:
            travel_time = self._get_travel_time(state.current_poi, end_poi)
            travel_cost = self._get_travel_cost(state.current_poi, end_poi)
            travel_mode = self._get_travel_mode(state.current_poi, end_poi)
            
            if travel_time is not None and travel_cost is not None:
                # Add final travel to end_poi
                arrival_time = state.current_time + timedelta(minutes=travel_time)
                
                final_travel = {
                    'poi_name': end_poi,
                    'arrival_time': arrival_time,
                    'departure_time': arrival_time,  # No visit time at end_poi
                    'visit_duration_minutes': 0,
                    'preference_score': 0,  # No preference score for ending location
                    'travel_time_minutes': travel_time,
                    'travel_cost': travel_cost,
                    'poi_cost': 0.0,  # No POI cost for ending location
                    'travel_mode': travel_mode,
                    'is_restaurant': False,
                    'is_hotel': True,  # Assuming end_poi is typically a hotel
                    'poi_type': 'endpoint'
                }
                
                solution['path'].append(final_travel)
                solution['total_travel_time'] += travel_time
                solution['total_cost'] += travel_cost
                solution['visited_pois'].add(end_poi)
                
                # Recalculate objective score with final travel
                solution['objective_score'] = (
                    self.poi_desirability * solution['preference_score'] - 
                    self.transportation_averseness * (solution['total_travel_time'] / 60.0) -  # Convert to hours
                    self.cost_sensitivity * solution['total_cost']
                )
        
        return solution


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    from data_processor import EnhancedDataProcessor
    
    processor = EnhancedDataProcessor()
    
    # Load sample data
    try:
        poi_data = {'POI_LIST': {
            'Athens, Greece': {
                'Acropolis of Athens': {'duration': 135, 'preference': 50},
                'Acropolis Museum': {'duration': 105, 'preference': 4},
                'Delta Restaurant': {'duration': None, 'preference': 5},
                'Grand Hyatt Athens': {'duration': None, 'preference': 50}
            }
        }}
        
        poi_details = {}
        routes = {
            'Athens, Greece': {
                'Grand Hyatt Athens': {
                    'Acropolis of Athens': {
                        'duration_minutes': 25,
                        'cost': 15.0,
                        'mode': 'driving'
                    }
                }
            }
        }
        
        # Create solver
        solver = IntracitySubproblemSolver(poi_data, poi_details, routes, 
                                         poi_desirability=1.0, cost_sensitivity=0.1, transportation_averseness=0.5)
        
        # Test the solver
        result = solver.solve_subproblem(
            city='Athens, Greece',
            start_poi='Grand Hyatt Athens',
            start_time=datetime(2025, 6, 14, 9, 0),
            end_time=datetime(2025, 6, 14, 18, 0),
            global_visited_pois=set(),
            beam_width=5,
            date='2025-06-14',  # Provide specific date for opening hours
            end_poi='Grand Hyatt Athens'  # Round trip - start and end at same hotel
        )
        
        print("Test Solution:")
        print(f"  Duration: {(datetime(2025, 6, 14, 18, 0) - datetime(2025, 6, 14, 9, 0)).total_seconds() / 3600:.1f} hours")
        print(f"  Restaurant required: {(datetime(2025, 6, 14, 18, 0) - datetime(2025, 6, 14, 9, 0)).total_seconds() / 3600 > 5.0}")
        print(f"  Objective Score: {result['objective_score']:.2f}")
        print(f"  Preference Score: {result['preference_score']:.2f}")
        print(f"  Visited POIs: {len(result['visited_pois'])}")
        print(f"  Total Cost: ${result['total_cost']:.2f}")
        print(f"  Travel Time: {result['total_travel_time']:.1f} minutes")
        print(f"  Feasible: {result['feasible']}")
        
        if result['path']:
            print("  Path:")
            for visit_info in result['path']:
                print(f"    {visit_info['poi_name']}: {visit_info['arrival_time'].strftime('%H:%M')} - {visit_info['departure_time'].strftime('%H:%M')}")
        
    except Exception as e:
        print(f"Test failed: {e}") 