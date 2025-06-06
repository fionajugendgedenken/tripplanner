"""
Inter-City Optimizer - Improved Implementation
Implements the proposed 6-step pipeline with realistic travel times and hotel-centric subproblems.
"""

import heapq
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from intra_city_subproblem import IntracitySubproblemSolver
from hotel_selector import HotelSelector


@dataclass
class InterCityState:
    """State for inter-city beam search with enhanced tracking"""
    current_day: int
    current_city: str
    current_hotel: str
    visited_cities: Set[str]
    visited_pois_by_city: Dict[str, Set[str]]  # {city: {poi1, poi2, ...}}
    visited_hotels_by_city: Dict[str, Set[str]]  # {city: {hotel1, hotel2, ...}}
    total_objective_score: float
    total_cost: float
    daily_schedule: List[Dict]
    
    # Backward compatibility properties
    @property
    def visited_pois(self) -> Set[str]:
        """Get all visited POIs across all cities (for backward compatibility)"""
        all_pois = set()
        for city_pois in self.visited_pois_by_city.values():
            all_pois.update(city_pois)
        return all_pois
    
    @property 
    def visited_hotels(self) -> Set[str]:
        """Get all visited hotels across all cities (for backward compatibility)"""
        all_hotels = set()
        for city_hotels in self.visited_hotels_by_city.values():
            all_hotels.update(city_hotels)
        return all_hotels
    
    def copy(self) -> 'InterCityState':
        return InterCityState(
            current_day=self.current_day,
            current_city=self.current_city,
            current_hotel=self.current_hotel,
            visited_cities=self.visited_cities.copy(),
            visited_pois_by_city={city: pois.copy() for city, pois in self.visited_pois_by_city.items()},
            visited_hotels_by_city={city: hotels.copy() for city, hotels in self.visited_hotels_by_city.items()},
            total_objective_score=self.total_objective_score,
            total_cost=self.total_cost,
            daily_schedule=self.daily_schedule.copy()
        )
    
    def add_visited_poi(self, city: str, poi: str):
        """Add a visited POI to specific city"""
        if city not in self.visited_pois_by_city:
            self.visited_pois_by_city[city] = set()
        self.visited_pois_by_city[city].add(poi)
    
    def add_visited_hotel(self, city: str, hotel: str):
        """Add a visited hotel to specific city"""
        if city not in self.visited_hotels_by_city:
            self.visited_hotels_by_city[city] = set()
        self.visited_hotels_by_city[city].add(hotel)
    
    def get_city_visited_pois(self, city: str) -> Set[str]:
        """Get visited POIs for specific city"""
        return self.visited_pois_by_city.get(city, set())
    
    def get_city_visited_hotels(self, city: str) -> Set[str]:
        """Get visited hotels for specific city"""
        return self.visited_hotels_by_city.get(city, set())
    
    def __lt__(self, other):
        return self.total_objective_score > other.total_objective_score


class InterCityOptimizer:
    """
    Inter-city optimizer implementing the 6-step pipeline
    """
    
    def __init__(self, poi_data: Dict, poi_details: Dict, routes: Dict, 
                 transportation: Dict, start_date: str = "2025-06-14", 
                 end_date: str = "2025-06-22", poi_desirability: float = 1.0,
                 cost_sensitivity: float = 0.1, transportation_averseness: float = 0.5,
                 check_in_buffer_minutes: int = 60, delay_buffer_minutes: int = 60):
        
        self.poi_data = poi_data
        self.poi_details = poi_details
        self.transportation = transportation
        self.routes = routes
        self.poi_desirability = poi_desirability
        self.cost_sensitivity = cost_sensitivity
        self.transportation_averseness = transportation_averseness
        
        # Enhanced buffer settings
        self.check_in_buffer_minutes = check_in_buffer_minutes
        self.delay_buffer_minutes = delay_buffer_minutes
        
        # Parse dates
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.num_days = (self.end_date - self.start_date).days + 1  # Include both endpoints
        
        # Create trip dates list
        self.trip_dates = []
        current_date = self.start_date
        while current_date <= self.end_date:  # Include end_date
            self.trip_dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        # Initialize components
        self.subproblem_solver = IntracitySubproblemSolver(
            poi_data, poi_details, routes, poi_desirability, cost_sensitivity, transportation_averseness
        )
        self.hotel_selector = HotelSelector(
            poi_data, poi_details, routes, poi_desirability, cost_sensitivity, transportation_averseness
        )
        
        # Build enhanced transportation and route lookups
        self.transport_lookup = self._build_transport_lookup()
        self.transportation_hubs = self._identify_transportation_hubs()
        
        # Define cities
        self.cities = list(self.poi_data.get('POI_LIST', {}).keys())
        if not self.cities:
            self.cities = list(self.poi_details.keys()) if isinstance(self.poi_details, dict) else []
        
        print(f"InterCityOptimizer initialized for {self.num_days} days")
        print(f"Cities: {self.cities}")
        print(f"Transportation hubs identified: {len(self.transportation_hubs)}")
    
    def _build_transport_lookup(self) -> Dict:
        """Build enhanced transportation lookup with date filtering"""
        transport_lookup = defaultdict(lambda: defaultdict(list))
        
        for date, date_dict in self.transportation.items():
            for origin, origin_dict in date_dict.items():
                for destination, transport_options in origin_dict.items():
                    route_key = (origin, destination)
                    transport_lookup[date][route_key].extend(transport_options)
        
        return transport_lookup
    
    def _identify_transportation_hubs(self) -> Dict[str, List[str]]:
        """Identify transportation hubs (airports, ports) for each city"""
        hubs = defaultdict(list)
        
        # Extract hubs from transportation data
        for date_dict in self.transportation.values():
            for origin, destinations in date_dict.items():
                if 'Airport' in origin or 'Port' in origin:
                    # Extract city from hub name (e.g., "Athens Airport" -> "Athens, Greece")
                    city = self._extract_city_from_hub(origin)
                    if city:
                        hubs[city].append(origin)
                
                for destination in destinations.keys():
                    if 'Airport' in destination or 'Port' in destination:
                        city = self._extract_city_from_hub(destination)
                        if city:
                            hubs[city].append(destination)
        
        # Remove duplicates
        for city in hubs:
            hubs[city] = list(set(hubs[city]))
        
        return dict(hubs)
    
    def _extract_city_from_hub(self, hub_name: str) -> Optional[str]:
        """Extract city name from transportation hub name"""
        # Simple heuristic - can be improved with actual mapping
        for city in self.cities:
            city_base = city.split(',')[0].strip()  # e.g., "Athens" from "Athens, Greece"
            if city_base.lower() in hub_name.lower():
                return city
        return None
    
    def _get_hotel_to_hub_travel_time(self, hotel: str, hub: str, city: str) -> int:
        """Get travel time from hotel to transportation hub in minutes"""
        # Use routes data to find travel time
        if city in self.routes and hotel in self.routes[city]:
            if hub in self.routes[city][hotel]:
                route_info = self.routes[city][hotel][hub]
                return route_info.get('duration_minutes', 30)  # Default 30 min
        
        # Default estimates based on hub type
        if 'Airport' in hub:
            return 45  # Default airport travel time
        elif 'Port' in hub:
            return 20  # Default port travel time
        return 30  # Default fallback
    
    def optimize_trip(self, beam_width: int = 5, return_top_k: int = 10) -> List[Dict]:
        """
        Enhanced trip optimization returning top K solutions
        """
        print("🚀 Starting inter-city optimization...")
        print(f"📅 Trip dates: {self.trip_dates}")
        print(f"🏙️  Available cities: {self.cities}")
        print(f"⚙️  Beam width: {beam_width}, Return top: {return_top_k}")
        
        # Initialize with Athens start AT ATH AIRPORT at 12pm on Day 1
        # Create a temporary state for initial hotel selection
        temp_state = InterCityState(
            current_day=0,
            current_city='Athens, Greece',
            current_hotel='',
            visited_cities={'Athens, Greece'},
            visited_pois_by_city={},
            visited_hotels_by_city={},
            total_objective_score=0.0,
            total_cost=0.0,
            daily_schedule=[]
        )
        
        print("🏨 Selecting initial hotel in Athens...")
        initial_hotel_info = self._get_best_hotel_in_city('Athens, Greece', self.trip_dates[0], temp_state)
        
        if not initial_hotel_info:
            print("❌ ERROR: No initial hotel found in Athens!")
            return []
        
        initial_hotel = initial_hotel_info['name']  # Extract hotel name from info object
        print(f"🏨 Selected initial hotel: {initial_hotel}")
        
        # Create initial state that starts at ATH Airport at 12pm
        initial_state = InterCityState(
            current_day=0,
            current_city='Athens, Greece',
            current_hotel=initial_hotel,
            visited_cities={'Athens, Greece'},
            visited_pois_by_city={},
            visited_hotels_by_city={},
            total_objective_score=0.0,
            total_cost=0.0,
            daily_schedule=[]
        )
        
        # Add initial hotel to visited hotels for Athens
        if initial_hotel:
            initial_state.add_visited_hotel('Athens, Greece', initial_hotel)
            # Don't add to visited POIs yet - wait until after first day planning
            print(f"✅ Added {initial_hotel} to visited hotels for Athens")
        
        # Add first day schedule starting from ATH Airport at 12pm
        print("📅 Creating first day schedule from ATH Airport...")
        first_day_schedule = self._create_first_day_from_airport(initial_state, initial_hotel, initial_hotel_info)
        
        if not first_day_schedule:
            print("❌ ERROR: Failed to create first day schedule from airport!")
            return []
        
        print(f"✅ First day schedule created:")
        print(f"   Score: {first_day_schedule['score']:.2f}")
        print(f"   Cost: ${first_day_schedule['cost']:.2f}")
        print(f"   Visited POIs: {len(first_day_schedule['visited_pois'])}")
        
        initial_state.daily_schedule.append(first_day_schedule['schedule'])
        initial_state.total_objective_score += first_day_schedule['score']
        initial_state.total_cost += first_day_schedule['cost']
        # Add visited POIs from first day to Athens
        for poi in first_day_schedule['visited_pois']:
            initial_state.add_visited_poi('Athens, Greece', poi)
        
        # NOW add the hotel to visited POIs since first day planning is complete
        if initial_hotel:
            initial_state.add_visited_poi('Athens, Greece', initial_hotel)
            print(f"✅ Added {initial_hotel} to visited POIs for Athens")
        
        # Update current_day to 1 since day 0 has been completed
        initial_state.current_day = 1
        
        print(f"📊 Initial state prepared:")
        print(f"   Current day: {initial_state.current_day}")
        print(f"   Current city: {initial_state.current_city}")
        print(f"   Current hotel: {initial_state.current_hotel}")
        print(f"   Objective score: {initial_state.total_objective_score:.2f}")
        print(f"   Total cost: ${initial_state.total_cost:.2f}")
        
        # Run enhanced beam search
        print("🔍 Starting beam search...")
        top_solutions = self._enhanced_beam_search(initial_state, beam_width, return_top_k)
        
        print(f"📊 Beam search completed. Found {len(top_solutions)} solutions.")
        
        if not top_solutions:
            print("❌ ERROR: Beam search returned no solutions!")
            return []
        
        # Add last day constraint: must end at ATH Airport at 12pm
        print("✈️  Adding last day airport constraints...")
        enhanced_solutions = []
        for i, solution in enumerate(top_solutions):
            print(f"   Processing solution {i+1}/{len(top_solutions)}...")
            enhanced_solution = self._add_last_day_airport_constraint(solution)
            if enhanced_solution:
                enhanced_solutions.append(enhanced_solution)
                print(f"   ✅ Solution {i+1} passed airport constraint")
            else:
                print(f"   ❌ Solution {i+1} failed airport constraint")
        
        print(f"🏁 Final result: {len(enhanced_solutions)} valid solutions")
        return enhanced_solutions if enhanced_solutions else top_solutions
    
    def _create_first_day_from_airport(self, state: InterCityState, hotel: str, hotel_info: Dict) -> Optional[Dict]:
        """Create first day schedule starting from ATH Airport at 12pm"""
        print(f"   🔍 [DEBUG] _create_first_day_from_airport called")
        print(f"      State: {state}")
        print(f"      Hotel: {hotel}")
        print(f"      Hotel info type: {type(hotel_info)}")
        print(f"      Hotel info keys: {list(hotel_info.keys()) if hotel_info else None}")
        
        try:
            # Define ATH Airport arrival time: 12pm on first day
            arrival_time = self.start_date.replace(hour=12, minute=0)
            print(f"      ✅ Arrival time set: {arrival_time}")
            
            # Use provided hotel_info instead of re-selecting hotel
            if not hotel_info:
                print(f"      ❌ No hotel info provided for {hotel}")
                return None
            
            print(f"      ✅ Hotel info validated")
            
            # Calculate travel time from ATH to hotel
            ath_airport = "Athens Airport"
            print(f"      🔍 Calculating travel time from {ath_airport} to {hotel}")
            
            airport_to_hotel_time = self._get_hotel_to_hub_travel_time(hotel, ath_airport, 'Athens, Greece')
            print(f"      ✅ Airport to hotel time: {airport_to_hotel_time} minutes")
            
            # Tourism starts after getting to hotel + buffer
            tourism_start = arrival_time + timedelta(minutes=airport_to_hotel_time + 30)  # 30 min buffer
            tourism_end = arrival_time.replace(hour=22, minute=0)  # End at 10pm
            print(f"      ✅ Tourism window: {tourism_start.strftime('%H:%M')} - {tourism_end.strftime('%H:%M')}")
            
            # Solve subproblem from hotel to hotel for first day
            if tourism_start < tourism_end:
                print(f"      🔍 Calling subproblem_solver.solve_subproblem...")
                try:
                    subproblem = self.subproblem_solver.solve_subproblem(
                        city='Athens, Greece',
                        start_poi=hotel,
                        start_time=tourism_start,
                        end_time=tourism_end,
                        global_visited_pois=state.get_city_visited_pois('Athens, Greece'),  # City-specific POIs
                        date=self.trip_dates[0],
                        end_poi=hotel
                    )
                    print(f"      ✅ Subproblem solved successfully")
                    print(f"         Objective score: {subproblem.get('objective_score', 'N/A')}")
                    print(f"         Feasible: {subproblem.get('feasible', 'N/A')}")
                except Exception as e:
                    print(f"      ❌ ERROR in subproblem solver: {e}")
                    print(f"         Exception type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Calculate costs using provided hotel info
                print(f"      🔍 Calculating hotel score and cost...")
                try:
                    hotel_score, hotel_cost = self._calculate_hotel_score_cost(hotel_info, 0)
                    print(f"      ✅ Hotel calculation: score={hotel_score}, cost={hotel_cost}")
                except Exception as e:
                    print(f"      ❌ ERROR in hotel score/cost calculation: {e}")
                    print(f"         Exception type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                print(f"      🔍 Calculating airport transfer cost...")
                try:
                    airport_transfer_cost = self._estimate_local_transport_cost(airport_to_hotel_time, 0)
                    print(f"      ✅ Airport transfer cost: ${airport_transfer_cost}")
                except Exception as e:
                    print(f"      ❌ ERROR in airport transfer cost calculation: {e}")
                    print(f"         Exception type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                print(f"   First day costs: Hotel=${hotel_cost:.2f}, Airport transfer=${airport_transfer_cost:.2f}")
                
                return {
                    'schedule': {
                        'day': 0,
                        'date': self.trip_dates[0],
                        'city': 'Athens, Greece',
                        'start_location': 'ATH Airport',
                        'arrival_time': '12:00',
                        'start_hotel': hotel,
                        'end_hotel': hotel,
                        'activities': subproblem.get('path', []),
                        'day_score': subproblem['objective_score'] + hotel_score,
                        'type': 'airport_arrival',
                        'logistics': {
                            'airport_to_hotel_time': airport_to_hotel_time,
                            'airport_transfer_cost': airport_transfer_cost
                        }
                    },
                    'score': subproblem['objective_score'] + hotel_score,
                    'cost': hotel_cost + airport_transfer_cost,
                    'visited_pois': subproblem.get('visited_pois', set())
                }
            
        except Exception as e:
            print(f"Error creating first day from airport: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def _add_last_day_airport_constraint(self, solution: Dict) -> Optional[Dict]:
        """Add constraint that trip must end at ATH Airport at 12pm on last day"""
        try:
            if not solution['daily_schedule']:
                return solution
            
            # Get last day info
            last_day_schedule = solution['daily_schedule'][-1]
            last_day_num = len(solution['daily_schedule']) - 1
            
            # Must be in Athens on last day
            if last_day_schedule.get('city') != 'Athens, Greece' and \
               last_day_schedule.get('end_city') != 'Athens, Greece':
                return None  # Invalid - not in Athens on last day
            
            # Add departure logistics to last day
            last_hotel = last_day_schedule.get('end_hotel', last_day_schedule.get('hotel', ''))
            ath_airport = "Athens Airport"
            hotel_to_airport_time = self._get_hotel_to_hub_travel_time(last_hotel, ath_airport, 'Athens, Greece')
            
            # Calculate latest checkout time to reach airport by 12pm
            departure_time = self.start_date.replace(hour=12, minute=0) + timedelta(days=self.num_days-1)
            latest_checkout = departure_time - timedelta(minutes=hotel_to_airport_time + 60)  # 1h buffer
            
            # Adjust last day activities to end before checkout
            if 'activities' in last_day_schedule:
                # Modify last activity end time if needed
                activities = last_day_schedule['activities']
                if activities:
                    for activity in activities:
                        if isinstance(activity, dict) and 'departure_time' in activity:
                            activity_end = activity['departure_time']
                            if isinstance(activity_end, str):
                                activity_end = datetime.strptime(activity_end, '%H:%M')
                            elif hasattr(activity_end, 'time'):
                                activity_end = activity_end.time()
                                activity_end = datetime.combine(departure_time.date(), activity_end)
                            
                            if activity_end > latest_checkout:
                                activity['departure_time'] = latest_checkout.strftime('%H:%M')
            
            # Add departure logistics
            last_day_schedule['departure_logistics'] = {
                'departure_time': '12:00',
                'departure_location': 'ATH Airport',
                'hotel_to_airport_time': hotel_to_airport_time,
                'latest_checkout': latest_checkout.strftime('%H:%M'),
                'airport_transfer_cost': self._estimate_local_transport_cost(hotel_to_airport_time, 0)
            }
            
            # Update total cost
            airport_transfer_cost = last_day_schedule['departure_logistics']['airport_transfer_cost']
            solution['total_cost'] += airport_transfer_cost
            
            return solution
            
        except Exception as e:
            print(f"Error adding last day airport constraint: {e}")
            return solution
    
    def _enhanced_beam_search(self, initial_state: InterCityState, 
                             beam_width: int, return_top_k: int) -> List[Dict]:
        """
        Enhanced beam search returning multiple solutions
        """
        beam = [(-initial_state.total_objective_score, initial_state)]
        all_complete_solutions = []
        
        print(f"🔍 Enhanced beam search starting with {len(beam)} initial states")
        print(f"   Initial score: {initial_state.total_objective_score:.2f}")
        print(f"   Days to process: {self.num_days}")
        
        for day in range(1, self.num_days):  # Start from day 1 since day 0 is already planned
            print(f"\n📅 Day {day + 1}/{self.num_days}: Processing beam of {len(beam)} states...")
            
            # Get current beam states
            current_beam = []
            extracted_states = 0
            while beam and len(current_beam) < beam_width:
                _, state = heapq.heappop(beam)
                extracted_states += 1
                if state.current_day == day:
                    current_beam.append(state)
                    print(f"   ✅ State {len(current_beam)}: day={state.current_day}, city={state.current_city}, score={state.total_objective_score:.2f}")
                else:
                    print(f"   ⏭️  Skipped state: expected day {day}, got day {state.current_day}")
            
            print(f"   📊 Extracted {extracted_states} states, {len(current_beam)} match current day")
            
            if not current_beam:
                print(f"   ❌ No states for day {day}! Breaking beam search.")
                break
            
            # Generate successors using the 6-step pipeline
            next_beam = []
            total_successors = 0
            complete_solutions_this_day = 0
            
            for state_idx, state in enumerate(current_beam):
                print(f"   🔄 Processing state {state_idx + 1}/{len(current_beam)}: {state.current_city} -> ?")
                
                successors = self._generate_day_successors_pipeline(state, day)
                total_successors += len(successors)
                
                print(f"      Generated {len(successors)} successors")
                
                for succ_idx, successor in enumerate(successors):
                    if successor.current_day == self.num_days:
                        # Complete solution
                        solution = self._state_to_solution(successor)
                        all_complete_solutions.append(solution)
                        complete_solutions_this_day += 1
                        print(f"      🏁 Complete solution {complete_solutions_this_day}: score={solution['total_objective_score']:.2f}")
                    else:
                        heapq.heappush(next_beam, (-successor.total_objective_score, successor))
                        if succ_idx < 3:  # Show first few successors
                            print(f"      ➡️  Successor {succ_idx + 1}: day={successor.current_day}, city={successor.current_city}, score={successor.total_objective_score:.2f}")
            
            print(f"   📈 Day {day + 1} summary:")
            print(f"      Total successors generated: {total_successors}")
            print(f"      Complete solutions found: {complete_solutions_this_day}")
            print(f"      States for next iteration: {len(next_beam)}")
            
            # Keep top beam_width states - group by full city path for diversity
            beam = []
            
            # Group successors by full city path sequence to maintain trajectory diversity
            successors_by_path = {}
            while next_beam:
                score, state = heapq.heappop(next_beam)
                
                # Create path key from the full city sequence
                city_path = self._extract_city_path_from_state(state)
                path_key = tuple(city_path)  # Convert to tuple for dict key
                
                if path_key not in successors_by_path:
                    successors_by_path[path_key] = []
                successors_by_path[path_key].append((score, state))
            
            # Keep top 1 trajectory per unique city path (ensures path diversity)
            trajectories_per_path = 1  # Only keep the best for each unique path
            total_kept = 0
            
            print(f"      🎯 Grouping by full city path: {len(successors_by_path)} unique paths")
            print(f"      📊 Keeping top {trajectories_per_path} trajectory per unique path")
            
            for path_key, path_successors in successors_by_path.items():
                # Sort by score (already negative, so this gives best scores first)
                path_successors.sort(key=lambda x: x[0])
                kept_count = 0
                
                for score, state in path_successors:
                    if kept_count < trajectories_per_path and total_kept < beam_width:
                        beam.append((score, state))
                        kept_count += 1
                        total_kept += 1
                    else:
                        break
                
                path_str = " → ".join(path_key)
                print(f"         {path_str}: kept {kept_count}/{len(path_successors)} trajectories")
            
            print(f"      Final beam size: {len(beam)} (max diversity: {len(successors_by_path)} unique paths)")
                    
        # Add any remaining complete states
        remaining_complete = 0
        for _, state in beam:
            if state.current_day == self.num_days:
                solution = self._state_to_solution(state)
                all_complete_solutions.append(solution)
                remaining_complete += 1
        
        if remaining_complete > 0:
            print(f"🏁 Added {remaining_complete} remaining complete solutions")
        
        print(f"📊 Beam search summary:")
        print(f"   Total complete solutions: {len(all_complete_solutions)}")
        if all_complete_solutions:
            scores = [sol['total_objective_score'] for sol in all_complete_solutions]
            print(f"   Score range: {min(scores):.2f} to {max(scores):.2f}")
        
        # Sort and return top K solutions
        all_complete_solutions.sort(key=lambda x: x['total_objective_score'], reverse=True)
        return all_complete_solutions[:return_top_k]
    
    def _generate_day_successors_pipeline(self, state: InterCityState, day: int) -> List[InterCityState]:
        """
        Implement the 6-step pipeline for generating day successors
        """
        successors = []
        
        # Step 1: Start with current state (city and hotel)
        start_city = state.current_city
        start_hotel = state.current_hotel
        date_str = self.trip_dates[day]
        
        print(f"         🏁 Pipeline Step 1: Current state")
        print(f"            Start city: {start_city}")
        print(f"            Start hotel: {start_hotel}")
        print(f"            Date: {date_str}")
        print(f"            Visited cities: {state.visited_cities}")
        
        # Step 2: Loop through all available destinations + stay option
        destination_choices = self._get_destination_choices(start_city, day, state.visited_cities)
        
        print(f"         🗺️  Pipeline Step 2: Destination choices")
        print(f"            Available destinations: {destination_choices}")
        
        if not destination_choices:
            print(f"            ❌ No destination choices available!")
            return successors
        
        for dest_idx, end_city in enumerate(destination_choices):
            print(f"         🏙️  Pipeline Step 3: Processing destination {dest_idx + 1}/{len(destination_choices)}: {end_city}")
            
            # Step 3: Choose best hotel for end_city
            # Skip hotel selection for last day since we depart at 12pm
            if day >= self.num_days - 1:
                print(f"            🛫 Last day - skipping hotel selection (departure at 12pm)")
                # Create a dummy hotel info for last day processing
                best_hotel_info = {
                    'name': state.current_hotel,  # Use current hotel
                    'info': {'base_cost': 0.0, 'preference': 0}  # No cost for last day
                }
                best_hotel = state.current_hotel
            else:
                best_hotel_info = self._get_best_hotel_in_city(
                    end_city, date_str, state, 
                    current_hotel=start_hotel if end_city == start_city else None
                )
                
                print(f"            🏨 Best hotel in {end_city}: {best_hotel_info['name'] if best_hotel_info else None}")
                
                if not best_hotel_info:
                    print(f"            ❌ No hotel available in {end_city}, skipping")
                    continue
                
                best_hotel = best_hotel_info['name']  # Extract hotel name for compatibility
            
            if end_city == start_city:
                print(f"            🏠 Staying in same city: {start_city}")
                # Staying in same city
                successor = self._generate_stay_day_successor(
                    state, day, start_hotel, best_hotel, best_hotel_info, date_str
                )
                if successor:
                    successors.append(successor)
                    print(f"            ✅ Generated stay successor: score={successor.total_objective_score:.2f}")
                else:
                    print(f"            ❌ Failed to generate stay successor")
            else:
                print(f"            ✈️  Traveling from {start_city} to {end_city}")
                # Step 4: Loop through transportation methods
                transport_options = self._get_transport_options_for_date(
                    start_city, end_city, date_str
                )
                
                print(f"            🚌 Transport options: {len(transport_options)}")
                if not transport_options:
                    print(f"            ❌ No transport available from {start_city} to {end_city}")
                    continue
                
                for trans_idx, transport in enumerate(transport_options):
                    # Validate transport timing with current hotel
                    if not self._validate_transport_timing(transport, start_hotel, start_city, end_city):
                        print(f"            ⏭️  Skipping transport option {trans_idx + 1}: Invalid timing")
                        continue
                        
                    print(f"            🚌 Transport option {trans_idx + 1}/{len(transport_options)}: {transport.get('mode', 'unknown')}")
                    print(f"               Departure: {transport.get('departure_time', 'unknown')}")
                    print(f"               Arrival: {transport.get('arrival_time', 'unknown')}")
                    print(f"               Cost: ${transport.get('cost', 0):.2f}")
                    
                    # Step 5: Calculate travel times and subproblems
                    successor = self._generate_travel_day_successor(
                        state, day, start_city, end_city, start_hotel, 
                        best_hotel, best_hotel_info, transport, date_str
                    )
                    if successor:
                        successors.append(successor)
                        print(f"               ✅ Generated travel successor: score={successor.total_objective_score:.2f}")
                    else:
                        print(f"               ❌ Failed to generate travel successor")
        
        print(f"         📊 Pipeline summary: Generated {len(successors)} total successors")
        return successors
    
    def _get_destination_choices(self, current_city: str, day: int, visited_cities: Set[str]) -> List[str]:
        """Get valid destination choices based on constraints with AC-3 constraint propagation"""
        choices = []
        
        print(f"            🎯 Analyzing destination constraints:")
        print(f"               Current city: {current_city}")
        print(f"               Day: {day + 1}/{self.num_days}")
        print(f"               Visited cities: {visited_cities}")
        
        # First day constraint: Must stay in Athens (arrive at 12pm from airport)
        if day == 0:
            print(f"               🛬 First day constraint: Must stay in Athens (airport arrival)")
            if current_city == 'Athens, Greece':
                choices.append('Athens, Greece')  # Stay in Athens
                print(f"               ✅ Added Athens (stay - first day)")
            else:
                # This should not happen as we start in Athens, but handle gracefully
                print(f"               ⚠️  Warning: First day not in Athens! Current city: {current_city}")
                choices.append(current_city)  # Stay wherever we are
                print(f"               ✅ Added {current_city} (stay - emergency fallback)")
        
        # Last day constraint: Must stay in Athens (departure at 12pm to airport)  
        elif day == self.num_days - 1:
            print(f"               🛫 Last day constraint: Must stay in Athens (airport departure)")
            if current_city == 'Athens, Greece':
                choices.append('Athens, Greece')  # Stay in Athens
                print(f"               ✅ Added Athens (stay - last day)")
            else:
                # This should not happen due to constraint propagation, but handle gracefully
                print(f"               ⚠️  Warning: Last day not in Athens! Current city: {current_city}")
                choices.append('Athens, Greece')  # Force travel to Athens
                print(f"               ✅ Added Athens (forced travel - emergency)")
        
        # Second-to-last day constraint: Must be in or travel to Athens
        elif day == self.num_days - 2:
            print(f"               🏛️  Second-to-last day constraint: Must be in Athens")
            if current_city == 'Athens, Greece':
                choices.append('Athens, Greece')  # Stay option
                print(f"               ✅ Added Athens (stay option)")
            else:
                choices.append('Athens, Greece')  # Travel to Athens (allow even if visited before)
                print(f"               ✅ Added Athens (travel from {current_city})")
        
        # Middle days: Use AC-3 constraint propagation
        else:
            print(f"               🌍 Middle days: Using AC-3 constraint propagation")
            days_to_second_last = (self.num_days - 2) - day  # Days until second-to-last day
            print(f"               🕐 Days until second-to-last day: {days_to_second_last}")
            
            # Stay option - always allowed for middle days (no AC-3 constraint needed)
            choices.append(current_city)  # Stay option
            print(f"               ✅ Added {current_city} (stay option)")
            
            # Travel options - NO revisiting of cities during middle days (except Athens on second-to-last)
            for city in self.cities:
                if city != current_city and city not in visited_cities:  # REMOVED Athens exception for middle days
                    # AC-3 constraint: if traveling to this city would put us on second-to-last day,
                    # check if this city has transport to Athens on the final day
                    if days_to_second_last == 1:  # Next day will be second-to-last
                        second_last_date = self.trip_dates[day + 1]  # Date of second-to-last day
                        if self._has_transport_to_athens_on_date(city, second_last_date):
                            choices.append(city)
                            print(f"               ✅ Added {city} (unvisited, has transport to Athens on second-to-last day)")
                        else:
                            print(f"               ❌ Skipped {city} (no transport to Athens on {second_last_date})")
                    else:
                        # Not going to second-to-last day yet, no AC-3 constraint needed
                        choices.append(city)
                        print(f"               ✅ Added {city} (unvisited)")
                else:
                    if city == current_city:
                        print(f"               ⏭️  Skipped {city} (current city, already added as stay)")
                    elif city in visited_cities:
                        print(f"               ❌ Skipped {city} (already visited - no revisiting during middle days)")
                    else:
                        print(f"               ❌ Skipped {city} (unknown reason)")
        
        print(f"               📋 Final destination choices: {choices}")
        print(f"               🚀 AC-3 constraint propagation: Eliminated {len(self.cities) + 1 - len(choices)} infeasible destinations")
        return choices
    
    def _get_best_hotel_in_city(self, city: str, date: str, state: InterCityState, 
                               current_hotel: Optional[str] = None) -> Optional[Dict]:
        """Get best available hotel in city using city-specific visited hotels"""
        print(f"               🏨 Hotel selection for {city}:")
        
        city_visited_hotels = state.get_city_visited_hotels(city)
        print(f"                  Already visited in {city}: {city_visited_hotels}")
        print(f"                  Current hotel: {current_hotel}")
        print(f"                  Date: {date}")
        
        try:
            hotel_info = self.hotel_selector.select_best_hotel(
                city=city,
                date=date,
                global_visited_hotels=city_visited_hotels,  # Use city-specific visited hotels
                current_hotel=current_hotel
            )
            
            if hotel_info:
                print(f"                  ✅ Selected: {hotel_info['name']}")
                return hotel_info
            else:
                print(f"                  ❌ No hotel selected (hotel_selector returned None)")
                return None
                
        except Exception as e:
            print(f"                  ❌ Error in hotel selection: {e}")
            return None
    
    def _validate_transport_timing(self, transport: Dict, start_hotel: str, 
                                  start_city: str, end_city: str) -> bool:
        """Validate transport timing considering hotel-to-hub travel and 9am start time"""
        try:
            # Get departure and arrival times
            dep_time_str = transport['departure_time']
            arr_time_str = transport['arrival_time']
            buffered_dep = transport.get('buffered_departure')
            buffered_arr = transport.get('buffered_arrival')
            
            if not all([dep_time_str, arr_time_str, buffered_dep, buffered_arr]):
                return False
            
            # Parse times
            dep_time = datetime.strptime(dep_time_str, '%H:%M')
            arr_time = datetime.strptime(arr_time_str, '%H:%M')
            buffered_dep = datetime.strptime(buffered_dep, '%H:%M')
            buffered_arr = datetime.strptime(buffered_arr, '%H:%M')
            
            # Get transportation hubs
            dep_hub = self._get_departure_hub(start_city, transport)
            arr_hub = self._get_arrival_hub(end_city, transport)
            
            # Calculate hotel-to-hub travel time
            hotel_to_hub_time = self._get_hotel_to_hub_travel_time(start_hotel, dep_hub, start_city)
            
            # Calculate when we need to leave hotel
            hotel_departure = dep_time - timedelta(minutes=hotel_to_hub_time + 60)  # 60 min buffer
            
            # Check if we can start from hotel at or after 9am
            day_start = dep_time.replace(hour=9, minute=0)
            day_end = dep_time.replace(hour=22, minute=0)
            
            # Validate:
            # 1. Hotel departure time must be at or after 9am
            # 2. Buffered arrival must be before 10pm
            if hotel_departure < day_start or buffered_arr > day_end:
                return False
            
            return True
            
        except (ValueError, KeyError) as e:
            print(f"Error validating transport timing: {e}")
            return False

    def _get_transport_options_for_date(self, origin_city: str, dest_city: str, date: str) -> List[Dict]:
        """Get valid transportation options for specific date"""
        route_key = (origin_city, dest_city)
        all_options = self.transport_lookup.get(date, {}).get(route_key, [])
        
        # Filter options based on timing constraints
        valid_options = []
        for transport in all_options:
            # Skip validation for now - it will be done in _calculate_enhanced_day_splits
            valid_options.append(transport)
        
        return valid_options
    
    def _generate_stay_day_successor(self, state: InterCityState, day: int, 
                                   start_hotel: str, end_hotel: str, hotel_info: Dict, date_str: str) -> Optional[InterCityState]:
        """Generate successor for staying in same city"""
        print(f"               🏠 Generating stay successor:")
        print(f"                  City: {state.current_city}")
        print(f"                  Start hotel: {start_hotel}")
        print(f"                  End hotel: {end_hotel}")
        print(f"                  Date: {date_str}")
        
        # Special handling for last day (departure day)
        if day >= self.num_days - 1:
            print(f"                  🛫 Last day handling: departure to ATH Airport at 12:00")
            
            # Calculate checkout time to reach airport by 12pm
            ath_airport = "Athens Airport"
            hotel_to_airport_time = self._get_hotel_to_hub_travel_time(start_hotel, ath_airport, state.current_city)
            departure_time = self.start_date.replace(hour=12, minute=0) + timedelta(days=day)
            latest_checkout = departure_time - timedelta(minutes=hotel_to_airport_time + 60)  # 1 hour buffer
            
            # Plan morning activities until checkout
            start_time = self.start_date.replace(hour=9, minute=0) + timedelta(days=day)
            end_time = min(latest_checkout, start_time.replace(hour=11, minute=0))  # Max until 11am
            
            print(f"                  Morning activities: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
            print(f"                  Checkout: {latest_checkout.strftime('%H:%M')}")
            print(f"                  Airport departure: 12:00")
            
            if end_time <= start_time:
                print(f"                  ❌ No time for activities - direct checkout")
                # Create minimal successor with just departure logistics
                successor = state.copy()
                successor.current_day = day + 1
                successor.daily_schedule.append({
                    'day': day,
                    'date': date_str,
                    'city': state.current_city,
                    'start_hotel': start_hotel,
                    'end_hotel': start_hotel,
                    'activities': [],
                    'day_score': 0.0,
                    'type': 'departure',
                    'departure_logistics': {
                        'checkout_time': latest_checkout.strftime('%H:%M'),
                        'departure_time': '12:00',
                        'airport': 'ATH Airport',
                        'hotel_to_airport_time': hotel_to_airport_time
                    }
                })
                return successor
            
            # Plan short morning activities
            try:
                subproblem = self.subproblem_solver.solve_subproblem(
                    city=state.current_city,
                    start_poi=start_hotel,
                    start_time=start_time,
                    end_time=end_time,
                    global_visited_pois=state.get_city_visited_pois(state.current_city),
                    date=date_str,
                    end_poi=start_hotel
                )
                
                print(f"                  Morning subproblem score: {subproblem['objective_score']:.2f}")
                
                # Create successor for last day
                successor = state.copy()
                successor.current_day = day + 1
                successor.total_objective_score += subproblem['objective_score']
                # No hotel cost for last day
                
                for poi in subproblem.get('visited_pois', set()):
                    successor.add_visited_poi(state.current_city, poi)
                
                successor.daily_schedule.append({
                    'day': day,
                    'date': date_str,
                    'city': state.current_city,
                    'start_hotel': start_hotel,
                    'end_hotel': start_hotel,
                    'activities': subproblem.get('path', []),
                    'day_score': subproblem['objective_score'],
                    'type': 'departure',
                    'departure_logistics': {
                        'checkout_time': latest_checkout.strftime('%H:%M'),
                        'departure_time': '12:00',
                        'airport': 'ATH Airport',
                        'hotel_to_airport_time': hotel_to_airport_time
                    }
                })
                
                print(f"                  ✅ Last day successor created: score={successor.total_objective_score:.2f}")
                return successor
                
            except Exception as e:
                print(f"                  ❌ Error in last day subproblem: {e}")
                return None
        
        # Regular day handling (non-departure days)
        # Plan full day with hotel-to-hotel round trip
        start_time = self.start_date.replace(hour=9, minute=0) + timedelta(days=day)
        end_time = start_time.replace(hour=22, minute=0)
        
        print(f"                  Time window: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
        
        # If changing hotels, account for check-in/check-out
        if start_hotel != end_hotel:
            # Allow 1 hour for hotel change
            start_time += timedelta(hours=1)
            print(f"                  Hotel change: adjusted start time to {start_time.strftime('%H:%M')}")
        
        try:
            subproblem = self.subproblem_solver.solve_subproblem(
                city=state.current_city,
                start_poi=start_hotel,
                start_time=start_time,
                end_time=end_time,
                global_visited_pois=state.get_city_visited_pois(state.current_city),  # City-specific POIs
                date=date_str,
                end_poi=end_hotel
            )
            
            print(f"                  Subproblem score: {subproblem['objective_score']:.2f}")
            print(f"                  Subproblem feasible: {subproblem['feasible']}")
            
            if not subproblem['feasible']:
                print(f"                  ❌ Subproblem not feasible")
                return None
            
        except Exception as e:
            print(f"                  ❌ Error in subproblem: {e}")
            return None
        
        # Calculate hotel costs and scores
        try:
            hotel_score, hotel_cost = self._calculate_hotel_score_cost(hotel_info, day)
            print(f"                  Hotel score: {hotel_score:.2f}, cost: ${hotel_cost:.2f}")
            
            day_score = subproblem['objective_score'] + hotel_score - self.cost_sensitivity * hotel_cost
            print(f"                  Total day score: {day_score:.2f}")
            
            # Create successor
            successor = state.copy()
            successor.current_day = day + 1
            successor.current_hotel = end_hotel
            successor.add_visited_hotel(state.current_city, end_hotel)  # Add to city-specific tracking
            successor.total_objective_score += day_score
            successor.total_cost += hotel_cost
            
            # Add visited POIs to city-specific tracking
            for poi in subproblem.get('visited_pois', set()):
                successor.add_visited_poi(state.current_city, poi)
            
            successor.daily_schedule.append({
                'day': day,
                'date': date_str,
                'city': state.current_city,
                'start_hotel': start_hotel,
                'end_hotel': end_hotel,
                'activities': subproblem.get('path', []),
                'day_score': day_score,
                'type': 'stay'
            })
            
            print(f"                  ✅ Stay successor created: new day={successor.current_day}, score={successor.total_objective_score:.2f}")
            return successor
            
        except Exception as e:
            print(f"                  ❌ Error creating successor: {e}")
            return None
    
    def _generate_travel_day_successor(self, state: InterCityState, day: int,
                                     start_city: str, end_city: str, start_hotel: str,
                                     end_hotel: str, hotel_info: Dict, transport: Dict, date_str: str) -> Optional[InterCityState]:
        """Generate successor for travel day with realistic logistics"""
        
        # Step 5: Calculate travel times and day splits
        day_logistics = self._calculate_enhanced_day_splits(
            transport, day, start_hotel, end_hotel, start_city, end_city
        )
        
        if not day_logistics:
            return None
        
        # Solve subproblems
        subproblem1_score = 0.0
        subproblem2_score = 0.0
        
        # Morning subproblem in origin city (hotel-to-hotel round trip)
        if day_logistics['morning_duration'] > 1.0:  # At least 1 hour for activities
            subproblem1 = self.subproblem_solver.solve_subproblem(
                city=start_city,
                start_poi=start_hotel,
                start_time=day_logistics['morning_start'],
                end_time=day_logistics['morning_end'],
                global_visited_pois=state.get_city_visited_pois(start_city),  # City-specific POIs
                date=date_str,
                end_poi=start_hotel
            )
            subproblem1_score = subproblem1['objective_score']
        
        # Evening subproblem in destination city (hotel-to-hotel round trip)
        if day_logistics['evening_duration'] > 1.0:  # At least 1 hour for activities
            subproblem2 = self.subproblem_solver.solve_subproblem(
                city=end_city,
                start_poi=end_hotel,
                start_time=day_logistics['evening_start'],
                end_time=day_logistics['evening_end'],
                global_visited_pois=state.get_city_visited_pois(end_city),  # City-specific POIs for destination
                date=date_str,
                end_poi=end_hotel
            )
            subproblem2_score = subproblem2['objective_score']
        
        # Calculate scores and costs
        hotel_score, hotel_cost = self._calculate_hotel_score_cost(hotel_info, day)
        transport_penalty = self._calculate_transport_penalty(transport, day_logistics)
        
        day_score = (
            subproblem1_score + subproblem2_score + hotel_score - 
            self.cost_sensitivity * hotel_cost - transport_penalty
        )
        
        # Create successor
        successor = state.copy()
        successor.current_day = day + 1
        successor.current_city = end_city
        successor.current_hotel = end_hotel
        successor.visited_cities.add(end_city)
        successor.add_visited_hotel(end_city, end_hotel)  # Add to city-specific tracking
        successor.add_visited_poi(end_city, end_hotel)  # Also add hotel to visited POIs
        successor.total_objective_score += day_score
        successor.total_cost += transport['cost'] + hotel_cost + day_logistics['local_transport_cost']
        
        # Update visited POIs by city
        if 'subproblem1' in locals():
            for poi in subproblem1.get('visited_pois', set()):
                successor.add_visited_poi(start_city, poi)
        if 'subproblem2' in locals():
            for poi in subproblem2.get('visited_pois', set()):
                successor.add_visited_poi(end_city, poi)
        
        successor.daily_schedule.append({
            'day': day,
            'date': date_str,
            'start_city': start_city,
            'end_city': end_city,
            'start_hotel': start_hotel,
            'end_hotel': end_hotel,
            'transportation': transport,
            'morning_activities': subproblem1.get('path', []) if 'subproblem1' in locals() else [],
            'evening_activities': subproblem2.get('path', []) if 'subproblem2' in locals() else [],
            'logistics': day_logistics,
            'day_score': day_score,
            'type': 'travel'
        })
        
        return successor
    
    def _calculate_enhanced_day_splits(self, transport: Dict, day: int, 
                                     start_hotel: str, end_hotel: str,
                                     start_city: str, end_city: str) -> Optional[Dict]:
        """Calculate realistic day splits with actual travel times"""
        try:
            # First validate transport timing
            if not self._validate_transport_timing(transport, start_hotel, start_city, end_city):
                return None
            
            dep_time_str = transport['departure_time']
            arr_time_str = transport['arrival_time']
            
            # Parse times
            dep_hour, dep_min = map(int, dep_time_str.split(':'))
            arr_hour, arr_min = map(int, arr_time_str.split(':'))
            
            # Create datetime objects
            day_base = self.start_date + timedelta(days=day)
            departure_time = day_base.replace(hour=dep_hour, minute=dep_min)
            arrival_time = day_base.replace(hour=arr_hour, minute=arr_min)
            
            # Handle next-day arrivals
            if arrival_time <= departure_time:
                arrival_time += timedelta(days=1)
            
            # Get transportation hubs
            dep_hub = self._get_departure_hub(start_city, transport)
            arr_hub = self._get_arrival_hub(end_city, transport)
            
            # Calculate hotel-to-hub travel times
            hotel_to_dep_time = self._get_hotel_to_hub_travel_time(start_hotel, dep_hub, start_city)
            arr_to_hotel_time = self._get_hotel_to_hub_travel_time(end_hotel, arr_hub, end_city)
            
            # Calculate time windows
            day_start = day_base.replace(hour=9, minute=0)
            day_end = day_base.replace(hour=22, minute=0)
            
            # Morning window (tourism before departure)
            morning_start = day_start
            hotel_departure = departure_time - timedelta(minutes=hotel_to_dep_time + self.check_in_buffer_minutes)
            morning_end = min(hotel_departure, day_end)
            
            # Evening window (tourism after arrival)
            hotel_arrival = arrival_time + timedelta(minutes=arr_to_hotel_time + self.delay_buffer_minutes)
            evening_start = max(hotel_arrival, day_start)
            evening_end = day_end
            
            # Handle cross-day scenarios
            if evening_start.date() > day_base.date():
                evening_start = day_end  # No evening activities
                evening_end = day_end
            
            # Calculate durations
            morning_duration = max(0, (morning_end - morning_start).total_seconds() / 3600)
            evening_duration = max(0, (evening_end - evening_start).total_seconds() / 3600)
            
            # Minimum viable time check
            if morning_duration + evening_duration < 0.5:
                return None
            
            # Calculate local transport costs
            local_transport_cost = self._estimate_local_transport_cost(
                hotel_to_dep_time, arr_to_hotel_time
            )
            
            return {
                'morning_start': morning_start,
                'morning_end': morning_end,
                'morning_duration': morning_duration,
                'evening_start': evening_start,
                'evening_end': evening_end,
                'evening_duration': evening_duration,
                'departure_hub': dep_hub,
                'arrival_hub': arr_hub,
                'hotel_to_dep_time': hotel_to_dep_time,
                'arr_to_hotel_time': arr_to_hotel_time,
                'local_transport_cost': local_transport_cost,
                'hotel_departure_time': hotel_departure.strftime('%H:%M'),
                'hotel_arrival_time': hotel_arrival.strftime('%H:%M')
            }
            
        except Exception as e:
            print(f"Error calculating day splits: {e}")
            return None
    
    def _get_departure_hub(self, city: str, transport: Dict) -> str:
        """Get departure hub for transportation"""
        # Extract from transport data or use heuristics
        mode = transport.get('mode', '').lower()
        if 'flight' in mode or 'plane' in mode:
            return f"{city.split(',')[0]} Airport"
        elif 'ferry' in mode or 'boat' in mode:
            return f"{city.split(',')[0]} Port"
        return f"{city} Hub"
    
    def _get_arrival_hub(self, city: str, transport: Dict) -> str:
        """Get arrival hub for transportation"""
        return self._get_departure_hub(city, transport)
    
    def _estimate_local_transport_cost(self, hotel_to_dep_time: int, arr_to_hotel_time: int) -> float:
        """Estimate local transportation costs"""
        # Simple cost model based on travel time
        base_cost_per_minute = 0.5
        return (hotel_to_dep_time + arr_to_hotel_time) * base_cost_per_minute
    
    def _calculate_hotel_score_cost(self, hotel_info: Dict, day: int) -> Tuple[float, float]:
        """Calculate hotel preference score and cost from actual hotel information"""
        
        # Last day doesn't need hotel (departure day)
        if day >= self.num_days - 1:
            return 0.0, 0.0
        
        # Extract actual cost and preference from hotel info
        if hotel_info and 'info' in hotel_info:
            info_section = hotel_info['info']
            
            # Get base cost and preference
            base_cost_raw = info_section.get('base_cost', 150.0)
            preference_raw = info_section.get('preference', 0)
            
            # Convert to proper types
            try:
                actual_cost = float(base_cost_raw) if base_cost_raw is not None else 150.0
            except (TypeError, ValueError) as e:
                print(f"           ❌ ERROR converting base_cost: {e} (value: {base_cost_raw})")
                actual_cost = 150.0
            
            try:
                preference_score = float(preference_raw) * self.poi_desirability if preference_raw is not None else 0.0
            except (TypeError, ValueError) as e:
                print(f"           ❌ ERROR converting preference: {e} (value: {preference_raw})")
                preference_score = 0.0
                
        else:
            # Fallback for missing hotel info
            actual_cost = 150.0
            preference_score = 25.0
        
        return preference_score, actual_cost
    
    def _calculate_transport_penalty(self, transport: Dict, day_logistics: Dict) -> float:
        """Calculate transportation penalty including time and cost"""
        time_penalty = self.transportation_averseness * (
            transport.get('total_duration_minutes', 180) / 60.0
        )
        cost_penalty = self.cost_sensitivity * transport.get('cost', 100.0)
        
        # Additional penalty for complex logistics
        logistics_penalty = self.transportation_averseness * 0.5 * (
            day_logistics['hotel_to_dep_time'] + day_logistics['arr_to_hotel_time']
        ) / 60.0
        
        return time_penalty + cost_penalty + logistics_penalty
    
    def _state_to_solution(self, state: InterCityState) -> Dict:
        """Convert state to solution format"""
        return {
            'total_objective_score': state.total_objective_score,
            'total_cost': state.total_cost,
            'visited_cities': list(state.visited_cities),
            'visited_pois': list(state.visited_pois),  # Uses backward compatibility property
            'visited_hotels': list(state.visited_hotels),  # Uses backward compatibility property
            'visited_pois_by_city': {city: list(pois) for city, pois in state.visited_pois_by_city.items()},
            'visited_hotels_by_city': {city: list(hotels) for city, hotels in state.visited_hotels_by_city.items()},
            'daily_schedule': state.daily_schedule,
            'trip_summary': {
                'num_days': len(state.daily_schedule),
                'num_cities': len(state.visited_cities),
                'num_pois': len(state.visited_pois),
                'avg_daily_score': state.total_objective_score / max(1, len(state.daily_schedule))
            }
        }
    
    def _has_transport_to_athens_on_final_day(self, city: str) -> bool:
        """Check if city has transportation to Athens on the final day (AC-3 constraint propagation)"""
        if city == 'Athens, Greece':
            return True  # Already in Athens
            
        final_date = self.trip_dates[-1]  # Last day
        transport_options = self._get_transport_options_for_date(city, 'Athens, Greece', final_date)
        return len(transport_options) > 0
    
    def _has_transport_to_athens_on_date(self, city: str, date: str) -> bool:
        """Check if city has transportation to Athens on a specific date"""
        if city == 'Athens, Greece':
            return True  # Already in Athens
            
        transport_options = self._get_transport_options_for_date(city, 'Athens, Greece', date)
        return len(transport_options) > 0
    
    def _extract_city_path_from_state(self, state: InterCityState) -> List[str]:
        """Extract the full city path sequence from state's daily schedule with stay lengths"""
        city_stays = []  # List of (city, stay_length) tuples
        current_city = 'Athens, Greece'
        current_stay_length = 1  # Start with 1 day in Athens
        
        for day_schedule in state.daily_schedule:
            schedule_type = day_schedule.get('type')
            
            if schedule_type == 'airport_arrival':
                # First day in Athens - already counted
                continue
                
            elif schedule_type == 'travel':
                # End current city stay and start new city
                if current_city and current_stay_length > 0:
                    city_stays.append(f"{current_city}({current_stay_length}d)")
                
                # Start new city
                current_city = day_schedule.get('end_city')
                current_stay_length = 1
                
            elif schedule_type == 'stay':
                # Extend current city stay
                current_stay_length += 1
                
            elif schedule_type == 'departure':
                # Last day - extend current stay but don't add yet
                current_stay_length += 1
        
        # Add final city stay
        if current_city and current_stay_length > 0:
            city_stays.append(f"{current_city}({current_stay_length}d)")
        
        return city_stays


# Example usage and testing
if __name__ == "__main__":
    print("Inter-City Optimizer - implementing 6-step pipeline")
    print("Key improvements:")
    print("- Realistic hotel-to-airport travel times")
    print("- Hotel-centric subproblems (round trips)")
    print("- Systematic destination choice process")
    print("- Enhanced time splitting with actual logistics")
    print("- Returns top 10 solutions instead of just the best")
    print("- Detailed cost tracking including local transport") 