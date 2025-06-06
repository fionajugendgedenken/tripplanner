"""
Enhanced Trip Planner - Main Interface
Integrates all components for sophisticated 2-layer trip planning
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from data_processor import EnhancedDataProcessor
from inter_city_optimizer import InterCityOptimizer
from intra_city_subproblem import IntracitySubproblemSolver
from hotel_selector import HotelSelector
from USER_PROFILE import POI_DESIRABILITY, COST_SENSITIVITY, TRANSPORTATION_AVERSENESS, START_DATE, END_DATE


class EnhancedTripPlanner:
    """
    Main trip planner that integrates the 2-layer optimization approach
    """
    
    def __init__(self, start_date: str = START_DATE, end_date: str = END_DATE,
                 poi_desirability: float = POI_DESIRABILITY,
                 cost_sensitivity: float = COST_SENSITIVITY, 
                 transportation_averseness: float = TRANSPORTATION_AVERSENESS):
        self.start_date = start_date
        self.end_date = end_date
        self.poi_desirability = poi_desirability
        self.cost_sensitivity = cost_sensitivity
        self.transportation_averseness = transportation_averseness
        
        # Initialize data processor
        self.data_processor = EnhancedDataProcessor(start_date, end_date)
        
        # Data storage
        self.poi_data = None
        self.poi_details = None
        self.routes = None
        self.transportation = None
        
        # Optimizer components
        self.inter_city_optimizer = None
        self.subproblem_solver = None
        self.hotel_selector = None
        
        print(f"Enhanced Trip Planner initialized for {start_date} to {end_date}")
        print(f"Objective weights: POI={poi_desirability}, Cost={cost_sensitivity}, Transport={transportation_averseness}")
    
    def load_all_data(self, poi_list_file: str = "POI_LIST.json", 
                     poi_results_file: str = "poi_results.json",
                     flights_file: str = "location_flights.json",
                     ferries_file: str = "ferry_results.json",
                     routes_file: str = "poi_routes.json") -> Dict:
        """
        Load and process all data files
        
        Returns:
            Dictionary with loading statistics
        """
        print("Loading and processing all data files...")
        start_time = time.time()
        
        # Load basic POI data
        print("1. Loading POI list...")
        with open(poi_list_file, 'r', encoding='utf-8') as f:
            self.poi_data = json.load(f)
        poi_count = sum(len(city_pois) for city_pois in self.poi_data.get('POI_LIST', {}).values())
        print(f"   Loaded {poi_count} POIs across {len(self.poi_data.get('POI_LIST', {}))} cities")
        
        # Load and process detailed POI data
        print("2. Processing detailed POI data...")
        self.poi_details = self.data_processor.load_and_process_poi_data(poi_results_file)
        
        # Load and filter transportation
        print("3. Loading and filtering transportation...")
        self.transportation = self.data_processor.load_and_filter_transportation(
            flights_file, ferries_file
        )
        
        # Load and process routes
        print("4. Processing POI routes...")
        routes_by_origin = self.data_processor.load_and_process_poi_routes(routes_file)
        
        # Organize routes by location (3-layer structure)
        print("5. Organizing routes by location...")
        self.routes = self.data_processor.organize_routes_by_location(routes_by_origin, self.poi_details)
        
        loading_time = time.time() - start_time
        
        # Calculate route count for 3-layer structure
        total_routes = sum(
            len(destinations)
            for location_dict in self.routes.values()
            for destinations in location_dict.values()
        )
        
        # Calculate transportation options count from nested structure
        total_transportation_options = sum(
            len(transport_options_list)
            for date_dict in self.transportation.values()
            for origin_dict in date_dict.values()
            for transport_options_list in origin_dict.values()
        )
        
        stats = {
            'loading_time_seconds': loading_time,
            'poi_count': poi_count,
            'cities_count': len(self.poi_data.get('POI_LIST', {})),
            'detailed_poi_locations': len(self.poi_details),
            'transportation_dates': len(self.transportation),
            'total_transportation_options': total_transportation_options,
            'route_locations': len(self.routes),
            'total_routes': total_routes
        }
        
        print(f"Data loading completed in {loading_time:.2f} seconds")
        print(f"Statistics: {stats}")
        
        return stats
    
    def initialize_optimizers(self) -> None:
        """Initialize all optimizer components"""
        if not all([self.poi_data, self.poi_details, self.routes, self.transportation]):
            raise ValueError("Must load data before initializing optimizers")
        
        print("Initializing optimizer components...")
        
        # Initialize inter-city optimizer (main optimizer)
        self.inter_city_optimizer = InterCityOptimizer(
            poi_data=self.poi_data,
            poi_details=self.poi_details,
            routes=self.routes,
            transportation=self.transportation,
            start_date=self.start_date,
            end_date=self.end_date,
            poi_desirability=self.poi_desirability,
            cost_sensitivity=self.cost_sensitivity,
            transportation_averseness=self.transportation_averseness
        )
        
        # Initialize standalone components for analysis
        self.subproblem_solver = IntracitySubproblemSolver(
            poi_data=self.poi_data,
            poi_details=self.poi_details,
            routes=self.routes,
            poi_desirability=self.poi_desirability,
            cost_sensitivity=self.cost_sensitivity,
            transportation_averseness=self.transportation_averseness
        )
        
        self.hotel_selector = HotelSelector(
            poi_data=self.poi_data,
            poi_details=self.poi_details,
            routes=self.routes,
            poi_desirability=self.poi_desirability,
            cost_sensitivity=self.cost_sensitivity,
            transportation_averseness=self.transportation_averseness
        )
        
        print("Optimizer components initialized successfully")
    
    def plan_trip(self, beam_width: int = 5) -> Dict:
        """
        Plan the complete trip using 2-layer optimization
        
        Args:
            beam_width: Beam search width for Layer 1
            
        Returns:
            Best trip plan with detailed schedule
        """
        if not self.inter_city_optimizer:
            raise ValueError("Must initialize optimizers before planning trip")
        
        print(f"Starting trip planning with beam width {beam_width}...")
        start_time = time.time()
        
        # Run inter-city optimization - get top solutions
        top_solutions = self.inter_city_optimizer.optimize_trip(beam_width=beam_width, return_top_k=10)
        
        planning_time = time.time() - start_time
        
        if not top_solutions:
            raise ValueError("No valid solutions found during optimization")
        
        # Get the best solution
        best_solution = top_solutions[0]
        
        # Enhance solution with additional details
        enhanced_solution = self._enhance_solution(best_solution, planning_time)
        
        print(f"Trip planning completed in {planning_time:.2f} seconds")
        print(f"Final objective score: {enhanced_solution['total_objective_score']:.2f}")
        print(f"Total cost: ${enhanced_solution['total_cost']:.2f}")
        print(f"Found {len(top_solutions)} alternative solutions")
        
        return enhanced_solution
    
    def plan_multiple_trips(self, beam_width: int = 5, return_top_k: int = 5) -> List[Dict]:
        """
        Plan multiple trip alternatives using 2-layer optimization
        
        Args:
            beam_width: Beam search width for Layer 1
            return_top_k: Number of alternative solutions to return
            
        Returns:
            List of trip plans with detailed schedules, sorted by objective score
        """
        if not self.inter_city_optimizer:
            raise ValueError("Must initialize optimizers before planning trip")
        
        print(f"Starting multi-trip planning with beam width {beam_width}...")
        start_time = time.time()
        
        # Run inter-city optimization - get top solutions
        top_solutions = self.inter_city_optimizer.optimize_trip(beam_width=beam_width, return_top_k=return_top_k)
        
        planning_time = time.time() - start_time
        
        if not top_solutions:
            raise ValueError("No valid solutions found during optimization")
        
        # Enhance all solutions with additional details
        enhanced_solutions = []
        for i, solution in enumerate(top_solutions):
            enhanced_solution = self._enhance_solution(solution, planning_time)
            enhanced_solution['solution_rank'] = i + 1
            enhanced_solutions.append(enhanced_solution)
        
        print(f"Multi-trip planning completed in {planning_time:.2f} seconds")
        print(f"Generated {len(enhanced_solutions)} alternative trip plans:")
        for i, solution in enumerate(enhanced_solutions):
            print(f"  {i+1}. Score: {solution['total_objective_score']:.2f}, Cost: ${solution['total_cost']:.2f}")
        
        return enhanced_solutions
    
    def _enhance_solution(self, solution: Dict, planning_time: float) -> Dict:
        """Add additional details and analysis to solution"""
        
        # Add planning metadata
        solution['planning_metadata'] = {
            'planning_time_seconds': planning_time,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'poi_desirability': self.poi_desirability,
            'cost_sensitivity': self.cost_sensitivity,
            'transportation_averseness': self.transportation_averseness,
            'algorithm': '2-layer beam search',
            'timestamp': datetime.now().isoformat()
        }
        
        # Add cost breakdown
        solution['cost_breakdown'] = self._calculate_cost_breakdown(solution)
        
        # Add city analysis
        solution['city_analysis'] = self._analyze_cities(solution)
        
        # Format daily schedule for readability
        solution['formatted_schedule'] = self._format_daily_schedule(solution)
        
        return solution
    
    def _calculate_cost_breakdown(self, solution: Dict) -> Dict:
        """Calculate detailed cost breakdown"""
        breakdown = {
            'hotel_costs': 0.0,
            'transportation_costs': 0.0,
            'local_transport_costs': 0.0,
            'total_cost': solution['total_cost']
        }
        
        for day_info in solution.get('daily_schedule', []):
            # Transportation costs
            if 'transportation' in day_info:
                breakdown['transportation_costs'] += day_info['transportation']['cost']
            
            # Hotel costs (estimated from preferences)
            if 'hotel' in day_info:
                # Estimate hotel cost based on preference
                breakdown['hotel_costs'] += 200.0  # Default estimate
            elif 'end_hotel' in day_info:
                breakdown['hotel_costs'] += 200.0
        
        breakdown['local_transport_costs'] = (
            breakdown['total_cost'] - 
            breakdown['hotel_costs'] - 
            breakdown['transportation_costs']
        )
        
        return breakdown
    
    def _analyze_cities(self, solution: Dict) -> Dict:
        """Analyze city visit patterns"""
        city_analysis = {}
        
        for city in solution.get('visited_cities', []):
            # Count days in each city
            days_in_city = sum(
                1 for day_info in solution.get('daily_schedule', [])
                if day_info.get('city') == city or day_info.get('end_city') == city
            )
            
            # Count POIs visited
            city_pois = set()
            for day_info in solution.get('daily_schedule', []):
                if day_info.get('city') == city or day_info.get('end_city') == city:
                    # Count POIs from activities (handle both old and new format)
                    activities = day_info.get('activities', []) + day_info.get('evening_activities', [])
                    for activity in activities:
                        if isinstance(activity, dict) and 'poi_name' in activity:
                            # New format: visit_info dictionary
                            city_pois.add(activity['poi_name'])
                        elif isinstance(activity, tuple) and len(activity) >= 3:
                            # Old format: (poi_name, start_time, end_time) tuple
                            city_pois.add(activity[0])
            
            city_analysis[city] = {
                'days_visited': days_in_city,
                'pois_visited': len(city_pois),
                'poi_names': list(city_pois)
            }
        
        return city_analysis
    
    def _format_time_safely(self, time_obj) -> str:
        """Safely format time object that could be datetime object or string"""
        if isinstance(time_obj, str):
            return time_obj
        elif hasattr(time_obj, 'strftime'):
            return time_obj.strftime('%H:%M')
        else:
            return str(time_obj)

    def _format_daily_schedule(self, solution: Dict) -> List[Dict]:
        """Format daily schedule for human readability"""
        formatted_schedule = []
        
        for day_info in solution.get('daily_schedule', []):
            day_num = day_info.get('day', 0) + 1
            date_str = day_info.get('date', '')
            
            formatted_day = {
                'day': day_num,
                'date': date_str,
                'summary': '',
                'activities': [],
                'logistics': {}
            }
            
            # Handle travel days
            if 'transportation' in day_info:
                start_city = day_info.get('start_city', '')
                end_city = day_info.get('end_city', '')
                transport = day_info['transportation']
                
                formatted_day['summary'] = f"Travel from {start_city} to {end_city}"
                formatted_day['logistics'] = {
                    'transportation': f"{transport['mode']} - {transport['departure_time']} to {transport['arrival_time']}",
                    'start_hotel': day_info.get('start_hotel', ''),
                    'end_hotel': day_info.get('end_hotel', ''),
                    'cost': f"${transport['cost']:.2f}"
                }
                
                # Add morning activities
                for visit_info in day_info.get('morning_activities', []):
                    formatted_day['activities'].append({
                        'time': f"{self._format_time_safely(visit_info['arrival_time'])} - {self._format_time_safely(visit_info['departure_time'])}",
                        'activity': visit_info['poi_name'],
                        'location': start_city,
                        'preference_score': visit_info['preference_score'],
                        'travel_cost': visit_info['travel_cost'],
                        'travel_mode': visit_info['travel_mode'],
                        'poi_type': visit_info['poi_type'],
                        'visit_duration': visit_info['visit_duration_minutes']
                    })
                
                # Add evening activities
                for visit_info in day_info.get('evening_activities', []):
                    formatted_day['activities'].append({
                        'time': f"{self._format_time_safely(visit_info['arrival_time'])} - {self._format_time_safely(visit_info['departure_time'])}",
                        'activity': visit_info['poi_name'],
                        'location': end_city,
                        'preference_score': visit_info['preference_score'],
                        'travel_cost': visit_info['travel_cost'],
                        'travel_mode': visit_info['travel_mode'],
                        'poi_type': visit_info['poi_type'],
                        'visit_duration': visit_info['visit_duration_minutes']
                    })
            
            # Handle stay days
            else:
                city = day_info.get('city', '')
                formatted_day['summary'] = f"Full day in {city}"
                formatted_day['logistics'] = {
                    'hotel': day_info.get('hotel', ''),
                    'city': city
                }
                
                # Add activities
                for visit_info in day_info.get('activities', []):
                    formatted_day['activities'].append({
                        'time': f"{self._format_time_safely(visit_info['arrival_time'])} - {self._format_time_safely(visit_info['departure_time'])}",
                        'activity': visit_info['poi_name'],
                        'location': city,
                        'preference_score': visit_info['preference_score'],
                        'travel_cost': visit_info['travel_cost'],
                        'travel_mode': visit_info['travel_mode'],
                        'poi_type': visit_info['poi_type'],
                        'visit_duration': visit_info['visit_duration_minutes']
                    })
            
            formatted_schedule.append(formatted_day)
        
        return formatted_schedule
    
    def analyze_city_worthiness(self, cities: List[str] = None, 
                               time_durations: List[float] = None) -> Dict:
        """
        Analyze worthiness scores for different cities and time allocations
        """
        if not self.subproblem_solver:
            raise ValueError("Must initialize optimizers before analysis")
        
        if cities is None:
            # Extract cities from POI_LIST keys
            cities = list(self.poi_data.get('POI_LIST', {}).keys())
            if not cities:
                # Fallback: extract from poi_details if POI_LIST is empty
                cities = list(self.poi_details.keys()) if isinstance(self.poi_details, dict) else []
        
        if time_durations is None:
            time_durations = [6, 12, 18, 24, 36, 48]  # Hours
        
        print("Analyzing city worthiness scores...")
        
        analysis_results = {}
        start_time = datetime.strptime(self.start_date + ' 09:00', '%Y-%m-%d %H:%M')
        
        for city in cities:
            city_results = {}
            
            for duration in time_durations:
                end_time = start_time + timedelta(hours=duration)
                
                # Find best hotel in city
                best_hotel = self.hotel_selector.select_best_hotel(
                    city=city,
                    date=self.start_date,
                    global_visited_hotels=set()
                )
                
                if best_hotel:
                    start_poi = best_hotel['name']
                    
                    # Solve subproblem
                    result = self.subproblem_solver.solve_subproblem(
                        city=city,
                        start_poi=start_poi,
                        start_time=start_time,
                        end_time=end_time,
                        global_visited_pois=set(),
                        beam_width=5,
                        date=self.start_date
                    )
                    
                    city_results[f"{duration}h"] = {
                        'objective_score': result['objective_score'],
                        'preference_score': result['preference_score'],
                        'pois_visited': len(result['visited_pois']),
                        'total_cost': result['total_cost'],
                        'travel_time': result['total_travel_time'],
                        'feasible': result['feasible']
                    }
                else:
                    city_results[f"{duration}h"] = {
                        'objective_score': 0.0,
                        'note': 'No hotels available'
                    }
            
            analysis_results[city] = city_results
        
        return analysis_results
    
    def get_hotel_rankings(self, cities: List[str] = None) -> Dict:
        """Get hotel rankings for each city"""
        if not self.hotel_selector:
            raise ValueError("Must initialize optimizers before analysis")
        
        if cities is None:
            # Extract cities from POI_LIST keys
            cities = list(self.poi_data.get('POI_LIST', {}).keys())
            if not cities:
                # Fallback: extract from poi_details if POI_LIST is empty
                cities = list(self.poi_details.keys()) if isinstance(self.poi_details, dict) else []
        
        print("Analyzing hotel rankings...")
        
        rankings = {}
        for city in cities:
            top_hotels = self.hotel_selector.get_top_hotels_in_city(
                city=city, 
                date=self.start_date, 
                top_k=5
            )
            
            rankings[city] = [
                {
                    'name': hotel['name'],
                    'score': hotel['score'],
                    'preference': hotel['info']['preference'],
                    'estimated_cost': hotel['info']['base_cost']
                }
                for hotel in top_hotels
            ]
        
        return rankings

