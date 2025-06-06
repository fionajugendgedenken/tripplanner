"""
Main Trip Planner - Complete Trip Planning System
Runs the enhanced trip planner and displays top 5 detailed travel plans
"""

import time
import sys
import os
from datetime import datetime, timedelta
from enhanced_trip_planner import EnhancedTripPlanner
from USER_PROFILE import START_DATE, END_DATE, POI_DESIRABILITY, COST_SENSITIVITY, TRANSPORTATION_AVERSENESS


class TeeOutput:
    """Class to write output to both console and file"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def save_trip_plans_to_file(travel_plans, file_path):
    """Save trip plans to a separate structured file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("ENHANCED TRIP PLANNER - RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Trip Period: {START_DATE} to {END_DATE}\n")
        f.write(f"Total Plans Generated: {len(travel_plans)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, trip_plan in enumerate(travel_plans, 1):
            f.write(f"TRIP PLAN #{i}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Objective Score: {trip_plan['total_objective_score']:.2f}\n")
            f.write(f"Total Cost: ${trip_plan['total_cost']:.2f}\n")
            f.write(f"Cities: {', '.join(trip_plan['visited_cities'])}\n")
            f.write(f"POIs Visited: {len(trip_plan['visited_pois'])}\n")
            f.write(f"Hotels Used: {len(trip_plan['visited_hotels'])}\n")
            f.write(f"Duration: {len(trip_plan['daily_schedule'])} days\n")
            
            f.write("\nDaily Schedule:\n")
            for day_info in trip_plan.get('daily_schedule', []):
                day_num = day_info.get('day', 0)
                date = day_info.get('date', 'Unknown')
                summary = day_info.get('summary', 'Unknown activity')
                f.write(f"  Day {day_num}: {date} - {summary}\n")
                
                activities = day_info.get('activities', [])
                for activity in activities:
                    if isinstance(activity, dict):
                        name = activity.get('activity', 'Unknown')
                        time_str = activity.get('time', 'Unknown')
                        preference = activity.get('preference_score', 0)
                        f.write(f"    {time_str}: {name}")
                        if preference > 0:
                            f.write(f" (⭐ {preference:.1f})")
                        f.write("\n")
            
            f.write("\n" + "="*80 + "\n\n")


def format_time(time_obj):
    """Format time object to string"""
    if isinstance(time_obj, str):
        return time_obj
    elif hasattr(time_obj, 'strftime'):
        return time_obj.strftime('%H:%M')
    else:
        return str(time_obj)


def display_trip_summary(plan_num, trip_plan):
    """Display a comprehensive summary of a trip plan"""
    print(f"\n{'='*80}")
    print(f"🏆 TRIP PLAN #{plan_num}")
    print(f"{'='*80}")
    
    # Calculate total travel time and extract city trajectory
    total_travel_time = 0  # in minutes
    city_trajectory = []
    
    # Extract city trajectory from daily schedule
    for day_info in trip_plan.get('daily_schedule', []):
        day_num = day_info.get('day', 0)
        summary = day_info.get('summary', '')
        
        # Extract city from summary
        if 'Full day in ' in summary:
            city = summary.replace('Full day in ', '')
            city_trajectory.append(f"Day {day_num}: {city}")
        elif 'Travel from ' in summary and ' to ' in summary:
            # Extract both cities for travel days
            parts = summary.replace('Travel from ', '').split(' to ')
            if len(parts) == 2:
                from_city, to_city = parts
                city_trajectory.append(f"Day {day_num}: {from_city} → {to_city}")
        elif 'Arrival from ' in summary:
            # Handle arrival day
            city_trajectory.append(f"Day {day_num}: Athens, Greece (arrival)")
    
    # Calculate travel time from travel costs
    for day_info in trip_plan.get('daily_schedule', []):
        activities = day_info.get('activities', [])
        for activity in activities:
            if isinstance(activity, dict):
                # Add travel times (from travel_cost > 0 activities)
                travel_cost = activity.get('travel_cost', 0)
                if travel_cost > 0:  # This indicates there was travel
                    # Estimate travel time from cost (rough approximation)
                    # Most local transport is ~$0.5/min, so travel_time ≈ travel_cost * 2
                    estimated_travel_time = travel_cost * 2
                    total_travel_time += estimated_travel_time
    
    # Overall Statistics
    print(f"📊 OVERALL STATISTICS:")
    print(f"   🎯 Total Objective Score: {trip_plan['total_objective_score']:.2f}")
    print(f"   🚌 Total Travel Time: {total_travel_time:.0f} minutes ({total_travel_time/60:.1f} hours)")
    print(f"   💰 Total Cost: ${trip_plan['total_cost']:.2f}")
    print(f"   🏙️  Cities Visited: {len(trip_plan['visited_cities'])}")
    print(f"   📍 POIs Visited: {len(trip_plan['visited_pois'])}")
    print(f"   🏨 Hotels Used: {len(trip_plan['visited_hotels'])}")
    print(f"   📅 Trip Duration: {len(trip_plan['daily_schedule'])} days")
    
    # City Trajectory
    print(f"\n🗺️  CITY TRAJECTORY:")
    for trajectory_item in city_trajectory:
        print(f"   {trajectory_item}")
    
    # Cost Breakdown
    if 'cost_breakdown' in trip_plan:
        print(f"\n💸 COST BREAKDOWN:")
        breakdown = trip_plan['cost_breakdown']
        print(f"   🏨 Hotel Costs: ${breakdown.get('hotel_costs', 0):.2f}")
        print(f"   ✈️  Transportation Costs: ${breakdown.get('transportation_costs', 0):.2f}")
        print(f"   🚌 Local Transport Costs: ${breakdown.get('local_transport_costs', 0):.2f}")
        print(f"   💰 Total: ${breakdown.get('total_cost', 0):.2f}")
    
    # Cities and POIs Summary - with POIs sorted by preference in decreasing order
    print(f"\n🗺️  CITIES & ATTRACTIONS:")
    
    # Collect POI preference scores from daily schedule
    poi_preferences = {}
    for day_info in trip_plan.get('daily_schedule', []):
        # Check different activity types
        activity_lists = []
        if day_info.get('activities'):
            activity_lists.append(day_info['activities'])
        if day_info.get('morning_activities'):
            activity_lists.append(day_info['morning_activities'])
        if day_info.get('evening_activities'):
            activity_lists.append(day_info['evening_activities'])
        
        for activities in activity_lists:
            for activity in activities:
                if isinstance(activity, dict):
                    poi_name = activity.get('poi_name')
                    preference = activity.get('preference_score', 0)
                    if poi_name and preference > 0:  # Only include POIs with actual preference scores
                        poi_preferences[poi_name] = preference
    
    for city in trip_plan['visited_cities']:
        city_pois = trip_plan.get('visited_pois_by_city', {}).get(city, [])
        city_hotels = trip_plan.get('visited_hotels_by_city', {}).get(city, [])
        
        # Sort POIs by preference score (decreasing order)
        city_pois_with_pref = [(poi, poi_preferences.get(poi, 0)) for poi in city_pois if poi in poi_preferences]
        city_pois_sorted = sorted(city_pois_with_pref, key=lambda x: x[1], reverse=True)
        
        print(f"   🏙️  {city}:")
        print(f"      📍 POIs visited: {len(city_pois_sorted)}")
        if city_pois_sorted:
            for poi, pref in city_pois_sorted[:3]:  # Show first 3
                print(f"         - {poi} (⭐ {pref:.1f})")
            if len(city_pois_sorted) > 3:
                print(f"         ... and {len(city_pois_sorted) - 3} more")
        print(f"      🏨 Hotels used: {', '.join(city_hotels) if city_hotels else 'None'}")


def display_daily_schedule(trip_plan):
    """Display detailed daily schedule with proper formatting and transportation details"""
    print(f"\n📅 DETAILED DAILY SCHEDULE:")
    print(f"{'-'*80}")
    
    schedule_data = trip_plan.get('formatted_schedule', trip_plan.get('daily_schedule', []))
    
    for day_info in schedule_data:
        # Fix day numbering - the data is already 1-indexed
        if 'day' in day_info:
            day_num = day_info['day']  # Don't add 1 - already correct
            date = day_info.get('date', 'Unknown')
        else:
            day_num = day_info.get('day', 1)  # Default to 1 if missing
            date = day_info.get('date', 'Unknown')
        
        print(f"\n📆 DAY {day_num} - {date}")
        print(f"{'─'*60}")
        
        # Handle the actual data format being used
        if 'summary' in day_info:
            # This is the enhanced trip planner format
            summary = day_info.get('summary', 'Unknown activity')
            print(f"📋 Summary: {summary}")
            
            # Show logistics/hotels
            logistics = day_info.get('logistics', {})
            if 'start_hotel' in logistics and 'end_hotel' in logistics:
                start_hotel = logistics['start_hotel']
                end_hotel = logistics['end_hotel']
                if start_hotel == end_hotel:
                    print(f"🏨 Hotel: {start_hotel}")
                else:
                    print(f"🏨 Hotels: {start_hotel} → {end_hotel}")
            elif 'hotel' in logistics:
                hotel = logistics['hotel']
                if hotel:
                    print(f"🏨 Hotel: {hotel}")
            elif 'city' in logistics:
                city = logistics['city']
                print(f"🏨 Hotel: (Stay in {city})")
            
            # Show transportation if it's a travel day
            if 'transportation' in logistics:
                transport_info = logistics['transportation']
                cost = logistics.get('cost', 'Unknown')
                print(f"✈️  Transportation: {transport_info}")
                print(f"💰 Cost: {cost}")
            
            # Show activities
            activities = day_info.get('activities', [])
            logistics = day_info.get('logistics', {})
            
            if activities:
                print(f"\n🎯 ACTIVITIES:")
                
                # For Day 1, add airport arrival first
                if day_num == 1:
                    print(f"   1. 12:00 - 12:30: Travel from ATH Airport to hotel")
                    print(f"       🚖 Taxi")
                    print(f"       💰 Cost: $15.00")
                    print(f"   2. 12:30 - 13:23: Hotel check-in and preparation")
                    event_num = 3
                    prev_location = "hotel"  # Start from hotel for Day 1
                    prev_end_time = "13:23"  # End of check-in
                else:
                    event_num = 1
                    prev_location = None
                    prev_end_time = "09:00"  # Start at 9 AM for other days
                
                # Track if we need to insert inter-city transportation
                has_intercity_transport = 'transportation' in logistics
                intercity_inserted = False
                
                for i, activity in enumerate(activities):
                    if isinstance(activity, dict):
                        activity_name = activity.get('activity', 'Unknown activity')
                        time_str = activity.get('time', 'Unknown time')
                        preference = activity.get('preference_score', 0)
                        travel_cost = activity.get('travel_cost', 0)
                        travel_mode = activity.get('travel_mode', 'unknown')
                        visit_duration = activity.get('visit_duration', 0)
                        poi_type = activity.get('poi_type', 'unknown')
                        poi_cost = activity.get('poi_cost', 0)  # Get POI cost
                        
                        # Parse the time string to get arrival and departure times
                        try:
                            if ' - ' in time_str:
                                arrival_str, departure_str = time_str.split(' - ')
                                # Validate the time format
                                datetime.strptime(arrival_str, '%H:%M')
                                datetime.strptime(departure_str, '%H:%M')
                            else:
                                arrival_str = time_str
                                departure_str = time_str
                                
                        except:
                            # Fallback if parsing fails
                            arrival_str = time_str.split(' - ')[0] if ' - ' in time_str else time_str
                            departure_str = time_str.split(' - ')[1] if ' - ' in time_str else time_str
                        
                        # Check if we need to insert inter-city transportation before this activity
                        # Insert when we switch from morning activities (first city) to evening activities (second city)
                        if has_intercity_transport and not intercity_inserted:
                            # Check if this activity is in the destination city
                            current_location = activity.get('location', '')
                            if 'start_hotel' in logistics and 'end_hotel' in logistics:
                                start_hotel = logistics['start_hotel']
                                end_hotel = logistics['end_hotel']
                                
                                # If we're now at the end hotel or in destination city activities
                                if (activity_name == end_hotel or 
                                    (current_location and current_location != logistics.get('start_city', ''))):
                                    
                                    # Insert inter-city transportation events
                                    transport_info = logistics.get('transportation', '')
                                    transport_cost = logistics.get('cost', 'Unknown')
                                    
                                    # Extract departure and arrival times from transport info
                                    if ' - ' in transport_info:
                                        mode_and_times = transport_info.split(' - ')
                                        if len(mode_and_times) == 2:
                                            mode = mode_and_times[0]
                                            times = mode_and_times[1].split(' to ')
                                            if len(times) == 2:
                                                dep_time, arr_time = times
                                                
                                                # Travel to departure hub
                                                print(f"   {event_num}. {prev_end_time} - {dep_time}: Travel to departure hub")
                                                print(f"       🚖 Taxi")
                                                print(f"       💰 Cost: $15.00")
                                                event_num += 1
                                                
                                                # Inter-city transportation
                                                print(f"   {event_num}. {dep_time} - {arr_time}: {mode.title()} transportation")
                                                if 'flight' in mode.lower():
                                                    print(f"       ✈️  Flight")
                                                elif 'ferry' in mode.lower():
                                                    print(f"       🚢 Ferry")
                                                else:
                                                    print(f"       🚌 {mode.title()}")
                                                print(f"       💰 Cost: {transport_cost}")
                                                event_num += 1
                                                
                                                # Travel from arrival hub to hotel
                                                print(f"   {event_num}. {arr_time} - {arrival_str}: Travel to hotel")
                                                print(f"       🚖 Taxi")
                                                print(f"       💰 Cost: $20.00")
                                                event_num += 1
                                                
                                                prev_end_time = arrival_str
                                                intercity_inserted = True
                        
                        # Show travel event (if there was travel cost and not already at hotel and not an endpoint)
                        if travel_cost > 0 and travel_mode not in ['already_here', 'walking'] and poi_type != 'endpoint':
                            # Use previous event's end time as travel start time
                            travel_start_str = prev_end_time if prev_end_time else "09:00"
                            
                            # Convert driving to taxi
                            display_mode = 'Taxi' if travel_mode == 'driving' else travel_mode.title()
                            
                            origin = prev_location if prev_location else "Previous location"
                            print(f"   {event_num}. {travel_start_str} - {arrival_str}: Travel from {origin} to {activity_name}")
                            print(f"       🚖 {display_mode}")
                            print(f"       💰 Cost: ${travel_cost:.2f}")
                            event_num += 1
                        
                        # Show short walk if walking (and not an endpoint)
                        elif travel_mode == 'walking' and prev_location and poi_type != 'endpoint':
                            print(f"   {event_num}. Walk from {prev_location} to {activity_name}")
                            print(f"       🚶 Walking (2-3 min)")
                            event_num += 1
                        
                        # Show visit event (if there's actual visit time)
                        if visit_duration > 0 and poi_type != 'endpoint':
                            print(f"   {event_num}. {arrival_str} - {departure_str}: Visit {activity_name}")
                            if preference > 0:
                                print(f"       ⭐ Preference: {preference:.1f}")
                            print(f"       ⏰ Duration: {visit_duration} min")
                            # Show POI cost if > 0
                            if poi_cost > 0:
                                print(f"       🎫 Entry cost: ${poi_cost:.2f}")
                            event_num += 1
                            # Update previous end time
                            prev_end_time = departure_str
                        
                        # For endpoint (return to hotel), show the complete return journey
                        elif poi_type == 'endpoint':
                            travel_start_str = prev_end_time if prev_end_time else "09:00"
                            if travel_cost > 0:
                                print(f"   {event_num}. {travel_start_str} - {arrival_str}: Return to {activity_name}")
                                display_mode = 'Taxi' if travel_mode == 'driving' else travel_mode.title()
                                print(f"       🚖 {display_mode}")
                                print(f"       💰 Cost: ${travel_cost:.2f}")
                                event_num += 1
                            elif travel_mode == 'walking':
                                print(f"   {event_num}. {travel_start_str} - {arrival_str}: Walk back to {activity_name}")
                                print(f"       🚶 Walking")
                                event_num += 1
                        
                        # Update previous location for next iteration
                        prev_location = activity_name
                    else:
                        print(f"   {event_num}. {activity}")
                        event_num += 1
            else:
                print(f"\n   No activities planned")
                
            # Show day score if available
            if 'day_score' in day_info:
                print(f"\n📊 Day Score: {day_info['day_score']:.2f}")
            
            continue  # Skip the old format handlers
        
        # Handle different schedule formats (old format - fallback)
        schedule_type = day_info.get('type', 'unknown')
        
        if schedule_type == 'airport_arrival':
            print(f"📋 Summary: Arrival from ATH Airport")
            end_hotel = day_info.get('end_hotel', 'Unknown')
            print(f"🏨 Hotel: {end_hotel}")
            print(f"🛬 Airport Arrival: {day_info.get('arrival_time', '12:00')}")
            
            # Show activities with proper transportation details
            activities = day_info.get('activities', [])
            if activities:
                print(f"\n🎯 ACTIVITIES ({len(activities) + 1} planned):")  # +1 for airport taxi
                
                # First show taxi from airport to hotel
                logistics = day_info.get('logistics', {})
                airport_to_hotel_time = logistics.get('airport_to_hotel_time', 30)
                arrival_time = day_info.get('arrival_time', '12:00')
                
                # Calculate taxi arrival time at hotel
                from datetime import datetime, timedelta
                try:
                    arrival_dt = datetime.strptime(arrival_time, '%H:%M')
                    hotel_arrival_dt = arrival_dt + timedelta(minutes=airport_to_hotel_time)
                    hotel_arrival_time = hotel_arrival_dt.strftime('%H:%M')
                except:
                    hotel_arrival_time = "12:30"  # fallback
                
                print(f"   1. {arrival_time} - {hotel_arrival_time}: Travel to {end_hotel}")
                print(f"       🚖 Transportation: Taxi from ATH Airport")
                print(f"       💰 Cost: ${logistics.get('airport_transfer_cost', 15.0):.2f}")
                print(f"       ⏱️  Travel time: {airport_to_hotel_time} min")
                
                # Then show other activities
                for i, activity in enumerate(activities, 2):  # Start from 2 since airport taxi is 1
                    display_activity_with_transport(activity, i, activities, i-2)
            else:
                print("   Only airport arrival and hotel check-in planned")
                
        elif schedule_type == 'travel':
            start_city = day_info.get('start_city', 'Unknown')
            end_city = day_info.get('end_city', 'Unknown')
            print(f"📋 Summary: Travel from {start_city} to {end_city}")
            
            # Show hotels
            start_hotel = day_info.get('start_hotel', 'Unknown')
            end_hotel = day_info.get('end_hotel', 'Unknown')
            print(f"🏨 Hotels: {start_hotel} → {end_hotel}")
            
            # Transportation details
            transport = day_info.get('transportation', {})
            if transport:
                mode = transport.get('mode', 'Unknown')
                dep_time = transport.get('departure_time', 'Unknown')
                arr_time = transport.get('arrival_time', 'Unknown')
                cost = transport.get('cost', 0)
                print(f"✈️  Inter-city Transportation: {mode}")
                print(f"   ⏰ Departure: {dep_time} from {start_city}")
                print(f"   ⏰ Arrival: {arr_time} in {end_city}")
                print(f"   💰 Cost: ${cost:.2f}")
            
            # Morning activities
            morning_activities = day_info.get('morning_activities', [])
            if morning_activities:
                print(f"\n🌅 MORNING ACTIVITIES in {start_city}:")
                for i, activity in enumerate(morning_activities, 1):
                    display_activity_with_transport(activity, i, morning_activities, i-1, start_hotel if i == 1 else None)
            
            # Evening activities  
            evening_activities = day_info.get('evening_activities', [])
            if evening_activities:
                print(f"\n🌆 EVENING ACTIVITIES in {end_city}:")
                for i, activity in enumerate(evening_activities, 1):
                    display_activity_with_transport(activity, i, evening_activities, i-1, end_hotel if i == 1 else None)
                            
        elif schedule_type == 'stay':
            city = day_info.get('city', 'Unknown')
            print(f"📋 Summary: Full day in {city}")
            start_hotel = day_info.get('start_hotel', 'Unknown')
            end_hotel = day_info.get('end_hotel', 'Unknown')
            
            if start_hotel == end_hotel:
                print(f"🏨 Hotel: {start_hotel}")
            else:
                print(f"🏨 Hotels: {start_hotel} → {end_hotel}")
            
            # Activities with transportation details
            activities = day_info.get('activities', [])
            if activities:
                print(f"\n🎯 ACTIVITIES in {city} ({len(activities)} planned):")
                for i, activity in enumerate(activities, 1):
                    # Determine start location for first activity
                    start_location = start_hotel if i == 1 else None
                    display_activity_with_transport(activity, i, activities, i-1, start_location)
        
        elif schedule_type == 'departure':
            city = day_info.get('city', 'Unknown')
            print(f"📋 Summary: Departure day in {city}")
            start_hotel = day_info.get('start_hotel', 'Unknown')
            print(f"🏨 Hotel: {start_hotel}")
            
            # Show departure logistics
            departure_logistics = day_info.get('departure_logistics', {})
            if departure_logistics:
                checkout_time = departure_logistics.get('checkout_time', 'Unknown')
                departure_time = departure_logistics.get('departure_time', '12:00')
                airport = departure_logistics.get('airport', 'ATH Airport')
                travel_time = departure_logistics.get('hotel_to_airport_time', 'Unknown')
                
                print(f"🛫 Departure Details:")
                print(f"   ⏰ Hotel Checkout: {checkout_time}")
                print(f"   ⏰ Airport Departure: {departure_time}")
                print(f"   🚖 Travel to {airport}: {travel_time} min")
            
            # Show morning activities if any
            activities = day_info.get('activities', [])
            if activities:
                print(f"\n🌅 MORNING ACTIVITIES before departure ({len(activities)} planned):")
                for i, activity in enumerate(activities, 1):
                    start_location = start_hotel if i == 1 else None
                    display_activity_with_transport(activity, i, activities, i-1, start_location)
            else:
                print(f"\n   No morning activities - direct departure")
        
        else:
            print(f"❌ Unknown schedule type: {schedule_type}")
            print(f"   Available data: {day_info}")
        
        # Show day score if available (for old format only)
        if 'day_score' in day_info and 'summary' not in day_info:
            print(f"\n📊 Day Score: {day_info['day_score']:.2f}")


def display_activity_with_transport(activity, activity_num, all_activities, prev_idx, start_location=None):
    """Display activity with transportation details from previous location"""
    if isinstance(activity, dict):
        name = activity.get('poi_name', 'Unknown POI')
        arr_time = format_time(activity.get('arrival_time', 'Unknown'))
        dep_time = format_time(activity.get('departure_time', 'Unknown'))
        pref_score = activity.get('preference_score', 0)
        visit_duration = activity.get('visit_duration_minutes', 0)
        travel_time = activity.get('travel_time_minutes', 0)
        travel_cost = activity.get('travel_cost', 0)
        travel_mode = activity.get('travel_mode', 'unknown')
        poi_cost = activity.get('poi_cost', 0)
        
        # Convert driving to taxi
        if travel_mode == 'driving':
            travel_mode = 'taxi'
        
        # Show main activity line
        if visit_duration > 0:
            print(f"   {activity_num:2d}. {arr_time} - {dep_time}: {name}")
        else:
            print(f"   {activity_num:2d}. {arr_time}: {name}")
        
        # Show preference score if > 0
        if pref_score > 0:
            print(f"       ⭐ Preference: {pref_score:.1f}")
        
        # Show transportation details if there was travel
        if travel_time > 0 and travel_mode != 'already_here':
            # Determine origin
            if start_location:
                origin = start_location
            elif prev_idx >= 0 and prev_idx < len(all_activities):
                prev_activity = all_activities[prev_idx]
                if isinstance(prev_activity, dict):
                    origin = prev_activity.get('poi_name', 'Previous location')
                else:
                    origin = 'Previous location'
            else:
                origin = 'Previous location'
            
            print(f"       🚖 Transportation: {travel_mode.title()} from {origin}")
            print(f"       💰 Travel cost: ${travel_cost:.2f}")
            print(f"       ⏱️  Travel time: {travel_time} min")
        
        # Show visit time if > 0
        if visit_duration > 0:
            print(f"       ⏰ Visit time: {visit_duration} min")
        
        # Show POI cost if > 0
        if poi_cost > 0:
            print(f"       🎫 Entry cost: ${poi_cost:.2f}")
    else:
        print(f"   {activity_num:2d}. {activity}")


if __name__ == "__main__":
    """Main function to run the enhanced trip planner"""
    
    # Create output directory if it doesn't exist
    output_dir = "trip_planner_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamped file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"trip_planner_log_{timestamp}.txt")
    results_file = os.path.join(output_dir, f"trip_plans_{timestamp}.txt")
    
    # Set up output redirection to both console and log file
    tee = TeeOutput(log_file)
    sys.stdout = tee
    
    try:
        print("🌟 ENHANCED TRIP PLANNER - COMPLETE SYSTEM")
        print("=" * 80)
        print(f"📅 Trip Period: {START_DATE} to {END_DATE}")
        print(f"🎯 User Preferences:")
        print(f"   POI Desirability: {POI_DESIRABILITY}")
        print(f"   Cost Sensitivity: {COST_SENSITIVITY}")
        print(f"   Transportation Averseness: {TRANSPORTATION_AVERSENESS}")
        print("=" * 80)
        print(f"📁 Output will be saved to:")
        print(f"   Log file: {log_file}")
        print(f"   Results file: {results_file}")
        print("=" * 80)
        
        # Initialize the enhanced trip planner
        print("\n🚀 1. INITIALIZING TRIP PLANNER...")
        planner = EnhancedTripPlanner(
            start_date=START_DATE,
            end_date=END_DATE,
            poi_desirability=POI_DESIRABILITY,
            cost_sensitivity=COST_SENSITIVITY,
            transportation_averseness=TRANSPORTATION_AVERSENESS
        )
        
        # Load all data
        print("\n📊 2. LOADING DATA...")
        load_start = time.time()
        stats = planner.load_all_data()
        load_time = time.time() - load_start
        
        print(f"   ✅ Data loaded in {load_time:.2f}s")
        print(f"   📊 Statistics:")
        print(f"      POIs: {stats['poi_count']}")
        print(f"      Cities: {stats['cities_count']}")
        print(f"      Transportation options: {stats['total_transportation_options']}")
        print(f"      Routes: {stats['total_routes']}")
        
        # Initialize optimizers
        print("\n⚙️  3. INITIALIZING OPTIMIZERS...")
        init_start = time.time()
        planner.initialize_optimizers()
        init_time = time.time() - init_start
        print(f"   ✅ Optimizers initialized in {init_time:.2f}s")
        
        # Plan multiple trips
        print("\n🎯 4. PLANNING MULTIPLE TRIP ALTERNATIVES...")
        plan_start = time.time()
        
        # Get top 5 travel plans
        travel_plans = planner.plan_multiple_trips(beam_width=10, return_top_k=5)

        plan_time = time.time() - plan_start
        
        if not travel_plans:
            print("❌ ERROR: No valid travel plans found!")
        
        else:

            print(f"   ✅ Planning completed in {plan_time:.2f}s")
            print(f"   🏆 Generated {len(travel_plans)} travel plan alternatives")
            
            # Display all travel plans
            print("\n" + "="*80)
            print("🏆 TOP 5 TRAVEL PLANS")
            print("="*80)
            
            for i, trip_plan in enumerate(travel_plans, 1):
                # Display trip summary
                display_trip_summary(i, trip_plan)
                
                # Display detailed daily schedule
                display_daily_schedule(trip_plan)
                
                # Separator between plans
                if i < len(travel_plans):
                    print("\n" + "🔸" * 80)
            
            # Performance summary
            total_time = load_time + init_time + plan_time
            print(f"\n⏱️  PERFORMANCE SUMMARY:")
            print(f"   Data Loading: {load_time:.2f}s")
            print(f"   Initialization: {init_time:.2f}s")
            print(f"   Planning: {plan_time:.2f}s")
            print(f"   Total Time: {total_time:.2f}s")
            
            # Comparison summary
            print(f"\n📊 QUICK COMPARISON OF TOP 5 PLANS:")
            print(f"{'Plan':<6}{'Score':<10}{'Cost':<12}{'Cities':<8}{'POIs':<6}{'Hotels':<7}")
            print("-" * 50)
            for i, plan in enumerate(travel_plans, 1):
                score = plan['total_objective_score']
                cost = plan['total_cost']
                cities = len(plan['visited_cities'])
                pois = len(plan['visited_pois'])
                hotels = len(plan['visited_hotels'])
                print(f"#{i:<5}{score:<10.2f}${cost:<11.2f}{cities:<8}{pois:<6}{hotels:<7}")
            
            print(f"\n🎉 TRIP PLANNING COMPLETED SUCCESSFULLY!")
            print(f"Choose your preferred plan and have an amazing trip! ✈️🏖️")
            
            # Save trip plans to structured file
            print(f"\n💾 SAVING RESULTS...")
            save_trip_plans_to_file(travel_plans, results_file)
            print(f"   ✅ Trip plans saved to: {results_file}")
            print(f"   ✅ Complete log saved to: {log_file}")
            
    except Exception as e:
        print(f"\n❌ ERROR: Trip planning failed!")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore stdout and close log file
        sys.stdout = tee.terminal
        tee.close()
        print(f"\n📁 All output has been saved to files in '{output_dir}' directory")