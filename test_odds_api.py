import requests
import json

api_key = "f6aac04a6ab847bab31a7db076ef89e8"

# Step 1: Get all NBA events
print("=" * 80)
print("STEP 1: Getting NBA events...")
print("=" * 80)

events_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
params = {'apiKey': api_key}

response = requests.get(events_url, params=params, timeout=10)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    events = response.json()
    print(f"Found {len(events)} events\n")
    
    # Find MIN vs LAL
    target_event = None
    for event in events:
        home = event.get('home_team', '')
        away = event.get('away_team', '')
        event_id = event.get('id', '')
        
        print(f"Event: {away} @ {home} (ID: {event_id})")
        
        if ('Lakers' in home or 'Lakers' in away) and \
           ('Timberwolves' in home or 'Timberwolves' in away):
            target_event = event
            print("  ⭐ FOUND MIN vs LAL!")
    
    if target_event:
        event_id = target_event['id']
        print("\n" + "=" * 80)
        print(f"STEP 2: Getting player props for event {event_id}")
        print("=" * 80)
        
        props_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
        props_params = {
            'apiKey': api_key,
            'regions': 'us',
            'bookmakers': 'fanduel',
            'markets': 'player_points',
            'oddsFormat': 'american'
        }
        
        props_response = requests.get(props_url, params=props_params, timeout=15)
        print(f"Status: {props_response.status_code}\n")
        
        if props_response.status_code == 200:
            props_data = props_response.json()
            
            print("Full response structure:")
            print(json.dumps(props_data, indent=2)[:2000])  # First 2000 chars
        else:
            print(f"Error: {props_response.text}")
else:
    print(f"Error: {response.text}")