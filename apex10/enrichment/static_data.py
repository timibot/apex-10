"""
Static data Lookups for non-API enrichments. 
Includes manager tenure and key player absences.
"""
from datetime import date

# Team mapping -> appointment date (YYYY-MM-DD)
# Kept static. Update when significant managerial changes occur.
# Used to calculate 'manager_days_in_post' (new manager bounce).
MANAGER_APPOINTMENTS = {
    "Manchester United": "2024-11-11",
    "Liverpool": "2024-06-01",  
    "Chelsea": "2024-06-01",    
    "Bayern Munich": "2024-05-29",
    "Barcelona": "2024-05-29",
    "Juventus": "2024-06-12", 
    "AC Milan": "2024-06-13", 
    "Napoli": "2024-06-05", 
    "Real Madrid": "2021-06-01",
    "Manchester City": "2016-07-01", 
    "Arsenal": "2019-12-20", 
    "Bayer Leverkusen": "2022-10-05", 
    "Inter": "2021-06-03", 
    "Atalanta": "2016-06-14", 
    "Atletico Madrid": "2011-12-23", 
    "Genoa": "2024-11-20", 
    "Roma": "2024-11-14", 
    "Lecce": "2024-11-11", 
    "Hoffenheim": "2024-11-11", 
    "Rennes": "2024-11-11", 
}

def get_manager_days_in_post(team: str) -> int:
    """Returns how many days the current manager has been in charge."""
    if team not in MANAGER_APPOINTMENTS:
        return 365 # Default assumption
        
    try:
        appt_date = date.fromisoformat(MANAGER_APPOINTMENTS[team])
        return max(1, (date.today() - appt_date).days)
    except ValueError:
        return 365
        
# Purpose: Maintain static, high-value enrichment data. 
# Update TEAMS_WITH_KEY_ABSENCES weekly before running inference.py

# Add the EXACT team name as it appears in your database if they are missing a structural pillar.
TEAMS_WITH_KEY_ABSENCES = {
    # "Manchester City",  # e.g., missing Rodri
    "Arsenal",          # e.g., missing Odegaard
    "Bournemouth",
    # "Real Madrid",      # e.g., missing Bellingham
}

def is_key_player_absent(team_name: str) -> int:
    """
    Returns 1 if the team is currently listed in the manual veto set, 0 otherwise.
    """
    return 1 if team_name in TEAMS_WITH_KEY_ABSENCES else 0
