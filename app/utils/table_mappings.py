"""Table-agent mappings and utilities for WSSD database."""

from typing import Dict, List

# Define which tables belong to which agent based on your structure
AGENT_TABLE_MAPPING = {
    "location": [
        "districts", "divisions", "sub_divisions", "regions",
        "region_circles", "circles", "blocks", "villages",
        "grampanchayats", "habitations", "states"
    ],
    "user": [
        "users", "citizen_users", "grievance_users"
    ],
    "grievance": [
        "grievances", "grievance_categories", "sub_grievance_categories"
    ],
    "schemes": [
        "scheme_categories", "scheme_types", "schemes"
    ],
    "tracker": [
        "grievance_resolve_tracks", "grievance_resolve_track_logs", 
        "grievance_assigned_accept_reject_users"
    ]
}

# Reverse mapping: table -> agent
TABLE_AGENT_MAPPING = {}
for agent, tables in AGENT_TABLE_MAPPING.items():
    for table in tables:
        TABLE_AGENT_MAPPING[table] = agent


def get_agent_for_table(table_name: str) -> str:
    """Get the agent responsible for a specific table."""
    return TABLE_AGENT_MAPPING.get(table_name, "location")  # Default to location


def get_tables_for_agent(agent_name: str) -> List[str]:
    """Get all tables managed by a specific agent."""
    return AGENT_TABLE_MAPPING.get(agent_name, [])


def get_all_agents() -> List[str]:
    """Get list of all available agents."""
    return list(AGENT_TABLE_MAPPING.keys())


def get_table_relationships() -> Dict[str, List[str]]:
    """Get table relationships for better query generation."""
    return {
        "users": ["districts", "circles", "blocks", "villages"],
        "grievances": ["users", "grievance_categories", "sub_grievance_categories"],
        "grievance_resolve_tracks": ["grievances", "users"],
        "schemes": ["scheme_categories", "scheme_types"],
        "districts": ["states", "divisions"],
        "circles": ["regions", "region_circles"],
        "blocks": ["districts", "circles"],
        "villages": ["blocks", "grampanchayats"],
        "grampanchayats": ["blocks", "habitations"]
    }