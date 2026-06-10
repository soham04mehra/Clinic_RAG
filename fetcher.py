import httpx

API_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_trials(condition: str, max_results: int = 10) -> list[dict]:
    """Fetch clinical trials for a given condition from ClinicalTrials.gov."""
    params = {
        "query.cond": condition,
        "pageSize": max_results,
    }

    print(f"  Fetching trials for: {condition}")
    response = httpx.get(API_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    studies = data.get("studies", [])

    trials = []
    for item in studies:
        # In v2, the study object is directly the item inside the list
        trials.append(item)

    print(f"  Found {len(trials)} trials for: {condition}")
    return trials


def extract_adverse_events(ae_module: dict, trial_info: dict) -> dict:
    """Extract and structure adverse event data from the AdverseEventsModule."""
    nct_id    = trial_info["nct_id"]
    drug      = trial_info["drug_name"]
    condition = trial_info["condition"]
    phase     = trial_info["phase"]

    # In v2, ae_module contains totalNumDeaths and frequencyThreshold / other keys
    total_patients = ae_module.get("frequencyThreshold", "N/A")
    deaths         = ae_module.get("totalNumDeaths", "N/A")

    def parse_events(event_list: list) -> list[dict]:
        parsed = []
        for event in event_list:
            num_affected = event.get("numAffected", 0)
            num_at_risk  = event.get("numAtRisk", 0)
            try:
                pct = round((int(num_affected) / int(num_at_risk)) * 100, 1) if int(num_at_risk) > 0 else 0.0
            except (ValueError, ZeroDivisionError):
                pct = 0.0
            parsed.append({
                "term":    event.get("term", "Unknown"),
                "organ":   event.get("organSystem", "Unknown"),
                "pct":     pct,
                "affected": num_affected,
                "at_risk":  num_at_risk,
            })
        return parsed

    serious_events = parse_events(ae_module.get("seriousEvents", []))
    other_events   = parse_events(ae_module.get("otherEvents", []))

    return {
        "nct_id":          nct_id,
        "drug":            drug,
        "condition":       condition,
        "phase":           phase,
        "total_patients":  total_patients,
        "deaths":          deaths,
        "serious_events":  serious_events,
        "other_events":    other_events,
    }

