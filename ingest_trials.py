from fetcher import fetch_trials
from embedder import ingest_trial

CONDITIONS = [
    "diabetes",
    "cancer",
    "hypertension",
    "alzheimers",
    "cardiovascular disease"
]


def main():
    import time
    for condition in CONDITIONS:
        print(f"\nProcessing: {condition}")
        trials = []
        for attempt in range(3):
            try:
                trials = fetch_trials(condition, max_results=10)
                break
            except Exception as e:
                print(f"  ⚠ Attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(3)
                else:
                    print(f"  ❌ Failed to fetch trials for {condition} after 3 attempts.")

        for trial_info in trials:
            ingest_trial(trial_info, condition)
    print("\nDone. Database populated.")


if __name__ == "__main__":
    main()

