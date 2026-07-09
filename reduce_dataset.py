from pathlib import Path
import pandas as pd
import json

DATA_DIR = Path("data")

KEEP_EVERY = 50
KEEP_EVERY_JSON = 15


def reduce_csv(filename):

    path = DATA_DIR / filename

    df = pd.read_csv(path)

    original = len(df)

    reduced = df.iloc[::KEEP_EVERY].reset_index(drop=True)

    reduced.to_csv(path, index=False)

    print(
        f"{filename:40}"
        f"{original:10,} -> {len(reduced):10,}"
    )


def reduce_json(filename):

    path = DATA_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original = len(data)

    reduced = data[::KEEP_EVERY_JSON]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            reduced,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"{filename:40}"
        f"{original:10,} -> {len(reduced):10,}"
    )


def main():

    print("=" * 70)
    print("Reducing Dataset")
    print("=" * 70)

    csv_files = [

        "patients.csv",

        "providers.csv",

        "prescriptions.csv",

        "pharmacovigilance_cases.csv",

        "case_medical_review_notes.csv",

        "medwatch_regulatory_reports.csv",

    ]

    json_files = [

        "medical_knowledge_base.json",

    ]

    print()

    for file in csv_files:

        reduce_csv(file)

    for file in json_files:

        reduce_json(file)

    print()

    print("=" * 70)
    print("Finished")
    print("=" * 70)


if __name__ == "__main__":

    main()