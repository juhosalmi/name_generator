import csv
import os
from typing import Dict, List, Tuple


def load_names_from_csv(language: str, gender: str) -> Tuple[List[str], List[int]]:
    """
    Load names from CSV files with prevalence data.

    Args:
        language: 'finnish' or 'swedish'.
        gender: 'male' or 'female'.

    Returns:
        Tuple of (names list, prevalence weights list).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filename = f"{language}_{gender}.csv"
    csv_path = os.path.join(script_dir, "names", csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    names: List[str] = []
    weights: List[int] = []

    try:
        with open(csv_path, "r", encoding="utf-8") as file:
            # Detect delimiter and quote character.
            if language == "finnish":
                reader = csv.reader(file)
            else:  # Swedish files use semicolon delimiter and different format.
                reader = csv.reader(file, delimiter=";")

            for row_num, row in enumerate(reader):
                if row_num == 0 and language == "swedish":
                    # Skip header row for Swedish files.
                    continue

                if len(row) >= 2:
                    name = row[0].strip().strip('"')
                    prevalence_str = row[1].strip().strip('"')

                    # Skip empty names or header-like rows.
                    if not name or name.lower() in ["f�rnamn", "förnamn", "name"]:
                        continue

                    try:
                        # Handle different prevalence formats.
                        if "," in prevalence_str:
                            # Finnish format: "29,887".
                            prevalence = int(prevalence_str.replace(",", ""))
                        else:
                            # Swedish format: 15.
                            prevalence = int(prevalence_str)

                        if prevalence > 0:  # Only include names with positive prevalence.
                            names.append(name)
                            weights.append(prevalence)
                    except ValueError:
                        # Skip rows with invalid prevalence data.
                        continue

    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails.
        with open(csv_path, "r", encoding="latin-1") as file:
            if language == "finnish":
                reader = csv.reader(file)
            else:
                reader = csv.reader(file, delimiter=";")

            for row_num, row in enumerate(reader):
                if row_num == 0 and language == "swedish":
                    continue

                if len(row) >= 2:
                    name = row[0].strip().strip('"')
                    prevalence_str = row[1].strip().strip('"')

                    if not name or name.lower() in ["f�rnamn", "förnamn", "name"]:
                        continue

                    try:
                        if "," in prevalence_str:
                            prevalence = int(prevalence_str.replace(",", ""))
                        else:
                            prevalence = int(prevalence_str)

                        if prevalence > 0:
                            names.append(name)
                            weights.append(prevalence)
                    except ValueError:
                        continue

    return names, weights


def load_names_by_language(language: str, gender: str) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Load names for a specific language and gender.

    Args:
        language: 'finnish' or 'swedish'.
        gender: 'boys', 'girls', or 'both'.

    Returns:
        Dictionary with gender as key and (names, weights) tuple as value.
    """
    result: Dict[str, Tuple[List[str], List[int]]] = {}

    if gender in ["boys", "both"]:
        try:
            boys_names, boys_weights = load_names_from_csv(language, "male")
            result["boys"] = (boys_names, boys_weights)
        except FileNotFoundError:
            print(f"Warning: Could not load {language} male names")
            result["boys"] = ([], [])

    if gender in ["girls", "both"]:
        try:
            girls_names, girls_weights = load_names_from_csv(language, "female")
            result["girls"] = (girls_names, girls_weights)
        except FileNotFoundError:
            print(f"Warning: Could not load {language} female names")
            result["girls"] = ([], [])

    return result

