import argparse
import hashlib
import json
import os
from typing import Dict, List, Set, Tuple

from data_loader import load_names_by_language
from markov_generator import MarkovNameGenerator


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the name generator CLI."""
    parser = argparse.ArgumentParser(
        description="Generate Finnish or Swedish names using Markov chains"
    )
    parser.add_argument(
        "--language",
        choices=["finnish", "swedish", "both"],
        default="finnish",
        help='Language of names to generate (use "both" for Finnish and Swedish combined)',
    )
    parser.add_argument(
        "--gender",
        choices=["boys", "girls", "both"],
        default="both",
        help="Gender of names to generate",
    )
    parser.add_argument(
        "--count", type=int, default=10, help="Number of names to generate"
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="Markov chain order (complexity)",
    )
    parser.add_argument(
        "--min-length", type=int, default=3, help="Minimum name length"
    )
    parser.add_argument(
        "--max-length", type=int, default=12, help="Maximum name length"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show model statistics"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="",
        help='Starting string for generated names (e.g., "ju" for names starting with "ju")',
    )
    parser.add_argument(
        "--end",
        type=str,
        default="",
        help='Ending string for generated names (e.g., "o" for names ending with "o")',
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow generating names that already exist in the training data",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable model caching (always retrain and do not save cache)",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining and overwrite any existing cached model",
    )
    return parser.parse_args()


def _compute_dataset_signature(
    language: str,
    gender: str,
    names_data: Dict[str, Tuple[List[str], List[int]]],
) -> str:
    """
    Compute a stable signature for the training data used for the model.
    This is based on the (name, weight) pairs that will be fed into training.
    """
    items: List[Tuple[str, int]] = []
    for g in ["boys", "girls"]:
        if g in names_data and gender in [g, "both"]:
            n, w = names_data[g]
            items.extend(zip(n, w))

    hasher = hashlib.sha256()
    hasher.update(language.encode("utf-8"))
    hasher.update(b"|")
    hasher.update(gender.encode("utf-8"))
    hasher.update(b"|")

    for name, weight in sorted(items):
        hasher.update(name.encode("utf-8"))
        hasher.update(b":")
        hasher.update(str(weight).encode("ascii"))
        hasher.update(b";")

    return hasher.hexdigest()


def _get_cache_dir() -> str:
    """Return the directory used for cached Markov models, creating it if needed."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "models")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_cache_path(language: str, gender: str, order: int) -> str:
    """Construct a cache file path for the given configuration."""
    safe_language = language.replace(os.sep, "_")
    safe_gender = gender.replace(os.sep, "_")
    filename = f"markov_{safe_language}_{safe_gender}_order{order}.json"
    return os.path.join(_get_cache_dir(), filename)


def _load_cached_model(
    cache_path: str, expected_signature: str
) -> MarkovNameGenerator | None:
    """
    Load a cached model if the cache exists, is valid JSON, and matches
    the expected dataset signature. Returns None on any mismatch or error.
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    stored_signature = payload.get("dataset_signature")
    model_data = payload.get("model")
    if stored_signature != expected_signature or not isinstance(model_data, dict):
        return None

    try:
        return MarkovNameGenerator.from_dict(model_data)
    except Exception:
        return None


def _save_cached_model(
    generator: MarkovNameGenerator, cache_path: str, dataset_signature: str
) -> None:
    """Persist a trained model and its dataset signature to disk."""
    payload = {
        "dataset_signature": dataset_signature,
        "model": generator.to_dict(),
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _build_prevalence_lookups(
    language: str,
    names_data: Dict[str, Tuple[List[str], List[int]]],
    names_data_fi: Dict[str, Tuple[List[str], List[int]]] | None = None,
    names_data_sv: Dict[str, Tuple[List[str], List[int]]] | None = None,
) -> tuple[Dict[str, int], Dict[str, int]]:
    """Build per-language prevalence lookups for duplicate display."""
    finnish_prevalence: Dict[str, int] = {}
    swedish_prevalence: Dict[str, int] = {}

    if language == "both" and names_data_fi is not None and names_data_sv is not None:
        for gender in ["boys", "girls"]:
            n_fi, w_fi = names_data_fi.get(gender, ([], []))
            for n, w in zip(n_fi, w_fi):
                finnish_prevalence[n] = finnish_prevalence.get(n, 0) + w
            n_sv, w_sv = names_data_sv.get(gender, ([], []))
            for n, w in zip(n_sv, w_sv):
                swedish_prevalence[n] = swedish_prevalence.get(n, 0) + w
    elif language == "finnish":
        for gender in ["boys", "girls"]:
            if gender in names_data:
                n, w = names_data[gender]
                for name, weight in zip(n, w):
                    finnish_prevalence[name] = finnish_prevalence.get(name, 0) + weight
    else:  # swedish
        for gender in ["boys", "girls"]:
            if gender in names_data:
                n, w = names_data[gender]
                for name, weight in zip(n, w):
                    swedish_prevalence[name] = swedish_prevalence.get(name, 0) + weight

    return finnish_prevalence, swedish_prevalence


def main() -> None:
    """Main application entry point."""
    args = parse_args()

    # Load the names with prevalence weights (one or both languages).
    try:
        if args.language == "both":
            names_data_fi = load_names_by_language("finnish", args.gender)
            names_data_sv = load_names_by_language("swedish", args.gender)
            # Merge: for each gender, combine names and weights from both languages.
            names_data: Dict[str, Tuple[List[str], List[int]]] = {}
            for gender in ["boys", "girls"]:
                names_fi, weights_fi = names_data_fi.get(gender, ([], []))
                names_sv, weights_sv = names_data_sv.get(gender, ([], []))
                names_data[gender] = (names_fi + names_sv, weights_fi + weights_sv)
        else:
            names_data = load_names_by_language(args.language, args.gender)
            names_data_fi = None
            names_data_sv = None
    except Exception as e:  # pragma: no cover - defensive logging
        print(f"Error loading names: {e}")
        return

    # Language flags and display.
    if args.language == "both":
        language_flag = "🇫🇮🇸🇪"
        language_name = "Finnish & Swedish"
    else:
        language_flag = "🇫🇮" if args.language == "finnish" else "🇸🇪"
        language_name = args.language.capitalize()

    print(f"{language_flag} {language_name} Name Generator using Markov Chains")
    print("=" * 60)

    # Combine names and weights based on gender selection.
    all_names: List[str] = []
    all_weights: List[int] = []
    total_names = 0

    for gender in ["boys", "girls"]:
        if gender in names_data and args.gender in [gender, "both"]:
            gender_names, gender_weights = names_data[gender]
            all_names.extend(gender_names)
            all_weights.extend(gender_weights)
            total_names += len(gender_names)

            print(f"Loaded {len(gender_names)} {language_name} {gender} names")

    if not all_names:
        print(f"No names found for {language_name} {args.gender}")
        return

    if args.gender == "both":
        print(f"Total: {total_names} {language_name} names (boys and girls)")

    # Build per-language prevalence lookups (for --allow-duplicates display).
    finnish_prevalence, swedish_prevalence = _build_prevalence_lookups(
        args.language, names_data, names_data_fi, names_data_sv
    )

    # Prepare caching configuration.
    use_cache = not args.no_cache
    dataset_signature = _compute_dataset_signature(
        args.language, args.gender, names_data
    )
    cache_path = _get_cache_path(args.language, args.gender, args.order)

    generator: MarkovNameGenerator

    if use_cache and not args.force_retrain:
        cached = _load_cached_model(cache_path, dataset_signature)
        if cached is not None:
            generator = cached
            print(f"\nUsing cached model from {cache_path}")
        else:
            generator = MarkovNameGenerator(order=args.order)
            generator.train(all_names, all_weights)
            _save_cached_model(generator, cache_path, dataset_signature)
    elif use_cache and args.force_retrain:
        generator = MarkovNameGenerator(order=args.order)
        generator.train(all_names, all_weights)
        _save_cached_model(generator, cache_path, dataset_signature)
    else:
        # Caching disabled: always train a fresh model and do not persist it.
        generator = MarkovNameGenerator(order=args.order)
        generator.train(all_names, all_weights)

    if args.stats:
        stats = generator.get_stats()
        total_weight = sum(all_weights)
        avg_weight = total_weight / len(all_weights) if all_weights else 0
        print("\nModel Statistics:")
        print(f"  Training names: {stats['training_names_count']}")
        print(f"  Total prevalence weight: {total_weight:,}")
        print(f"  Average prevalence: {avg_weight:.1f}")
        print(f"  Unique contexts: {stats['unique_contexts']}")
        print(f"  Chain order: {stats['chain_order']}")
        print(f"  Average name length: {stats['average_name_length']:.1f}")
        print()

    # Display generation info.
    duplicate_info = " (including training data)" if args.allow_duplicates else ""
    parts: List[str] = []
    if args.start:
        parts.append(f"starting with '{args.start}'")
    if args.end:
        parts.append(f"ending with '{args.end}'")
    if parts:
        print(f"\nGenerating {args.count} names {', '.join(parts)}{duplicate_info}:")
        print("-" * 50)
    else:
        print(f"\nGenerating {args.count} names{duplicate_info}:")
        print("-" * 40)

    # Generate names.
    generated_names: Set[str] = set()
    attempts = 0
    max_attempts = args.count * 10  # Prevent infinite loops.

    while len(generated_names) < args.count and attempts < max_attempts:
        try:
            name = generator.generate(
                max_length=args.max_length,
                min_length=args.min_length,
                start_with=args.start,
                end_with=args.end,
            )
            if name:
                # Check if we should allow duplicates or not.
                if args.allow_duplicates or name not in all_names:
                    generated_names.add(name)
        except ValueError as e:  # pragma: no cover - user-facing error path
            print(f"Error: {e}")
            break
        attempts += 1

    for i, name in enumerate(sorted(generated_names), 1):
        line = f"{i:2d}. {name}"
        if args.allow_duplicates:
            parts = []
            if name in finnish_prevalence:
                parts.append(f"FI: {finnish_prevalence[name]:,}")
            if name in swedish_prevalence:
                parts.append(f"SE: {swedish_prevalence[name]:,}")
            if parts:
                line += "  (" + " | ".join(parts) + ")"
        print(line)

    if len(generated_names) < args.count:
        print(
            f"\nNote: Only generated {len(generated_names)} names after {max_attempts} attempts."
        )


__all__ = ["main", "parse_args"]

