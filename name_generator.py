#!/usr/bin/env python3
"""
Finnish/Swedish Name Generator using Markov Chains
A simple application that generates Finnish or Swedish names for children using Markov chain text generation.
"""

import random
import re
import csv
import os
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple
import argparse


class MarkovNameGenerator:
    """
    A Markov chain-based name generator for Finnish and Swedish names.
    """
    
    def __init__(self, order: int = 2):
        """
        Initialize the Markov chain generator.
        
        Args:
            order: The order of the Markov chain (number of characters to consider for next character prediction)
        """
        self.order = order
        self.chains = defaultdict(Counter)
        self.names = []
        
    def train(self, names: List[str], weights: Optional[List[int]] = None) -> None:
        """
        Train the Markov chain on a list of names with optional weights.
        
        Args:
            names: List of names to train on
            weights: Optional list of weights (prevalence) for each name
        """
        if weights is None:
            weights = [1] * len(names)
        
        if len(names) != len(weights):
            raise ValueError("Names and weights lists must have the same length")
        
        self.names = []
        for name, weight in zip(names, weights):
            if self._is_valid_name(name):
                clean_name = self._clean_name(name)
                self.names.append(clean_name)
                
                # Add start and end markers
                padded_name = '^' * self.order + clean_name.lower() + '$'
                
                # Build the chain with weighted counts
                for i in range(len(padded_name) - self.order):
                    context = padded_name[i:i + self.order]
                    next_char = padded_name[i + self.order]
                    self.chains[context][next_char] += weight
    
    def generate(self, max_length: int = 12, min_length: int = 2, start_with: str = '', end_with: str = '') -> str:
        """
        Generate a new name using the trained Markov chain.
        
        Args:
            max_length: Maximum length of generated name
            min_length: Minimum length of generated name
            start_with: Optional string to start the name with
            end_with: Optional string to end the name with
            
        Returns:
            Generated name
        """
        if not self.chains:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Clean and validate the starting string
        start_with = start_with.lower().strip()
        if start_with and not self._is_valid_name_part(start_with):
            raise ValueError(f"Invalid starting string: '{start_with}'. Must contain only letters.")
        
        # Clean and validate the ending string
        end_with = end_with.lower().strip()
        if end_with and not self._is_valid_name_part(end_with):
            raise ValueError(f"Invalid ending string: '{end_with}'. Must contain only letters.")
        
        # If we have a starting string, use it to initialize
        if start_with:
            name = start_with
            # Build context from the end of the starting string
            if len(start_with) >= self.order:
                context = start_with[-self.order:]
            else:
                # Pad with start markers if starting string is shorter than order
                context = '^' * (self.order - len(start_with)) + start_with
        else:
            # Start with the beginning context
            context = '^' * self.order
            name = ''
        
        # Continue generating from where we left off
        for _ in range(max_length - len(name)):
            if context not in self.chains:
                # If we can't continue from current context, try to find a similar one
                if not self._try_alternative_context(context):
                    break
                
            # Get possible next characters and their weights
            possible_chars = self.chains[context]
            if not possible_chars:
                break
                
            # Choose next character based on probability
            chars, weights = zip(*possible_chars.items())
            next_char = random.choices(chars, weights=weights)[0]
            
            # End if we hit the end marker
            if next_char == '$':
                if len(name) >= min_length:
                    break
                else:
                    # Too short, try again if we started with a string
                    if start_with:
                        # Try a different continuation
                        return self.generate(max_length, min_length, start_with, end_with)
                    else:
                        # Reset completely
                        context = '^' * self.order
                        name = ''
                        continue
            
            name += next_char
            # Update context for next iteration
            context = context[1:] + next_char
        
        result = self._capitalize_name(name) if name else self.generate(max_length, min_length, start_with, end_with)
        # If we need a specific ending, retry until we get one
        if result and end_with and not result.lower().endswith(end_with):
            return self.generate(max_length, min_length, start_with, end_with)
        return result
    
    def _is_valid_name_part(self, text: str) -> bool:
        """Check if a text part is valid for names (contains only letters)."""
        return bool(re.match(r'^[a-zA-ZäöåÄÖÅ]+$', text.strip()))
    
    def _try_alternative_context(self, context: str) -> bool:
        """Try to find an alternative context when the current one doesn't exist."""
        # Try progressively shorter contexts
        for i in range(1, len(context)):
            shorter_context = context[i:]
            if shorter_context in self.chains:
                return True
        return False
    
    def _clean_name(self, name: str) -> str:
        """Clean and normalize a name."""
        # Remove extra whitespace and convert to title case
        name = re.sub(r'\s+', ' ', name.strip())
        return name.title()
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if a name is valid (contains only letters and basic punctuation)."""
        # Support Finnish (äöå) and Swedish (äöå) characters
        return bool(re.match(r'^[a-zA-ZäöåÄÖÅ\s\-\']+$', name.strip()))
    
    def _capitalize_name(self, name: str) -> str:
        """Properly capitalize a name."""
        return name.title()
    
    def get_stats(self) -> Dict:
        """Get statistics about the trained model."""
        return {
            'training_names_count': len(self.names),
            'unique_contexts': len(self.chains),
            'chain_order': self.order,
            'average_name_length': sum(len(name) for name in self.names) / len(self.names) if self.names else 0
        }


def load_names_from_csv(language: str, gender: str) -> Tuple[List[str], List[int]]:
    """
    Load names from CSV files with prevalence data.
    
    Args:
        language: 'finnish' or 'swedish'
        gender: 'male' or 'female'
    
    Returns:
        Tuple of (names list, prevalence weights list)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filename = f"{language}_{gender}.csv"
    csv_path = os.path.join(script_dir, "names", csv_filename)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    names = []
    weights = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            # Detect delimiter and quote character
            if language == 'finnish':
                reader = csv.reader(file)
            else:  # Swedish files use semicolon delimiter and different format
                reader = csv.reader(file, delimiter=';')
            
            for row_num, row in enumerate(reader):
                if row_num == 0 and language == 'swedish':
                    # Skip header row for Swedish files
                    continue
                    
                if len(row) >= 2:
                    name = row[0].strip().strip('"')
                    prevalence_str = row[1].strip().strip('"')
                    
                    # Skip empty names or header-like rows
                    if not name or name.lower() in ['f�rnamn', 'förnamn', 'name']:
                        continue
                    
                    try:
                        # Handle different prevalence formats
                        if ',' in prevalence_str:
                            # Finnish format: "29,887"
                            prevalence = int(prevalence_str.replace(',', ''))
                        else:
                            # Swedish format: 15
                            prevalence = int(prevalence_str)
                        
                        if prevalence > 0:  # Only include names with positive prevalence
                            names.append(name)
                            weights.append(prevalence)
                    except ValueError:
                        # Skip rows with invalid prevalence data
                        continue
                        
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(csv_path, 'r', encoding='latin-1') as file:
            if language == 'finnish':
                reader = csv.reader(file)
            else:
                reader = csv.reader(file, delimiter=';')
            
            for row_num, row in enumerate(reader):
                if row_num == 0 and language == 'swedish':
                    continue
                    
                if len(row) >= 2:
                    name = row[0].strip().strip('"')
                    prevalence_str = row[1].strip().strip('"')
                    
                    if not name or name.lower() in ['f�rnamn', 'förnamn', 'name']:
                        continue
                    
                    try:
                        if ',' in prevalence_str:
                            prevalence = int(prevalence_str.replace(',', ''))
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
        language: 'finnish' or 'swedish'
        gender: 'boys', 'girls', or 'both'
    
    Returns:
        Dictionary with gender as key and (names, weights) tuple as value
    """
    result = {}
    
    if gender in ['boys', 'both']:
        try:
            boys_names, boys_weights = load_names_from_csv(language, 'male')
            result['boys'] = (boys_names, boys_weights)
        except FileNotFoundError:
            print(f"Warning: Could not load {language} male names")
            result['boys'] = ([], [])
    
    if gender in ['girls', 'both']:
        try:
            girls_names, girls_weights = load_names_from_csv(language, 'female')
            result['girls'] = (girls_names, girls_weights)
        except FileNotFoundError:
            print(f"Warning: Could not load {language} female names")
            result['girls'] = ([], [])
    
    return result


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Generate Finnish or Swedish names using Markov chains')
    parser.add_argument('--language', choices=['finnish', 'swedish', 'both'], default='finnish',
                       help='Language of names to generate (use "both" for Finnish and Swedish combined)')
    parser.add_argument('--gender', choices=['boys', 'girls', 'both'], default='both',
                       help='Gender of names to generate')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of names to generate')
    parser.add_argument('--order', type=int, default=2,
                       help='Markov chain order (complexity)')
    parser.add_argument('--min-length', type=int, default=3,
                       help='Minimum name length')
    parser.add_argument('--max-length', type=int, default=12,
                       help='Maximum name length')
    parser.add_argument('--stats', action='store_true',
                       help='Show model statistics')
    parser.add_argument('--start', type=str, default='',
                       help='Starting string for generated names (e.g., "ju" for names starting with "ju")')
    parser.add_argument('--end', type=str, default='',
                       help='Ending string for generated names (e.g., "o" for names ending with "o")')
    parser.add_argument('--allow-duplicates', action='store_true',
                       help='Allow generating names that already exist in the training data')
    
    args = parser.parse_args()
    
    # Load the names with prevalence weights (one or both languages)
    try:
        if args.language == 'both':
            names_data_fi = load_names_by_language('finnish', args.gender)
            names_data_sv = load_names_by_language('swedish', args.gender)
            # Merge: for each gender, combine names and weights from both languages
            names_data = {}
            for gender in ['boys', 'girls']:
                names_fi, weights_fi = names_data_fi.get(gender, ([], []))
                names_sv, weights_sv = names_data_sv.get(gender, ([], []))
                names_data[gender] = (names_fi + names_sv, weights_fi + weights_sv)
        else:
            names_data = load_names_by_language(args.language, args.gender)
    except Exception as e:
        print(f"Error loading names: {e}")
        return
    
    # Language flags and display
    if args.language == 'both':
        language_flag = "🇫🇮🇸🇪"
        language_name = "Finnish & Swedish"
    else:
        language_flag = "🇫🇮" if args.language == "finnish" else "🇸🇪"
        language_name = args.language.capitalize()
    
    print(f"{language_flag} {language_name} Name Generator using Markov Chains")
    print("=" * 60)
    
    # Combine names and weights based on gender selection
    all_names = []
    all_weights = []
    total_names = 0
    
    for gender in ['boys', 'girls']:
        if gender in names_data and args.gender in [gender, 'both']:
            gender_names, gender_weights = names_data[gender]
            all_names.extend(gender_names)
            all_weights.extend(gender_weights)
            total_names += len(gender_names)
            
            if args.gender == 'both':
                print(f"Loaded {len(gender_names)} {language_name} {gender} names")
            else:
                print(f"Loaded {len(gender_names)} {language_name} {gender} names")
    
    if not all_names:
        print(f"No names found for {language_name} {args.gender}")
        return
    
    if args.gender == 'both':
        print(f"Total: {total_names} {language_name} names (boys and girls)")
    
    # Build per-language prevalence lookups (for --allow-duplicates display)
    finnish_prevalence: Dict[str, int] = {}
    swedish_prevalence: Dict[str, int] = {}
    if args.language == 'both':
        for gender in ['boys', 'girls']:
            n_fi, w_fi = names_data_fi.get(gender, ([], []))
            for n, w in zip(n_fi, w_fi):
                finnish_prevalence[n] = finnish_prevalence.get(n, 0) + w
            n_sv, w_sv = names_data_sv.get(gender, ([], []))
            for n, w in zip(n_sv, w_sv):
                swedish_prevalence[n] = swedish_prevalence.get(n, 0) + w
    elif args.language == 'finnish':
        for gender in ['boys', 'girls']:
            if gender in names_data:
                n, w = names_data[gender]
                for name, weight in zip(n, w):
                    finnish_prevalence[name] = finnish_prevalence.get(name, 0) + weight
    else:
        for gender in ['boys', 'girls']:
            if gender in names_data:
                n, w = names_data[gender]
                for name, weight in zip(n, w):
                    swedish_prevalence[name] = swedish_prevalence.get(name, 0) + weight
    
    # Create and train the generator with weighted data
    generator = MarkovNameGenerator(order=args.order)
    generator.train(all_names, all_weights)
    
    if args.stats:
        stats = generator.get_stats()
        total_weight = sum(all_weights)
        avg_weight = total_weight / len(all_weights) if all_weights else 0
        print(f"\nModel Statistics:")
        print(f"  Training names: {stats['training_names_count']}")
        print(f"  Total prevalence weight: {total_weight:,}")
        print(f"  Average prevalence: {avg_weight:.1f}")
        print(f"  Unique contexts: {stats['unique_contexts']}")
        print(f"  Chain order: {stats['chain_order']}")
        print(f"  Average name length: {stats['average_name_length']:.1f}")
        print()
    
    # Display generation info
    duplicate_info = " (including training data)" if args.allow_duplicates else ""
    parts = []
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
    
    # Generate names
    generated_names = set()
    attempts = 0
    max_attempts = args.count * 10  # Prevent infinite loops
    
    while len(generated_names) < args.count and attempts < max_attempts:
        try:
            name = generator.generate(
                max_length=args.max_length,
                min_length=args.min_length,
                start_with=args.start,
                end_with=args.end
            )
            if name:
                # Check if we should allow duplicates or not
                if args.allow_duplicates or name not in all_names:
                    generated_names.add(name)
        except ValueError as e:
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
        print(f"\nNote: Only generated {len(generated_names)} names after {max_attempts} attempts.")


if __name__ == "__main__":
    main()
