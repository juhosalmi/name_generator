import random
import re
from collections import defaultdict, Counter
from typing import List, Dict, Optional


class MarkovNameGenerator:
    """
    A Markov chain-based name generator for Finnish and Swedish names.
    """

    def __init__(self, order: int = 2):
        """
        Initialize the Markov chain generator.

        Args:
            order: The order of the Markov chain (number of characters to
                   consider for next character prediction).
        """
        self.order = order
        self.chains = defaultdict(Counter)
        self.names: List[str] = []

    def train(self, names: List[str], weights: Optional[List[int]] = None) -> None:
        """
        Train the Markov chain on a list of names with optional weights.

        Args:
            names: List of names to train on.
            weights: Optional list of weights (prevalence) for each name.
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

                # Add start and end markers.
                padded_name = "^" * self.order + clean_name.lower() + "$"

                # Build the chain with weighted counts.
                for i in range(len(padded_name) - self.order):
                    context = padded_name[i : i + self.order]
                    next_char = padded_name[i + self.order]
                    self.chains[context][next_char] += weight

    def generate(
        self,
        max_length: int = 12,
        min_length: int = 2,
        start_with: str = "",
        end_with: str = "",
    ) -> str:
        """
        Generate a new name using the trained Markov chain.

        Args:
            max_length: Maximum length of generated name.
            min_length: Minimum length of generated name.
            start_with: Optional string to start the name with.
            end_with: Optional string to end the name with.

        Returns:
            Generated name.
        """
        if not self.chains:
            raise ValueError("Model not trained yet. Call train() first.")

        # Clean and validate the starting string.
        start_with = start_with.lower().strip()
        if start_with and not self._is_valid_name_part(start_with):
            raise ValueError(
                f"Invalid starting string: '{start_with}'. Must contain only letters."
            )

        # Clean and validate the ending string.
        end_with = end_with.lower().strip()
        if end_with and not self._is_valid_name_part(end_with):
            raise ValueError(
                f"Invalid ending string: '{end_with}'. Must contain only letters."
            )

        # If we have a starting string, use it to initialize.
        if start_with:
            name = start_with
            # Build context from the end of the starting string.
            if len(start_with) >= self.order:
                context = start_with[-self.order :]
            else:
                # Pad with start markers if starting string is shorter than order.
                context = "^" * (self.order - len(start_with)) + start_with
        else:
            # Start with the beginning context.
            context = "^" * self.order
            name = ""

        # Continue generating from where we left off.
        for _ in range(max_length - len(name)):
            if context not in self.chains:
                # If we can't continue from current context, try to find a similar one.
                if not self._try_alternative_context(context):
                    break

            # Get possible next characters and their weights.
            possible_chars = self.chains[context]
            if not possible_chars:
                break

            # Choose next character based on probability.
            chars, weights = zip(*possible_chars.items())
            next_char = random.choices(chars, weights=weights)[0]

            # End if we hit the end marker.
            if next_char == "$":
                if len(name) >= min_length:
                    break
                else:
                    # Too short, try again if we started with a string.
                    if start_with:
                        # Try a different continuation.
                        return self.generate(max_length, min_length, start_with, end_with)
                    else:
                        # Reset completely.
                        context = "^" * self.order
                        name = ""
                        continue

            name += next_char
            # Update context for next iteration.
            context = context[1:] + next_char

        result = self._capitalize_name(name) if name else self.generate(
            max_length, min_length, start_with, end_with
        )
        # If we need a specific ending, retry until we get one.
        if result and end_with and not result.lower().endswith(end_with):
            return self.generate(max_length, min_length, start_with, end_with)
        return result

    def _is_valid_name_part(self, text: str) -> bool:
        """Check if a text part is valid for names (contains only letters)."""
        return bool(re.match(r"^[a-zA-ZäöåÄÖÅ]+$", text.strip()))

    def _try_alternative_context(self, context: str) -> bool:
        """Try to find an alternative context when the current one doesn't exist."""
        # Try progressively shorter contexts.
        for i in range(1, len(context)):
            shorter_context = context[i:]
            if shorter_context in self.chains:
                return True
        return False

    def _clean_name(self, name: str) -> str:
        """Clean and normalize a name."""
        # Remove extra whitespace and convert to title case.
        name = re.sub(r"\s+", " ", name.strip())
        return name.title()

    def _is_valid_name(self, name: str) -> bool:
        """Check if a name is valid (contains only letters and basic punctuation)."""
        # Support Finnish (äöå) and Swedish (äöå) characters.
        return bool(re.match(r"^[a-zA-ZäöåÄÖÅ\s\-']+$", name.strip()))

    def _capitalize_name(self, name: str) -> str:
        """Properly capitalize a name."""
        return name.title()

    def get_stats(self) -> Dict:
        """Get statistics about the trained model."""
        return {
            "training_names_count": len(self.names),
            "unique_contexts": len(self.chains),
            "chain_order": self.order,
            "average_name_length": (
                sum(len(name) for name in self.names) / len(self.names)
                if self.names
                else 0
            ),
        }

