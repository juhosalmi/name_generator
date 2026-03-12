import json
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set


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
        # Feedback labels from reinforcement learning sessions, grouped by state.
        # JSON schema:
        #   "feedback": {
        #       "accepted": ["Name1", "Name2"],
        #       "skipped": ["Name3"],
        #       "rejected": ["Name4"]
        #   }
        self.feedback: Dict[str, Set[str]] = {
            "accepted": set(),
            "skipped": set(),
            "rejected": set(),
        }

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

    def apply_feedback(self, name: str, factor: float) -> None:
        """
        Apply reinforcement feedback to the model for a single name.

        A factor greater than 1.0 strengthens the transitions that produced the
        name, while a factor between 0 and 1.0 weakens them. A factor of 1.0
        leaves the model unchanged.

        Args:
            name: The generated name to reinforce.
            factor: Multiplicative factor to apply to each transition in the
                    name. Must be positive and non-zero.
        """
        if factor == 1.0 or factor <= 0.0:
            return

        if not self._is_valid_name(name):
            return

        clean_name = self._clean_name(name)
        padded_name = "^" * self.order + clean_name.lower() + "$"

        for i in range(len(padded_name) - self.order):
            context = padded_name[i : i + self.order]
            next_char = padded_name[i + self.order]
            counter = self.chains[context]
            current = float(counter.get(next_char, 0.0)) or 1.0
            new_value = current * factor
            if new_value <= 0.0:
                del counter[next_char]
                if not counter:
                    # Remove empty contexts to keep the model compact.
                    del self.chains[context]
            else:
                counter[next_char] = new_value

        if factor > 1.0:
            # Track positively reinforced names for statistics.
            self.names.append(clean_name)

    def reinforce_accept(self, name: str, reward: float = 1.0) -> None:
        """
        Reinforce the model to make a given name more likely in the future.

        Args:
            name: The accepted name.
            reward: Reward factor to apply (default: 1.0). Values > 1.0 make
                    the name more likely; values <= 0 are ignored.
        """
        if reward <= 0.0:
            return
        self.apply_feedback(name, factor=reward)

    def reinforce_reject(self, name: str, reward: float = 1.0) -> None:
        """
        Reinforce the model to make a given name less likely in the future.

        Args:
            name: The rejected name.
            reward: Penalty factor to apply (default: 1.0). Values > 1.0 make
                    the name less likely by dividing the transition weights by
                    this factor; values <= 0 are ignored.
        """
        if reward <= 0.0:
            return
        # Dividing by the reward factor weakens the transitions.
        self.apply_feedback(name, factor=1.0 / reward)

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

    def to_dict(self) -> Dict:
        """
        Serialize the model state to a JSON-serializable dictionary.
        """
        return {
            "order": self.order,
            "names": self.names,
            "chains": {
                context: dict(counter) for context, counter in self.chains.items()
            },
            "feedback": {
                label: sorted(list(names))
                for label, names in self.feedback.items()
                if names
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MarkovNameGenerator":
        """
        Reconstruct a MarkovNameGenerator instance from a dictionary
        produced by to_dict().
        """
        order = int(data.get("order", 2))
        instance = cls(order=order)
        instance.names = list(data.get("names", []))

        chains_data = data.get("chains", {})
        instance.chains = defaultdict(Counter)
        for context, next_chars in chains_data.items():
            instance.chains[context] = Counter(next_chars)

        # Load any stored reinforcement feedback labels.
        raw_feedback = data.get("feedback", {})
        categories = {"accepted", "skipped", "rejected"}
        instance.feedback = {c: set() for c in categories}

        if isinstance(raw_feedback, dict):
            # New schema: {"accepted": [...], "skipped": [...], "rejected": [...]}
            if any(k in categories for k in raw_feedback.keys()):
                for label in categories:
                    names = raw_feedback.get(label, [])
                    if isinstance(names, list):
                        instance.feedback[label].update(str(n) for n in names)
            else:
                # Backwards compatibility: old schema {name: "accepted" | "skipped" | "rejected"}
                for name, label in raw_feedback.items():
                    if label in categories:
                        instance.feedback[label].add(str(name))

        return instance

    def save_to_json(self, path: str) -> None:
        """Save the model state to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_from_json(cls, path: str) -> "MarkovNameGenerator":
        """Load the model state from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

