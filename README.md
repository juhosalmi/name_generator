# Finnish Name Generator 🇫🇮

A Python application that uses Markov chains to generate authentic-sounding Finnish names for children.

## How It Works

The application uses Markov chains to analyze patterns in existing Finnish names and generate new names that follow similar phonetic and structural patterns. A Markov chain predicts the next character in a name based on the previous N characters (where N is the "order" of the chain).

## Features

- Generate both boys' and girls' names
- Configurable Markov chain order for different creativity levels
- Adjustable name length constraints
- Statistics about the training data
- Avoids generating exact duplicates of training names

## Usage

### Basic Usage

```bash
# Generate 10 names (both boys and girls)
python main.py

# Generate only girls' names
python main.py --gender girls

# Generate only boys' names  
python main.py --gender boys

# Generate 20 names
python main.py --count 20
```

### Advanced Options

```bash
# Use higher order chain for more realistic names (less creative)
python main.py --order 3

# Use lower order chain for more creative names (less realistic)
python main.py --order 1

# Set name length constraints
python main.py --min-length 4 --max-length 8

# Show model statistics
python main.py --stats
```

### Full Options

```
usage: main.py [-h] [--gender {boys,girls,both}] [--count COUNT] [--order ORDER]
               [--min-length MIN_LENGTH] [--max-length MAX_LENGTH] [--stats]

optional arguments:
  -h, --help            show this help message and exit
  --gender {boys,girls,both}
                        Gender of names to generate (default: both)
  --count COUNT         Number of names to generate (default: 10)
  --order ORDER         Markov chain order/complexity (default: 2)
  --min-length MIN_LENGTH
                        Minimum name length (default: 3)
  --max-length MAX_LENGTH
                        Maximum name length (default: 12)
  --stats               Show model statistics
```

## Examples

### Sample Generated Names

**Boys:**
- Kalle
- Miilo
- Veelis
- Tommi

**Girls:**
- Liina
- Emmi
- Villia
- Aura

## Technical Details

- **Markov Chain Order**: Controls how many previous characters influence the next character
  - Order 1: More random/creative but less realistic
  - Order 2: Good balance (default)
  - Order 3+: More realistic but less creative
  
- **Training Data**: Uses a curated list of authentic Finnish names
- **Character Handling**: Supports Finnish characters (ä, ö, å)
- **Name Validation**: Ensures generated names follow Finnish naming patterns

## Requirements

- Python 3.7+
- No external dependencies (uses only standard library)

## File Structure

```
finnish_name_generator/
├── main.py           # Main application
├── requirements.txt  # Dependencies (empty - uses stdlib only)
└── README.md        # This file
```

## Future Enhancements

Potential improvements could include:
- Loading names from external files or databases
- Web interface for easier use
- Regional Finnish name variations
- Name meaning generation
- Export generated names to files
- Machine learning evaluation metrics
