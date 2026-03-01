# Finnish & Swedish Name Generator 🇫🇮🇸🇪

A Python application that uses Markov chains to generate authentic-sounding Finnish and Swedish names for children, with prevalence-based weighting from real name databases.

## How It Works

The application uses Markov chains to analyze patterns in existing Finnish and Swedish names and generate new names that follow similar phonetic and structural patterns. A Markov chain predicts the next character in a name based on the previous N characters (where N is the "order" of the chain).

The system loads real name data from CSV files containing names and their prevalence (popularity) in the respective countries. Names with higher prevalence have proportionally more influence on the generation patterns, making the output more realistic and culturally authentic.

## Features

- **Multi-language Support**: Generate Finnish, Swedish, or both languages combined
- **Gender Selection**: Generate boys', girls', or both genders
- **Prevalence Weighting**: Real name popularity data influences generation patterns
- **Starting String**: Generate names that start with specific letters or syllables
- **Ending String**: Generate names that end with specific letters or syllables
- **Configurable Markov Chain**: Adjustable order for different creativity levels
- **Length Constraints**: Set minimum and maximum name lengths
- **Rich Statistics**: View detailed information about training data
- **Model Caching**: Reuses trained Markov models between runs for faster startup
- **Duplicate Avoidance**: Prevents generating exact copies of training names
- **Character Support**: Full support for Nordic characters (ä, ö, å)
- **Interactive Reinforcement Learning**: Accept/reject generated names to fine-tune the model
- **Custom Pretrained Models**: Save and load Markov models as JSON files

## Usage

### Basic Usage

```bash
# Generate 10 Finnish names (both boys and girls)
python3 name_generator.py

# Generate Finnish girls' names
python3 name_generator.py --gender girls

# Generate Swedish boys' names  
python3 name_generator.py --language swedish --gender boys

# Generate names using both Finnish and Swedish (blended patterns)
python3 name_generator.py --language both --count 10

# Generate 20 Finnish names
python3 name_generator.py --count 20

# Generate Swedish names starting with "an"
python3 name_generator.py --language swedish --start an --count 5

# Generate names ending with "o"
python3 name_generator.py --end o --count 5
```

### Advanced Options

```bash
# Use higher order chain for more realistic names (less creative)
python3 name_generator.py --order 3

# Use lower order chain for more creative names (less realistic)
python3 name_generator.py --order 1

# Set name length constraints
python3 name_generator.py --min-length 4 --max-length 8

# Show detailed model statistics
python3 name_generator.py --stats

# Generate names starting or ending with specific letters
python3 name_generator.py --start j --end o

# Allow names that appear in the training data (default is to exclude them)
python3 name_generator.py --allow-duplicates

# Always retrain the model and skip loading/saving cache
python3 name_generator.py --no-cache

# Force retraining and overwrite any existing cached model
python3 name_generator.py --force-retrain

# Combine multiple options
python3 name_generator.py --language swedish --gender girls --start ma --count 5 --stats

# Enable interactive reinforcement learning with stronger feedback
python3 name_generator.py --language finnish --gender boys --reinforce --reward 3 --count 20

# Train (or load) and then save a custom pretrained model
python3 name_generator.py --language finnish --gender boys --count 20 --save-model models/my_fi_boys.json

# Load a custom pretrained model and generate names without retraining
python3 name_generator.py --load-model models/my_fi_boys.json --count 10
```

### Full Options

```
usage: name_generator.py [-h] [--language {finnish,swedish,both}] [--gender {boys,girls,both}]
               [--count COUNT] [--order ORDER] [--min-length MIN_LENGTH]
               [--max-length MAX_LENGTH] [--stats] [--start START] [--end END]
               [--allow-duplicates] [--no-cache] [--force-retrain]
               [--reinforce] [--reward REWARD]
               [--load-model LOAD_MODEL] [--save-model SAVE_MODEL]

Generate Finnish or Swedish names using Markov chains

optional arguments:
  -h, --help            show this help message and exit
  --language {finnish,swedish,both}
                        Language of names to generate; "both" trains on Finnish and Swedish together (default: finnish)
  --gender {boys,girls,both}
                        Gender of names to generate (default: both)
  --count COUNT         Number of names to generate (default: 10)
  --order ORDER         Markov chain order/complexity (default: 2)
  --min-length MIN_LENGTH
                        Minimum name length (default: 3)
  --max-length MAX_LENGTH
                        Maximum name length (default: 12)
  --stats               Show model statistics
  --start START         Starting string for generated names (e.g., "ju")
  --end END             Ending string for generated names (e.g., "o")
  --allow-duplicates    Allow generating names that already exist in the training data
  --no-cache            Disable model caching (always retrain; do not read/write cache)
  --force-retrain       Force retraining and overwrite any existing cached model
  --reinforce, -R       Enable interactive reinforcement learning (accept/reject each generated name)
  --reward REWARD       Base reinforcement reward magnitude; acceptance applies +reward
                        and rejection applies -reward to the Markov transition counts (default: 1)
  --load-model LOAD_MODEL
                        Path to a custom pretrained Markov model JSON file to load instead
                        of training from CSV data
  --save-model SAVE_MODEL
                        Path to save the trained/reinforced Markov model as a JSON file
                        (custom pretrained model)
```

## Examples

### Sample Generated Names

**Finnish Boys:**
- Juhannismo, Juhari, Jukki, Juuti (starting with "ju")
- Aaro, Markko, Ville, Hannu

**Finnish Girls:**
- Aada, Emilia, Helmi, Venla
- Aarkiena, Anneti, Almani (starting with "a")

**Swedish Boys:**
- Alexand, Alexandertin, Alexanus (starting with "alex")
- Erik, Gustaf, Magnus, Nils

**Swedish Girls:**
- Anittargarin, Ankaa, Annelisabeth (starting with "an")
- Astrid, Ingrid, Margareta, Sofia

### Sample Statistics Output

```
🇫🇮 Finnish Name Generator using Markov Chains
============================================================
Loaded 7209 Finnish boys names

Model Statistics:
  Training names: 7102
  Total prevalence weight: 2,643,974
  Average prevalence: 366.8
  Unique contexts: 610
  Chain order: 2
  Average name length: 6.3
```

## Technical Details

- **Markov Chain Order**: Controls how many previous characters influence the next character
  - Order 1: More random/creative but less realistic
  - Order 2: Good balance (default)
  - Order 3+: More realistic but less creative
  
- **Training Data**: Uses real name databases from CSV files
  - Finnish: 7,200+ male names, 9,500+ female names
  - Swedish: 19,400+ male names, 22,000+ female names
  
- **Prevalence Weighting**: Names with higher popularity have more influence
  - A name with prevalence 1000 has 10x more impact than prevalence 100
  - Ensures generated names follow patterns of common names
  
- **Character Support**: Full Nordic character support (ä, ö, å)
- **Starting String**: Generate names beginning with specific patterns
- **Ending String**: Generate names ending with specific patterns (retries until a matching name is found)
- **Context Handling**: Smart fallback for missing character combinations
- **Model Caching**: Trains a model once per (language, gender, order, dataset) combination
  and then reuses the cached model on subsequent runs unless `--no-cache` or `--force-retrain`
  are used
  
- **Interactive Reinforcement Learning**:
  - Use `--reinforce` to enter an interactive loop where the program suggests one name at a time.
  - For each suggestion, type `a` to accept, `r` to reject, or `q` to quit.
  - The `--reward` option controls how strongly each piece of feedback shifts the underlying
    Markov transition probabilities (acceptance applies +reward, rejection applies -reward).
  - When caching is enabled, the adapted model can be stored back into the dataset-based cache;
    you can also save it explicitly via `--save-model`.

- **Custom Pretrained Models**:
  - Use `--save-model PATH` to persist the current model (after training and optionally
    reinforcement) as a JSON file.
  - Use `--load-model PATH` to load such a custom pretrained model in a later run instead of
    retraining from CSVs.
  - When `--load-model` is used, the Markov chain order is taken from the saved model; the
    `--order` argument is ignored in this case.

## Requirements

- Python 3.7+
- No external dependencies (uses only standard library)

## Project Structure

```
name_generator/
├── name_generator.py      # Thin façade & CLI entry point
├── markov_generator.py    # Core Markov-chain name generation logic
├── data_loader.py         # CSV loading & prevalence utilities
├── cli.py                 # Argument parsing & command-line orchestration
├── requirements.txt       # Dependencies (empty - uses stdlib only)
├── README.md              # This file
└── names/                 # Name databases
    ├── finnish_female.csv # Finnish female names with prevalence
    ├── finnish_male.csv   # Finnish male names with prevalence
    ├── swedish_female.csv # Swedish female names with prevalence
    └── swedish_male.csv   # Swedish male names with prevalence
```

## Data Sources

The name databases include authentic names with their popularity/prevalence data:

- **Finnish Names**: Based on official Finnish name registry data
- **Swedish Names**: Based on official Swedish name registry data

Each CSV file contains names in the first column and prevalence counts in the second column, representing how many people have that name in the respective country.

## Future Enhancements

Potential improvements could include:
- Additional Nordic languages (Norwegian, Danish, Icelandic)
- Web interface for easier use
- Regional name variations and dialects
- Name meaning and etymology generation
- Export generated names to files
- Advanced filtering options (popularity ranges, cultural regions)
- Machine learning evaluation metrics
