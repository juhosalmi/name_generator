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
- **Duplicate Avoidance**: Prevents generating exact copies of training names
- **Character Support**: Full support for Nordic characters (ä, ö, å)

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

# Generate names starting with specific letters
python3 name_generator.py --start "ju" --count 8

# Generate names ending with specific letters
python3 name_generator.py --end "o" --count 8

# Combine start and end (e.g. names starting with "j" and ending with "o")
python3 name_generator.py --start j --end o --count 5

# Combine multiple options
python3 name_generator.py --language swedish --gender girls --start "ma" --count 5 --stats
```

### Full Options

```
usage: name_generator.py [-h] [--language {finnish,swedish}] [--gender {boys,girls,both}] 
               [--count COUNT] [--order ORDER] [--min-length MIN_LENGTH] 
               [--max-length MAX_LENGTH] [--stats] [--start START] [--end END]

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

## Requirements

- Python 3.7+
- No external dependencies (uses only standard library)

## File Structure

```
name_generator/
├── name_generator.py              # Main application
├── requirements.txt     # Dependencies (empty - uses stdlib only)
├── README.md           # This file
└── names/              # Name databases
    ├── finnish_female.csv   # Finnish female names with prevalence
    ├── finnish_male.csv     # Finnish male names with prevalence
    ├── swedish_female.csv   # Swedish female names with prevalence
    └── swedish_male.csv     # Swedish male names with prevalence
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
