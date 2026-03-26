"""Visa tier data by passport country.

Curated visa accessibility tiers for destination suggestions.
Data should be periodically reviewed for accuracy.
"""

VISA_TIERS = {
    "IN": {
        "visa_free": [
            "Thailand", "Georgia", "Serbia", "Mauritius", "Nepal",
            "Bhutan", "Fiji", "Maldives", "Indonesia", "Qatar",
        ],
        "e_visa_easy": [
            "Vietnam", "Turkey", "Sri Lanka", "Cambodia", "Kenya",
            "Ethiopia", "Myanmar", "Laos", "Azerbaijan",
        ],
        "visa_required": ["Japan", "South Korea", "UAE", "Malaysia"],
        "hard_visa": ["US", "UK", "EU/Schengen", "Canada", "Australia", "NZ"],
    },
}
