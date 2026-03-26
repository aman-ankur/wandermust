"""Budget tier data by passport country.

Daily budget ranges in local currency with example destinations.
"""

BUDGET_TIERS = {
    "IN": {
        "budget": {
            "daily_range": "₹2,000-4,000",
            "examples": ["Vietnam", "Nepal", "Georgia", "Cambodia", "Sri Lanka"],
            "note": "Hostels, street food, local transport",
        },
        "moderate": {
            "daily_range": "₹5,000-8,000",
            "examples": ["Thailand", "Turkey", "Bali", "Malaysia", "Azerbaijan"],
            "note": "Mid-range hotels, restaurants, some activities",
        },
        "comfortable": {
            "daily_range": "₹10,000-15,000",
            "examples": ["Japan", "South Korea", "Dubai", "Eastern Europe"],
            "note": "Good hotels, full experiences, domestic flights",
        },
        "luxury": {
            "daily_range": "₹15,000+",
            "examples": ["Western Europe", "Australia", "Scandinavia", "Switzerland"],
            "note": "INR exchange hurts here — budget 3-4x more than SE Asia",
        },
    },
}
