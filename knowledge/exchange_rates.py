"""Exchange rate favorability by passport country.

Helps frame budget expectations relative to home currency.
"""

EXCHANGE_FAVORABILITY = {
    "IN": {
        "great_value": ["THB", "GEL", "VND", "LKR", "NPR", "KHR", "IDR"],
        "decent_value": ["TRY", "MYR", "AZN", "RSD", "KES"],
        "poor_value": ["EUR", "GBP", "USD", "AUD", "JPY", "CHF", "SGD", "NZD"],
        "note": "INR to EUR/GBP/USD is unfavorable — European/US trips cost 3-4x more than SE Asia equivalent",
    },
}
