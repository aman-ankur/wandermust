"""Destination seasonality data.

Best and avoid months for top destinations relevant to Indian travelers.
"""

SEASONALITY = {
    "Thailand": {"best": [11, 12, 1, 2, 3], "avoid": [4, 5], "note": "Nov-Mar dry season"},
    "Georgia": {"best": [5, 6, 7, 8, 9], "avoid": [12, 1, 2], "note": "Summer wine harvest, warm weather"},
    "Vietnam": {"best": [2, 3, 4, 10, 11], "avoid": [7, 8, 9], "note": "Varies by region, central coast best in spring"},
    "Turkey": {"best": [4, 5, 6, 9, 10], "avoid": [12, 1, 2], "note": "Spring and autumn ideal, summer hot"},
    "Sri Lanka": {"best": [1, 2, 3, 4, 12], "avoid": [5, 6], "note": "West coast best Dec-Apr, east coast May-Sep"},
    "Nepal": {"best": [10, 11, 3, 4], "avoid": [6, 7, 8], "note": "Oct-Nov best for trekking, monsoon Jun-Aug"},
    "Maldives": {"best": [1, 2, 3, 4, 12], "avoid": [6, 7, 8], "note": "Dry season Dec-Apr, monsoon May-Oct"},
    "Indonesia": {"best": [5, 6, 7, 8, 9], "avoid": [12, 1, 2], "note": "Dry season May-Sep, Bali best Jun-Aug"},
    "Cambodia": {"best": [11, 12, 1, 2, 3], "avoid": [5, 6, 7], "note": "Cool dry season Nov-Feb"},
    "Japan": {"best": [3, 4, 10, 11], "avoid": [6, 7, 8], "note": "Cherry blossom Mar-Apr, autumn foliage Oct-Nov"},
    "South Korea": {"best": [4, 5, 9, 10], "avoid": [7, 8], "note": "Spring cherry blossom, autumn foliage"},
    "Malaysia": {"best": [3, 4, 5, 6, 9], "avoid": [11, 12], "note": "West coast best Dec-Apr, Borneo Mar-Oct"},
    "Azerbaijan": {"best": [4, 5, 6, 9, 10], "avoid": [12, 1, 2], "note": "Spring and autumn mild, winter cold"},
    "Serbia": {"best": [5, 6, 9, 10], "avoid": [12, 1, 2], "note": "Summer music festivals, mild spring/autumn"},
    "Mauritius": {"best": [5, 6, 7, 8, 9, 10], "avoid": [1, 2, 3], "note": "Winter (May-Oct) dry and mild"},
    "Kenya": {"best": [1, 2, 7, 8, 9, 10], "avoid": [4, 5], "note": "Great Migration Jul-Oct, dry Jan-Feb"},
    "Dubai": {"best": [11, 12, 1, 2, 3], "avoid": [6, 7, 8], "note": "Winter pleasant, summer extreme heat"},
    "Qatar": {"best": [11, 12, 1, 2, 3], "avoid": [6, 7, 8], "note": "Winter mild, summer extreme heat"},
    "Mexico": {"best": [11, 12, 1, 2, 3, 4], "avoid": [6, 7, 8, 9], "note": "Dry season Nov-Apr, hurricane season Jun-Nov"},
    "Bhutan": {"best": [3, 4, 5, 10, 11], "avoid": [6, 7, 8], "note": "Spring festivals, autumn clear skies"},
    "Fiji": {"best": [5, 6, 7, 8, 9, 10], "avoid": [1, 2, 3], "note": "Dry season May-Oct"},
}
