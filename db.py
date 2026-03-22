import sqlite3

class HistoryDB:
    def __init__(self, db_path: str = "travel_history.db"):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS flight_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin TEXT NOT NULL,
                destination TEXT NOT NULL,
                departure_date TEXT NOT NULL,
                price REAL NOT NULL,
                currency TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS hotel_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                checkin_date TEXT NOT NULL,
                avg_nightly REAL NOT NULL,
                currency TEXT NOT NULL,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS social_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                destination TEXT NOT NULL,
                month INTEGER NOT NULL,
                timing_score REAL,
                crowd_level TEXT,
                events TEXT,
                itinerary_tips TEXT,
                sentiment TEXT,
                source TEXT,
                fetched_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)
        self._conn.commit()

    def save_flight(self, origin, destination, departure_date, price, currency):
        self._conn.execute(
            "INSERT INTO flight_prices (origin, destination, departure_date, price, currency) VALUES (?, ?, ?, ?, ?)",
            (origin, destination, departure_date, price, currency))
        self._conn.commit()

    def get_flight(self, origin, destination, departure_date, tolerance_days=0):
        if tolerance_days > 0:
            row = self._conn.execute(
                "SELECT price, currency, fetched_at FROM flight_prices "
                "WHERE origin=? AND destination=? AND ABS(julianday(departure_date)-julianday(?))<=? "
                "ORDER BY ABS(julianday(departure_date)-julianday(?)) ASC, fetched_at DESC LIMIT 1",
                (origin, destination, departure_date, tolerance_days, departure_date)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT price, currency, fetched_at FROM flight_prices "
                "WHERE origin=? AND destination=? AND departure_date=? ORDER BY fetched_at DESC LIMIT 1",
                (origin, destination, departure_date)).fetchone()
        return dict(row) if row else None

    def save_hotel(self, city, checkin_date, avg_nightly, currency):
        self._conn.execute(
            "INSERT INTO hotel_prices (city, checkin_date, avg_nightly, currency) VALUES (?, ?, ?, ?)",
            (city, checkin_date, avg_nightly, currency))
        self._conn.commit()

    def get_hotel(self, city, checkin_date, tolerance_days=0):
        if tolerance_days > 0:
            row = self._conn.execute(
                "SELECT avg_nightly, currency, fetched_at FROM hotel_prices "
                "WHERE city=? AND ABS(julianday(checkin_date)-julianday(?))<=? "
                "ORDER BY ABS(julianday(checkin_date)-julianday(?)) ASC, fetched_at DESC LIMIT 1",
                (city, checkin_date, tolerance_days, checkin_date)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT avg_nightly, currency, fetched_at FROM hotel_prices "
                "WHERE city=? AND checkin_date=? ORDER BY fetched_at DESC LIMIT 1",
                (city, checkin_date)).fetchone()
        return dict(row) if row else None

    def save_social(self, destination, month, timing_score, crowd_level,
                    events, itinerary_tips, sentiment, source):
        self._conn.execute(
            "INSERT INTO social_insights (destination, month, timing_score, crowd_level, "
            "events, itinerary_tips, sentiment, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (destination, month, timing_score, crowd_level, events, itinerary_tips, sentiment, source))
        self._conn.commit()

    def get_social(self, destination, month, tolerance_months=0):
        if tolerance_months > 0:
            row = self._conn.execute(
                "SELECT timing_score, crowd_level, events, itinerary_tips, sentiment, source, fetched_at "
                "FROM social_insights WHERE destination=? AND ABS(month - ?) <= ? "
                "ORDER BY ABS(month - ?) ASC, fetched_at DESC LIMIT 1",
                (destination, month, tolerance_months, month)).fetchone()
        else:
            row = self._conn.execute(
                "SELECT timing_score, crowd_level, events, itinerary_tips, sentiment, source, fetched_at "
                "FROM social_insights WHERE destination=? AND month=? "
                "ORDER BY fetched_at DESC LIMIT 1",
                (destination, month)).fetchone()
        return dict(row) if row else None

    def close(self):
        self._conn.close()
