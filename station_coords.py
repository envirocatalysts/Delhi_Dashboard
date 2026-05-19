"""Approximate lat/lon for Delhi CPCB / DPCC / IMD monitoring stations."""

from __future__ import annotations

import re

# (lat, lon) — CPCB open-data / map pins (approximate).
STATION_COORDS: dict[str, tuple[float, float]] = {
    "Alipur, Delhi - DPCC": (28.8150, 77.1410),
    "Anand Vihar, Delhi - DPCC": (28.6469, 77.3160),
    "Ashok Vihar, Delhi - DPCC": (28.6974, 77.1812),
    "Aya Nagar, Delhi - IMD": (28.4701, 77.1278),
    "Bawana, Delhi - DPCC": (28.7982, 77.0346),
    "Burari Crossing, Delhi - IMD": (28.7500, 77.1950),
    "CRRI Mathura Road, Delhi - IMD": (28.5400, 77.2700),
    "Chandni Chowk, Delhi - IITM": (28.6560, 77.2307),
    "DTU, Delhi - CPCB": (28.7500, 77.1170),
    "Dr. Karni Singh Shooting Range, Delhi - DPCC": (28.5020, 77.2930),
    "Dwarka-Sector 8, Delhi - DPCC": (28.5921, 77.0460),
    "IGI Airport (T3), Delhi - IMD": (28.5562, 77.0869),
    "IHBAS, Dilshad Garden, Delhi - CPCB": (28.6820, 77.3100),
    "IIT Delhi, Delhi - IITM": (28.5450, 77.1920),
    "ITO, Delhi - CPCB": (28.6289, 77.2405),
    "Jahangirpuri, Delhi - DPCC": (28.7285, 77.1662),
    "Jawaharlal Nehru Stadium, Delhi - DPCC": (28.5830, 77.2330),
    "Lodhi Road, Delhi - IITM": (28.5890, 77.2300),
    "Lodhi Road, Delhi - IMD": (28.5895, 77.2310),
    "Major Dhyan Chand National Stadium, Delhi - DPCC": (28.6120, 77.2370),
    "Mandir Marg, Delhi - DPCC": (28.6369, 77.2088),
    "Mundka, Delhi - DPCC": (28.6844, 77.0288),
    "NSIT Dwarka, Delhi - CPCB": (28.6090, 77.0390),
    "Najafgarh, Delhi - DPCC": (28.6130, 76.9850),
    "Narela, Delhi - DPCC": (28.8225, 77.0931),
    "Nehru Nagar, Delhi - DPCC": (28.5677, 77.2582),
    "North Campus, DU, Delhi - IMD": (28.6880, 77.2120),
    "Okhla Phase-2, Delhi - DPCC": (28.5300, 77.2700),
    "Patparganj, Delhi - DPCC": (28.6300, 77.2950),
    "Punjabi Bagh, Delhi - DPCC": (28.6683, 77.1422),
    "Pusa, Delhi - DPCC": (28.6370, 77.1480),
    "Pusa, Delhi - IMD": (28.6380, 77.1490),
    "R K Puram, Delhi - DPCC": (28.5706, 77.1844),
    "Rohini, Delhi - DPCC": (28.7340, 77.1025),
    "Shadipur, Delhi - CPCB": (28.6514, 77.1473),
    "Sirifort, Delhi - CPCB": (28.5506, 77.2030),
    "Sonia Vihar, Delhi - DPCC": (28.7200, 77.2800),
    "Sri Aurobindo Marg, Delhi - DPCC": (28.5280, 77.2100),
    "Vivek Vihar, Delhi - DPCC": (28.6710, 77.3120),
    "Wazirpur, Delhi - DPCC": (28.7028, 77.1726),
}

# Small nudge for co-located pairs (keep inside NCT — large offsets push pins outside border).
STATION_PLOT_OFFSETS: dict[str, tuple[float, float]] = {
    "Lodhi Road, Delhi - IITM": (-0.003, -0.004),
    "Lodhi Road, Delhi - IMD": (0.003, 0.004),
    "Pusa, Delhi - DPCC": (-0.002, -0.003),
    "Pusa, Delhi - IMD": (0.002, 0.003),
}

MAP_CENTER = {"lat": 28.652, "lon": 77.143}


def station_lat_lon(station: str) -> tuple[float, float]:
    if station in STATION_COORDS:
        return STATION_COORDS[station]
    key = re.sub(r"\s+", " ", str(station).strip().lower())
    for name, coords in STATION_COORDS.items():
        if name.lower() in key or key in name.lower():
            return coords
    # Deterministic jitter inside NCT Delhi for unknown labels.
    h = abs(hash(station)) % 10_000
    return (
        MAP_CENTER["lat"] + ((h % 97) - 48) * 0.003,
        MAP_CENTER["lon"] + (((h // 97) % 97) - 48) * 0.003,
    )
