stations = [
    {"name": "Arizona, US", "lat": 31.59, "lon": -110.51},
    {"name": "Fort Peck Montana, US", "lat": 48.31, "lon": -105.10},
    {"name": "Brookings Illinois, US", "lat": 44.35, "lon": -96.84},
    {"name": "Bondville Illinois, US", "lat": 40.01, "lon": -88.29},
    {"name": "Bondville comp. Illinois, US", "lat": 40.01, "lon": -88.29},
    {"name": "Gebesse, DE", "lat": 51.10, "lon": 10.91},
    {"name": "Ozark Missouri, US", "lat": 38.74, "lon": -92.20},
    {"name": "Morgan Monroe Indiana, US", "lat": 39.32, "lon": -86.41},
    {"name": "Collelongo beech, IT", "lat": 41.85, "lon": 13.59},
    {"name": "Hainich, DE", "lat": 51.08, "lon": 10.45},
    {"name": "Loobos, NL", "lat": 52.17, "lon": 5.74},
    {"name": "Le Brai, NL", "lat": 44.72, "lon": -0.77},
    {"name": "Black Hills South Dakota, US", "lat": 44.16, "lon": -103.65},
    {"name": "North Carolina, US", "lat": 35.98, "lon": -79.10},
]

def find_nearest_station(lat_array, lon_array, tolerance=0.3):
    """Trouve les indices (i, j) des pixels proches d'une station"""
    matches = []
    for station in stations:
        lat_diff = abs(lat_array - station["lat"])
        lon_diff = abs(lon_array - station["lon"])
        total_diff = lat_diff + lon_diff
        mask = (lat_diff < tolerance) & (lon_diff < tolerance)
        indices = list(zip(*mask.nonzero()))
        for i, j in indices:
            matches.append({
                "station": station["name"],
                "station_lat": station["lat"],
                "station_lon": station["lon"],
                "i": i,
                "j": j,
                "lat": lat_array[i, j],
                "lon": lon_array[i, j],
                "distance": total_diff[i, j],
            })
    return matches