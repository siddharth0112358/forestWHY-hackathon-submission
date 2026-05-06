"""Watched deforestation hotspots — 18 locations across the major tropical biomes.

Coordinates point to the centre of a 5 km tile. They are chosen near known
deforestation fronts (Hansen GFC loss hotspots and PRODES alerts), but the
exact pixel does not matter — `predict.py` matches the satellite's current
position against a tolerance derived from --size-km.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    id: str
    lon: float
    lat: float
    biome: str
    country: str
    description: str


LOCATIONS: tuple[Location, ...] = (
    Location("amazon_acre",          -68.4000,  -9.1000, "amazon",         "Brazil",      "Acre arc-of-deforestation, BR-364 frontier."),
    Location("amazon_para",          -53.0000,  -3.5000, "amazon",         "Brazil",      "Pará Trans-Amazon Highway clearing front."),
    Location("amazon_rondonia",      -62.0000, -10.7000, "amazon",         "Brazil",      "Rondônia fishbone settlement pattern."),
    Location("amazon_mato_grosso",   -54.6000, -11.5000, "amazon",         "Brazil",      "Mato Grosso soy/cattle expansion edge."),
    Location("amazon_madre_dios",    -71.0000, -12.5000, "amazon",         "Peru",        "Madre de Dios artisanal gold mining."),
    Location("amazon_beni",          -65.0000, -14.0000, "amazon",         "Bolivia",     "Beni lowlands cattle frontier."),
    Location("borneo_kalimantan",    113.0000,  -1.5000, "borneo",         "Indonesia",   "Central Kalimantan oil palm conversion."),
    Location("borneo_sabah",         117.0000,   5.5000, "borneo",         "Malaysia",    "Sabah selective logging concessions."),
    Location("sumatra_riau",         102.5000,  -1.0000, "sumatra",        "Indonesia",   "Riau peatland drainage and pulp plantations."),
    Location("papua_indonesia",      138.5000,  -3.0000, "new_guinea",     "Indonesia",   "Indonesian Papua oil-palm and road expansion."),
    Location("png_western",          144.0000,  -5.5000, "new_guinea",     "Papua New Guinea", "Western Province logging concessions."),
    Location("congo_equateur",        21.0000,   1.0000, "congo_basin",    "DRC",         "Equateur shifting cultivation belt."),
    Location("congo_tshopo",          25.0000,   0.5000, "congo_basin",    "DRC",         "Tshopo logging road expansion."),
    Location("madagascar_east",       48.5000, -18.5000, "madagascar",     "Madagascar",  "Eastern rainforest slash-and-burn (tavy)."),
    Location("cerrado_centre",       -47.0000, -13.0000, "cerrado",        "Brazil",      "Cerrado MATOPIBA agricultural frontier."),
    Location("choco_colombia",       -76.5000,   5.5000, "choco",          "Colombia",    "Chocó Pacific rainforest small-scale mining."),
    Location("cambodia_mondulkiri",  107.0000,  12.5000, "indochina",      "Cambodia",    "Mondulkiri rubber concession edge."),
    Location("atlantic_forest",      -39.5000, -16.0000, "atlantic_forest","Brazil",      "Bahia Atlantic Forest fragment loss."),
)


LOCATIONS_BY_ID: dict[str, Location] = {loc.id: loc for loc in LOCATIONS}
