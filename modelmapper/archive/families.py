"""Coarse concept families for 'related vs unrelated' comparisons (fixed 16 Oct 2026 BEFORE the cross-activation and
corpus-profile results existed). Wave categories are mapped onto shared families; mixed categories are split by
concept; the 19 CR-2 concepts (which have no category in prompts/) are assigned by hand.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CATEGORY = {
    "animals": "animals", "animals_2": "animals",
    "food_drink": "food", "food_2": "food",
    "sport_games": "sport", "sport_2": "sport",
    "science": "science", "science_math": "science",
    "technology": "tech", "tech_2": "tech", "vehicles_machines": "tech",
    "abstract": "abstract", "emotion_abstract": "abstract", "abstract_social": "abstract",
    "body_medicine": "body",
    "arts": "arts", "art_music": "arts", "music_art_craft": "arts",
    "nature": "nature", "weather_seasons": "nature",
    "places": "places", "countries_cities": "places", "places_landmarks": "places", "places_2": "places",
    "people": "people", "people_modern": "people",
    "history": "history", "historical_events": "history",
    "mythology_religion": "myth", "fiction_media": "fiction", "professions": "professions",
    "colours_light": "colours", "everyday_objects": "objects",
}
BY_CONCEPT = {
    # mixed wave categories, split by concept
    "the Great Wall of China": "places", "Egypt's pyramids": "places", "Marie Curie": "people", "Charles Darwin": "people",
    "the oak tree": "nature", "bamboo": "nature", "honey": "food", "the tomato": "food",
    # CR-2 (no category on file), by hand
    "Albert Einstein": "people", "Frida Kahlo": "people", "Napoleon Bonaparte": "people", "William Shakespeare": "people",
    "Wolfgang Amadeus Mozart": "people", "Harry Potter": "fiction", "Eiffel Tower": "places", "Golden Gate Bridge": "places",
    "Mount Everest": "places", "Tokyo": "places", "Venice": "places", "Sahara Desert": "places", "chess": "sport",
    "coffee": "food", "democracy": "abstract", "photosynthesis": "science", "the French Revolution": "history",
    "the Moon": "science", "the Titanic": "history", "the violin": "arts",
}


def family_map():
    fam = {}
    for f in ("concepts_cr3", "concepts_wave2", "concepts_wave3", "concepts_wave4"):
        for cat, cs in json.loads((ROOT / f"prompts/{f}.json").read_text(encoding="utf-8"))["concepts"].items():
            for c in cs:
                if cat in CATEGORY:
                    fam[c] = CATEGORY[cat]
    fam.update(BY_CONCEPT)
    return fam
