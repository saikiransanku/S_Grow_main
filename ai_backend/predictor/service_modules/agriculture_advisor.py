from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List, Optional

PLANT_DISEASE_PREDICTOR = "Plant Disease Predictor"
CROP_RECOMMENDATION_ENGINE = "Crop Recommendation Engine"
FARM_PLANNING_ADVISOR = "Farm Planning / Multi-cropping Advisor"
GENERAL_AGRICULTURE_QA = "General Agriculture Q&A"
CLARIFICATION_NEEDED = "Clarification Needed"

SOIL_KEYWORDS = (
    "black soil",
    "red soil",
    "sandy soil",
    "clay soil",
    "loamy soil",
    "mixed soil",
)
WATER_SOURCE_KEYWORDS = (
    "rainfed",
    "borewell",
    "canal",
    "tank",
    "drip irrigation",
    "sprinkler irrigation",
)
IRRIGATION_LEVEL_KEYWORDS = ("low", "medium", "high")
SEASON_KEYWORDS = ("kharif", "rabi", "summer")

DISEASE_KEYWORDS = (
    "disease",
    "infection",
    "infected",
    "pest",
    "fungus",
    "fungal",
    "spots",
    "spot",
    "curling",
    "yellowing",
    "blight",
    "wilt",
    "rot",
    "mildew",
    "rust",
    "leaf damage",
    "damaged leaf",
    "symptom",
)
CROP_RECOMMENDATION_KEYWORDS = (
    "what crop",
    "what should i grow",
    "which crop",
    "best crop",
    "crop recommendation",
    "recommend crop",
    "what to plant",
    "what should i plant",
    "suits my soil",
    "suitable crop",
    "crop selection",
)
FARM_PLANNING_KEYWORDS = (
    "intercrop",
    "intercropping",
    "multi cropping",
    "multi-cropping",
    "mixed cropping",
    "crop rotation",
    "farm planning",
    "crop combination",
    "double crop",
    "sequential crop",
    "sequential cropping",
)
GENERAL_AGRI_KEYWORDS = (
    "fertilizer",
    "manure",
    "irrigation",
    "rainfall",
    "weather",
    "soil health",
    "seed rate",
    "germination",
    "sowing",
)

CROP_LIBRARY: List[Dict[str, Any]] = [
    {
        "name": "Red gram (pigeonpea)",
        "soil": {"red soil", "black soil", "loamy soil", "mixed soil"},
        "water_need": "low",
        "seasons": {"kharif"},
        "goals": {
            "food crop",
            "cash crop",
            "long-duration crop",
            "multi-cropping",
            "intercropping",
        },
        "risk_profile": "low",
        "budget_fit": {"low", "medium", "high"},
        "telangana_note": "Very common in Telangana rainfed conditions and useful as a stabilizing pulse crop.",
        "limitations": "Returns are slower because it is a longer-duration crop.",
        "fallback": "Green gram (moong)",
    },
    {
        "name": "Groundnut",
        "soil": {"red soil", "sandy soil", "loamy soil", "mixed soil"},
        "water_need": "low",
        "seasons": {"kharif", "rabi"},
        "goals": {"cash crop", "food crop", "short-duration crop", "multi-cropping"},
        "risk_profile": "medium",
        "budget_fit": {"low", "medium"},
        "telangana_note": "Fits many Telangana red-soil belts when water is limited.",
        "limitations": "Waterlogging and heavy clay soils can reduce performance.",
        "fallback": "Sesame",
    },
    {
        "name": "Cotton",
        "soil": {"black soil", "red soil", "loamy soil"},
        "water_need": "medium",
        "seasons": {"kharif"},
        "goals": {"cash crop", "long-duration crop", "intercropping"},
        "risk_profile": "medium",
        "budget_fit": {"medium", "high"},
        "telangana_note": "A major Telangana cash crop, especially where black soil and monsoon support are available.",
        "limitations": "Input cost and pest pressure can become high.",
        "fallback": "Maize",
    },
    {
        "name": "Maize",
        "soil": {"black soil", "red soil", "loamy soil", "mixed soil"},
        "water_need": "medium",
        "seasons": {"kharif", "rabi"},
        "goals": {"food crop", "cash crop", "fodder", "short-duration crop", "multi-cropping"},
        "risk_profile": "medium",
        "budget_fit": {"medium", "high"},
        "telangana_note": "Works well in many Telangana districts with moderate water support.",
        "limitations": "Moisture stress during tasseling can cut yield sharply.",
        "fallback": "Sorghum (jowar)",
    },
    {
        "name": "Paddy (rice)",
        "soil": {"clay soil", "loamy soil", "black soil"},
        "water_need": "high",
        "seasons": {"kharif", "rabi"},
        "goals": {"food crop", "long-duration crop"},
        "risk_profile": "medium",
        "budget_fit": {"medium", "high"},
        "telangana_note": "Suitable in Telangana only where water is dependable.",
        "limitations": "Not a safe choice under low or uncertain irrigation.",
        "fallback": "Maize",
    },
    {
        "name": "Bengal gram (chickpea)",
        "soil": {"black soil", "red soil", "loamy soil"},
        "water_need": "low",
        "seasons": {"rabi"},
        "goals": {"food crop", "cash crop", "short-duration crop"},
        "risk_profile": "low",
        "budget_fit": {"low", "medium"},
        "telangana_note": "A practical Rabi pulse for Telangana where residual moisture is available.",
        "limitations": "Heavy late irrigation can increase disease risk.",
        "fallback": "Black gram (urd)",
    },
    {
        "name": "Green gram (moong)",
        "soil": {"red soil", "sandy soil", "loamy soil", "mixed soil"},
        "water_need": "low",
        "seasons": {"kharif", "rabi", "summer"},
        "goals": {"food crop", "cash crop", "short-duration crop", "multi-cropping", "intercropping"},
        "risk_profile": "low",
        "budget_fit": {"low", "medium"},
        "telangana_note": "Short-duration pulse that helps diversify Telangana cropping plans.",
        "limitations": "Standing water and long wet spells can be harmful.",
        "fallback": "Black gram (urd)",
    },
    {
        "name": "Black gram (urd)",
        "soil": {"black soil", "red soil", "loamy soil", "mixed soil"},
        "water_need": "low",
        "seasons": {"kharif", "rabi"},
        "goals": {"food crop", "cash crop", "short-duration crop", "multi-cropping"},
        "risk_profile": "low",
        "budget_fit": {"low", "medium"},
        "telangana_note": "A safer low-input pulse for Telangana when moisture is moderate.",
        "limitations": "Very heavy rain during flowering can reduce yield.",
        "fallback": "Green gram (moong)",
    },
    {
        "name": "Sorghum (jowar)",
        "soil": {"black soil", "red soil", "sandy soil", "loamy soil"},
        "water_need": "low",
        "seasons": {"kharif", "rabi"},
        "goals": {"food crop", "fodder", "short-duration crop", "multi-cropping"},
        "risk_profile": "low",
        "budget_fit": {"low", "medium"},
        "telangana_note": "Useful in drier Telangana blocks where low-risk crops are preferred.",
        "limitations": "Market value is usually lower than premium cash crops.",
        "fallback": "Pearl millet (bajra)",
    },
    {
        "name": "Pearl millet (bajra)",
        "soil": {"sandy soil", "red soil", "loamy soil"},
        "water_need": "low",
        "seasons": {"kharif"},
        "goals": {"food crop", "fodder", "short-duration crop", "multi-cropping"},
        "risk_profile": "low",
        "budget_fit": {"low"},
        "telangana_note": "A practical drought-tolerant option for low-input Telangana farms.",
        "limitations": "Not ideal if the farmer wants a premium market crop.",
        "fallback": "Sorghum (jowar)",
    },
    {
        "name": "Sesame",
        "soil": {"red soil", "sandy soil", "loamy soil"},
        "water_need": "low",
        "seasons": {"kharif", "summer"},
        "goals": {"cash crop", "short-duration crop", "multi-cropping"},
        "risk_profile": "medium",
        "budget_fit": {"low", "medium"},
        "telangana_note": "A useful low-water cash crop when rainfall is not excessive.",
        "limitations": "Continuous wet weather during flowering can be risky.",
        "fallback": "Groundnut",
    },
    {
        "name": "Sunflower",
        "soil": {"black soil", "red soil", "loamy soil"},
        "water_need": "medium",
        "seasons": {"kharif", "rabi"},
        "goals": {"cash crop", "short-duration crop"},
        "risk_profile": "medium",
        "budget_fit": {"medium"},
        "telangana_note": "Can fit Telangana crop plans where moderate water and oilseed demand exist.",
        "limitations": "Requires better nutrient management than pulses.",
        "fallback": "Sesame",
    },
]


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _current_season() -> str:
    month = datetime.utcnow().month
    if month in {6, 7, 8, 9, 10}:
        return "kharif"
    if month in {11, 12, 1, 2}:
        return "rabi"
    return "summer"


def classify_agriculture_request(
    message: str,
    *,
    has_prediction_context: bool = False,
) -> str:
    text = _normalize_text(message)

    if _contains_any(text, FARM_PLANNING_KEYWORDS):
        return FARM_PLANNING_ADVISOR
    if _contains_any(text, CROP_RECOMMENDATION_KEYWORDS):
        return CROP_RECOMMENDATION_ENGINE
    if _contains_any(text, GENERAL_AGRI_KEYWORDS):
        return GENERAL_AGRICULTURE_QA
    if _contains_any(text, DISEASE_KEYWORDS):
        return PLANT_DISEASE_PREDICTOR
    if has_prediction_context:
        return PLANT_DISEASE_PREDICTOR
    if "crop" in text or "plant" in text or "soil" in text:
        return CROP_RECOMMENDATION_ENGINE
    return CLARIFICATION_NEEDED


def _normalize_profile_context(profile_context: Optional[dict]) -> Dict[str, Any]:
    raw = profile_context if isinstance(profile_context, dict) else {}
    return {
        "full_name": str(raw.get("full_name") or "").strip(),
        "district": str(raw.get("district") or "").strip(),
        "mandal_village": str(raw.get("mandal_village") or "").strip(),
        "soil_type": str(raw.get("soil_type") or "").strip(),
        "water_source": str(raw.get("water_source") or "").strip(),
        "irrigation_level": str(raw.get("irrigation_level") or "").strip(),
        "season": str(raw.get("season") or "").strip(),
        "crop_purpose": str(raw.get("crop_purpose") or "").strip(),
        "land_size": raw.get("land_size"),
        "previous_crop": str(raw.get("previous_crop") or "").strip(),
        "budget": str(raw.get("budget") or "").strip(),
        "market_preference": str(raw.get("market_preference") or "").strip(),
        "risk_preference": str(raw.get("risk_preference") or "").strip(),
        "cropping_preference": str(raw.get("cropping_preference") or "").strip(),
        "location_label": str(raw.get("location_label") or "").strip(),
        "state": str(raw.get("state") or "").strip(),
    }


def _normalize_advisor_context(advisor_context: Optional[dict]) -> Dict[str, Any]:
    raw = advisor_context if isinstance(advisor_context, dict) else {}
    return {
        "active_module": str(raw.get("active_module") or "").strip(),
        "pending_confirmation": str(raw.get("pending_confirmation") or "").strip(),
        "use_profile_land": raw.get("use_profile_land"),
        "last_recommendations": list(raw.get("last_recommendations") or []),
    }


def _profile_has_saved_land(profile: Dict[str, Any]) -> bool:
    land_fields = (
        "district",
        "mandal_village",
        "soil_type",
        "water_source",
        "irrigation_level",
        "land_size",
        "previous_crop",
    )
    return any(profile.get(field) for field in land_fields)


def _detect_same_land_answer(message: str, advisor_state: Dict[str, Any]) -> Optional[bool]:
    text = _normalize_text(message)
    pending = advisor_state.get("pending_confirmation") == "same_land"
    if "same land" in text or "same field" in text or "use my profile" in text:
        return True
    if "different land" in text or "new land" in text or "other land" in text:
        return False
    if pending and text in {"yes", "yes use it", "yes proceed", "same", "same one"}:
        return True
    if pending and text in {"no", "no use new land", "different", "not same"}:
        return False
    return None


def _extract_explicit_inputs(message: str) -> Dict[str, Any]:
    text = _normalize_text(message)
    extracted: Dict[str, Any] = {}

    for soil_type in SOIL_KEYWORDS:
        if soil_type in text:
            extracted["soil_type"] = soil_type.title()
            break

    for water_source in WATER_SOURCE_KEYWORDS:
        if water_source in text:
            extracted["water_source"] = water_source.title()
            break

    for season in SEASON_KEYWORDS:
        if season in text:
            extracted["season"] = season.title()
            break

    if "food crop" in text:
        extracted["crop_purpose"] = "Food crop"
    elif "cash crop" in text:
        extracted["crop_purpose"] = "Cash crop"
    elif "fodder" in text:
        extracted["crop_purpose"] = "Fodder"
    elif "short duration" in text or "short-duration" in text:
        extracted["crop_purpose"] = "Short-duration crop"
    elif "long duration" in text or "long-duration" in text:
        extracted["crop_purpose"] = "Long-duration crop"
    elif "intercrop" in text or "intercropping" in text:
        extracted["crop_purpose"] = "Intercropping"
    elif "multi crop" in text or "multi-cropping" in text or "mixed cropping" in text:
        extracted["crop_purpose"] = "Multi-cropping"

    if "single crop" in text or "monocrop" in text or "mono crop" in text:
        extracted["cropping_preference"] = "Single crop"
    elif "intercrop" in text or "intercropping" in text:
        extracted["cropping_preference"] = "Intercropping"
    elif "multi crop" in text or "multi-cropping" in text or "mixed cropping" in text:
        extracted["cropping_preference"] = "Multi-cropping"

    if "low budget" in text:
        extracted["budget"] = "Low"
    elif "medium budget" in text:
        extracted["budget"] = "Medium"
    elif "high budget" in text:
        extracted["budget"] = "High"

    if "low risk" in text:
        extracted["risk_preference"] = "Low"
    elif "medium risk" in text:
        extracted["risk_preference"] = "Medium"
    elif "high risk" in text:
        extracted["risk_preference"] = "High"

    irrigation_match = re.search(r"\b(low|medium|high)\s+irrigation\b", text)
    if irrigation_match:
        extracted["irrigation_level"] = irrigation_match.group(1).title()

    land_size_match = re.search(r"(\d+(?:\.\d+)?)\s*(acre|acres)", text)
    if land_size_match:
        extracted["land_size"] = float(land_size_match.group(1))

    previous_crop_match = re.search(r"(?:previous crop|last crop|after)\s+(?:was\s+)?([a-z ]+)", text)
    if previous_crop_match:
        extracted["previous_crop"] = previous_crop_match.group(1).strip().title()

    district_match = re.search(r"([a-z ]+?)\s+district", text)
    if district_match:
        extracted["district"] = district_match.group(1).strip().title()

    return extracted


def _merge_inputs(
    *,
    profile: Dict[str, Any],
    extracted: Dict[str, Any],
    use_profile_land: Optional[bool],
) -> Dict[str, Any]:
    inputs: Dict[str, Any] = {}

    if use_profile_land is True:
        inputs.update(profile)

    for key, value in extracted.items():
        if value not in {"", None, "Unknown"}:
            inputs[key] = value

    if not inputs.get("season"):
        if use_profile_land is True and profile.get("season"):
            inputs["season"] = profile["season"]
        else:
            inputs["season"] = _current_season().title()

    if not inputs.get("location_label"):
        inputs["location_label"] = profile.get("location_label") or ""

    if not inputs.get("state"):
        inputs["state"] = profile.get("state") or "Telangana"

    return inputs


def _normalize_level(value: str) -> str:
    text = _normalize_text(value)
    if text in {"low", "rainfed"}:
        return "low"
    if text in {"high", "canal"}:
        return "high"
    if text in {"medium", "borewell", "tank", "drip irrigation", "sprinkler irrigation"}:
        return "medium"
    return "unknown"


def _soil_score(user_soil: str, crop_soils: set[str]) -> tuple[float, str]:
    soil = _normalize_text(user_soil)
    if not soil or soil == "unknown":
        return 1.0, "Soil type is not confirmed, so this is based on broad Telangana suitability."
    if soil in crop_soils:
        return 3.0, f"Matches {soil} well."
    return -1.5, f"{soil.title()} is not the strongest fit for this crop."


def _water_score(inputs: Dict[str, Any], crop_need: str) -> tuple[float, str]:
    water_source = str(inputs.get("water_source") or "")
    irrigation_level = str(inputs.get("irrigation_level") or "")
    water_level = _normalize_level(irrigation_level or water_source)

    if water_level == "unknown":
        return 1.0, "Water status is not fully known, so this is estimated."

    if crop_need == "low":
        if water_level == "low":
            return 3.0, "Fits low-water or rainfed conditions."
        return 2.0, "Water is enough for this relatively low-water crop."

    if crop_need == "medium":
        if water_level == "medium":
            return 3.0, "Water support looks suitable."
        if water_level == "high":
            return 2.0, "Water is available, but avoid over-irrigation."
        return -1.5, "Low water can make this crop risky."

    if water_level == "high":
        return 3.0, "High water availability suits this crop."
    if water_level == "medium":
        return 0.5, "Possible only if irrigation stays reliable."
    return -3.0, "This crop is risky under low water conditions."


def _season_score(user_season: str, crop_seasons: set[str]) -> tuple[float, str]:
    season = _normalize_text(user_season)
    if not season:
        return 1.0, "Season was not given, so a broad seasonal estimate is used."
    if season in crop_seasons:
        return 3.0, f"Good fit for {season.title()} season."
    return -2.0, f"Not the strongest match for {season.title()} season."


def _goal_score(inputs: Dict[str, Any], crop_goals: set[str]) -> tuple[float, str]:
    goal = _normalize_text(inputs.get("crop_purpose") or inputs.get("cropping_preference") or "")
    if not goal or goal == "unknown":
        return 1.0, "Crop goal was not fully provided, so general suitability is used."
    if goal in {_normalize_text(item) for item in crop_goals}:
        return 2.5, "Matches the requested crop goal."
    return 0.0, "Usable, but not the closest match to the requested goal."


def _budget_score(inputs: Dict[str, Any], budget_fit: set[str]) -> tuple[float, str]:
    budget = _normalize_text(inputs.get("budget") or "")
    if not budget or budget == "unknown":
        return 0.5, "Budget level is not fully known."
    if budget in {_normalize_text(item) for item in budget_fit}:
        return 1.5, "Budget fit looks acceptable."
    return -1.0, "Input cost may be higher than the preferred budget."


def _risk_score(inputs: Dict[str, Any], crop_risk: str) -> tuple[float, str]:
    preference = _normalize_text(inputs.get("risk_preference") or "")
    if not preference or preference == "unknown":
        return 0.5, "Risk preference was not fully provided."
    crop_risk_level = _normalize_level(crop_risk)
    wanted = _normalize_level(preference)
    if wanted == crop_risk_level:
        return 1.5, "Risk level matches the stated preference."
    if wanted == "low" and crop_risk_level in {"medium", "high"}:
        return -1.0, "This crop carries more risk than the preferred level."
    return 0.5, "Risk is manageable if inputs are planned well."


def _estimate_rainfall_concern(inputs: Dict[str, Any], crop_need: str) -> str:
    season = _normalize_text(inputs.get("season") or "")
    if season == "kharif":
        if crop_need == "high":
            return "Rainfall support may help in Kharif, but uneven monsoon spells still make water planning important."
        return "Rainfall risk is estimated from Telangana Kharif conditions; uneven monsoon gaps can affect establishment."
    if season == "rabi":
        return "Rainfall is usually less dependable in Rabi, so stored moisture or irrigation matters more."
    if season == "summer":
        return "Summer rainfall support is usually weak, so this needs dependable irrigation."
    return "Rainfall risk is estimated from season and Telangana context because live weather data is not connected."


def _recommendation_label(score: float) -> str:
    if score >= 8.0:
        return "High"
    if score >= 5.5:
        return "Medium"
    return "Moderate"


def _build_crop_reason(crop: Dict[str, Any], notes: List[str]) -> str:
    primary = notes[0] if notes else "Broad Telangana suitability."
    return f"{primary} {crop['telangana_note']}".strip()


def _generate_ranked_recommendations(
    inputs: Dict[str, Any],
    advisor_state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    previous_names = {
        _normalize_text(item) for item in advisor_state.get("last_recommendations", [])
    }

    for crop in CROP_LIBRARY:
        soil_score, soil_note = _soil_score(str(inputs.get("soil_type") or ""), crop["soil"])
        water_score, water_note = _water_score(inputs, crop["water_need"])
        season_score, season_note = _season_score(str(inputs.get("season") or ""), crop["seasons"])
        goal_score, goal_note = _goal_score(inputs, crop["goals"])
        budget_score, budget_note = _budget_score(inputs, crop["budget_fit"])
        risk_score, risk_note = _risk_score(inputs, crop["risk_profile"])

        total = soil_score + water_score + season_score + goal_score + budget_score + risk_score
        if _normalize_text(crop["name"]) in previous_names:
            total -= 0.75

        notes = [soil_note, water_note, season_note, goal_note, budget_note, risk_note]
        ranked.append(
            {
                "name": crop["name"],
                "suitability": _recommendation_label(total),
                "score": round(total, 2),
                "reason": _build_crop_reason(crop, notes),
                "water_need": crop["water_need"].title(),
                "soil_fit": soil_note,
                "seasonal_fit": season_note,
                "rainfall_concern": _estimate_rainfall_concern(inputs, crop["water_need"]),
                "risk": crop["limitations"],
                "fallback": crop["fallback"],
            }
        )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:5]


def _generate_combo_options(inputs: Dict[str, Any], route: str) -> List[Dict[str, str]]:
    season = _normalize_text(inputs.get("season") or "")
    water_level = _normalize_level(str(inputs.get("irrigation_level") or inputs.get("water_source") or ""))
    preference = _normalize_text(inputs.get("cropping_preference") or inputs.get("crop_purpose") or "")

    if route != FARM_PLANNING_ADVISOR and preference not in {"intercropping", "multi-cropping", "multi cropping"}:
        return [
            {
                "name": "Cotton + Red gram",
                "why": "Red gram spreads risk and uses space well in Telangana Kharif fields.",
                "risk": "Not ideal where waterlogging is common.",
                "best_condition": "Kharif, medium water, black or red soil.",
            },
            {
                "name": "Groundnut + Red gram",
                "why": "Useful for red-soil rainfed farms that want oilseed plus pulse diversification.",
                "risk": "Avoid where heavy continuous rain is expected.",
                "best_condition": "Kharif, low to medium water, red or sandy-loam soils.",
            },
        ]

    options: List[Dict[str, str]] = []
    if season == "kharif":
        options.append(
            {
                "name": "Cotton + Red gram",
                "why": "A proven Telangana intercropping pattern for spreading risk between cash crop and pulse.",
                "risk": "Cotton input cost stays higher, so not ideal for very low budgets.",
                "best_condition": "Kharif, medium irrigation, black or red soil.",
            }
        )
        options.append(
            {
                "name": "Maize + Red gram",
                "why": "Gives one cereal plus one pulse and makes better use of medium water conditions.",
                "risk": "Maize becomes risky if irrigation drops at flowering.",
                "best_condition": "Kharif, medium water, loamy or red soils.",
            }
        )
        if water_level in {"low", "unknown"}:
            options.append(
                {
                    "name": "Groundnut + Red gram",
                    "why": "Useful where rainfed or low-water conditions need a safer mixed plan.",
                    "risk": "Continuous wet spells can hurt both crops.",
                    "best_condition": "Kharif, low water, red or sandy-loam soils.",
                }
            )
    elif season == "rabi":
        options.append(
            {
                "name": "Green gram -> Bengal gram",
                "why": "A short crop followed by a pulse can improve land use when Rabi moisture is available.",
                "risk": "Needs timely sowing and at least moderate moisture management.",
                "best_condition": "Rabi planning with short-duration preference.",
            }
        )
        options.append(
            {
                "name": "Maize -> Bengal gram",
                "why": "A sequential option where irrigation is available and the farmer wants crop diversification.",
                "risk": "Not suitable if water is weak or land preparation is delayed.",
                "best_condition": "Rabi with moderate to high irrigation support.",
            }
        )
    else:
        options.append(
            {
                "name": "Fodder maize + Cowpea-style pulse slot",
                "why": "Helps use irrigation efficiently where summer fodder is the main goal.",
                "risk": "Summer heat makes this risky without reliable irrigation.",
                "best_condition": "Summer, assured irrigation, fodder planning.",
            }
        )

    return options[:3]


def _format_user_summary(inputs: Dict[str, Any]) -> str:
    location = (
        inputs.get("location_label")
        or ", ".join(
            part
            for part in [inputs.get("mandal_village"), inputs.get("district"), inputs.get("state")]
            if part
        )
        or "Not provided"
    )
    water_parts = [
        str(inputs.get("water_source") or "").strip(),
        str(inputs.get("irrigation_level") or "").strip(),
    ]
    water_value = ", ".join(part for part in water_parts if part) or "Not provided"

    return (
        "1. User Summary\n"
        f"- Location: {location}\n"
        f"- Soil: {inputs.get('soil_type') or 'Unknown'}\n"
        f"- Water availability: {water_value}\n"
        f"- Season: {inputs.get('season') or 'Unknown'}\n"
        f"- Goal: {inputs.get('crop_purpose') or inputs.get('cropping_preference') or 'Not provided'}"
    )


def _format_recommendation_rank(recommendations: List[Dict[str, Any]]) -> str:
    lines = ["2. Recommendation Rank"]
    for index, item in enumerate(recommendations[:5], start=1):
        lines.extend(
            [
                f"- Crop {index}: {item['name']}",
                f"  - Suitability: {item['suitability']}",
                f"  - Reason: {item['reason']}",
                f"  - Water need: {item['water_need']}",
                f"  - Soil fit: {item['soil_fit']}",
                f"  - Seasonal fit: {item['seasonal_fit']}",
                f"  - Rainfall concern: {item['rainfall_concern']}",
                f"  - Major risk: {item['risk']}",
            ]
        )
    return "\n".join(lines)


def _format_combo_options(combos: List[Dict[str, str]]) -> str:
    lines = ["3. Multi-cropping / Intercropping Options"]
    if not combos:
        lines.append("- Combination 1: No strong combination was selected for the current inputs.")
        return "\n".join(lines)

    for index, combo in enumerate(combos, start=1):
        lines.extend(
            [
                f"- Combination {index}: {combo['name']}",
                f"  - Why it works: {combo['why']}",
                f"  - Risk: {combo['risk']}",
                f"  - Best condition: {combo['best_condition']}",
            ]
        )
    return "\n".join(lines)


def _format_final_advice(inputs: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
    best_choice = recommendations[0]["name"] if recommendations else "Not available"
    backup_choice = recommendations[1]["name"] if len(recommendations) > 1 else best_choice
    main_risk = recommendations[0]["risk"] if recommendations else "Inputs are incomplete."
    missing_data = []
    for key, label in (
        ("district", "district"),
        ("soil_type", "soil type"),
        ("water_source", "water source"),
        ("irrigation_level", "irrigation level"),
        ("previous_crop", "previous crop"),
    ):
        if not inputs.get(key):
            missing_data.append(label)

    extra_data = ", ".join(missing_data) if missing_data else "Live weather or market price data"

    return (
        "4. Final Advice\n"
        f"- Best choice: {best_choice}\n"
        f"- Safer backup choice: {backup_choice}\n"
        f"- When not to choose the main crop: {main_risk}\n"
        f"- What extra data would improve accuracy: {extra_data}"
    )


def _build_general_agriculture_answer(profile: Dict[str, Any]) -> str:
    location = profile.get("location_label") or profile.get("district") or "Telangana"
    return (
        "Route: General Agriculture Q&A\n"
        f"I can help with general farming questions for {location}. "
        "If you want crop recommendation or intercropping advice, please share the crop goal, season, and whether I should use the same land saved in your profile."
    )


def build_agriculture_advisor_response(
    *,
    prompt: str,
    profile_name: str,
    profile_context: Optional[dict],
    advisor_context: Optional[dict],
    conversation_history: Optional[List[dict]] = None,
    has_prediction_context: bool = False,
) -> Dict[str, Any]:
    del profile_name
    del conversation_history

    profile = _normalize_profile_context(profile_context)
    state = _normalize_advisor_context(advisor_context)
    same_land_answer = _detect_same_land_answer(prompt, state)

    route = state["active_module"] or classify_agriculture_request(
        prompt,
        has_prediction_context=has_prediction_context,
    )

    if route == CLARIFICATION_NEEDED:
        return {
            "answer": "Do you want crop recommendation, farm planning, or plant disease prediction?",
            "advisor_context": {
                **state,
                "active_module": CLARIFICATION_NEEDED,
            },
            "route": route,
        }

    if route == GENERAL_AGRICULTURE_QA:
        return {
            "answer": _build_general_agriculture_answer(profile),
            "advisor_context": {
                **state,
                "active_module": GENERAL_AGRICULTURE_QA,
            },
            "route": route,
        }

    if route not in {CROP_RECOMMENDATION_ENGINE, FARM_PLANNING_ADVISOR}:
        return {
            "answer": "",
            "advisor_context": {
                **state,
                "active_module": route,
            },
            "route": route,
        }

    use_profile_land = state.get("use_profile_land")
    if same_land_answer is not None:
        use_profile_land = same_land_answer

    if _profile_has_saved_land(profile) and use_profile_land is None:
        return {
            "answer": "Do you want me to use the same land details saved in your profile for this recommendation?",
            "advisor_context": {
                **state,
                "active_module": route,
                "pending_confirmation": "same_land",
                "use_profile_land": None,
            },
            "route": route,
        }

    extracted = _extract_explicit_inputs(prompt)
    inputs = _merge_inputs(
        profile=profile,
        extracted=extracted,
        use_profile_land=bool(use_profile_land),
    )
    recommendations = _generate_ranked_recommendations(inputs, state)
    combos = _generate_combo_options(inputs, route)

    answer = "\n\n".join(
        [
            _format_user_summary(inputs),
            _format_recommendation_rank(recommendations),
            _format_combo_options(combos),
            _format_final_advice(inputs, recommendations),
        ]
    )

    return {
        "answer": answer,
        "advisor_context": {
            "active_module": route,
            "pending_confirmation": "",
            "use_profile_land": bool(use_profile_land),
            "last_recommendations": [item["name"] for item in recommendations[:3]],
        },
        "route": route,
    }
