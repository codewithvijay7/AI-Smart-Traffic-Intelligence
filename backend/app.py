from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import joblib
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)


GOOGLE_MAPS_API_KEY = os.getenv(
    "GOOGLE_MAPS_API_KEY"
)


# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "ml",
    "models",
    "traffic_model.pkl"
)


# ============================================================
# LOAD ML MODEL
# ============================================================

model = None


def find_prediction_model(
    obj,
    path="root"
):

    if hasattr(
        obj,
        "predict"
    ):

        print(
            f"ML prediction model found at: {path}"
        )

        return obj


    if isinstance(
        obj,
        dict
    ):

        for key, value in obj.items():

            found = find_prediction_model(
                value,
                f"{path}['{key}']"
            )

            if found is not None:

                return found


    elif isinstance(
        obj,
        (list, tuple)
    ):

        for index, value in enumerate(obj):

            found = find_prediction_model(
                value,
                f"{path}[{index}]"
            )

            if found is not None:

                return found


    return None


try:

    loaded_model = joblib.load(
        MODEL_PATH
    )

    print("=" * 70)

    print(
        "Pickle file loaded successfully."
    )

    print(
        "Pickle object type:",
        type(loaded_model).__name__
    )


    model = find_prediction_model(
        loaded_model
    )


    if model is not None:

        print(
            "Traffic ML model loaded successfully"
        )

        print(
            "Actual model type:",
            type(model).__name__
        )

    else:

        print(
            "WARNING: No prediction model found."
        )


    print("=" * 70)


except Exception as e:

    print(
        "WARNING: Could not load traffic model:"
    )

    print(e)

    model = None


# ============================================================
# GOOGLE API STATUS
# ============================================================

if GOOGLE_MAPS_API_KEY:

    print(
        "Google Routes API key: FOUND"
    )

else:

    print(
        "WARNING: GOOGLE_MAPS_API_KEY not found."
    )


# ============================================================
# INDIAN CITY LOCATIONS
# ============================================================

LOCATIONS = {

    # Andhra Pradesh
    "kadapa": [14.4673, 78.8242],
    "cuddapah": [14.4673, 78.8242],
    "tirupati": [13.6288, 79.4192],
    "nellore": [14.4426, 79.9865],
    "vijayawada": [16.5062, 80.6480],
    "visakhapatnam": [17.6868, 83.2185],
    "vizag": [17.6868, 83.2185],
    "guntur": [16.3067, 80.4365],
    "kurnool": [15.8281, 78.0373],
    "anantapur": [14.6819, 77.6006],
    "rajahmundry": [16.9891, 81.2293],
    "ongole": [15.5057, 80.0499],
    "eluru": [16.7107, 81.0952],
    "machilipatnam": [16.1875, 81.1389],

    # Telangana
    "hyderabad": [17.3850, 78.4867],
    "warangal": [17.9784, 79.5941],
    "karimnagar": [18.4386, 79.1288],
    "nizamabad": [18.6725, 78.0941],
    "khammam": [17.2473, 80.1514],

    # Tamil Nadu
    "chennai": [13.0827, 80.2707],
    "coimbatore": [11.0168, 76.9558],
    "madurai": [9.9252, 78.1198],
    "salem": [11.6643, 78.1460],
    "vellore": [12.9165, 79.1325],
    "trichy": [10.7905, 78.7047],
    "tiruchirappalli": [10.7905, 78.7047],

    # Karnataka
    "bangalore": [12.9716, 77.5946],
    "bengaluru": [12.9716, 77.5946],
    "mysore": [12.2958, 76.6394],
    "mysuru": [12.2958, 76.6394],
    "mangalore": [12.9141, 74.8560],
    "hubli": [15.3647, 75.1240],
    "hubballi": [15.3647, 75.1240],

    # Kerala
    "kochi": [9.9312, 76.2673],
    "thiruvananthapuram": [8.5241, 76.9366],
    "trivandrum": [8.5241, 76.9366],
    "kannur": [11.8745, 75.3704],
    "thrissur": [10.5276, 76.2144],

    # Maharashtra
    "mumbai": [19.0760, 72.8777],
    "pune": [18.5204, 73.8567],
    "nagpur": [21.1458, 79.0882],
    "nashik": [19.9975, 73.7898],
    "aurangabad": [19.8762, 75.3433],

    # North India
    "delhi": [28.6139, 77.2090],
    "new delhi": [28.6139, 77.2090],
    "jaipur": [26.9124, 75.7873],
    "lucknow": [26.8467, 80.9462],
    "kanpur": [26.4499, 80.3319],
    "varanasi": [25.3176, 82.9739],
    "agra": [27.1767, 78.0081],
    "chandigarh": [30.7333, 76.7794],

    # East
    "bhubaneswar": [20.2961, 85.8245],
    "cuttack": [20.4625, 85.8830],
    "kolkata": [22.5726, 88.3639],
    "siliguri": [26.7271, 88.3953],

    # Gujarat
    "ahmedabad": [23.0225, 72.5714],
    "surat": [21.1702, 72.8311],
    "vadodara": [22.3072, 73.1812],
    "rajkot": [22.3039, 70.8022],

    # Madhya Pradesh
    "bhopal": [23.2599, 77.4126],
    "indore": [22.7196, 75.8577],

    # Rajasthan
    "udaipur": [24.5854, 73.7125],
    "jodhpur": [26.2389, 73.0243],

    # Bihar
    "patna": [25.5941, 85.1376]
}


# ============================================================
# LOCATION FUNCTIONS
# ============================================================

def normalize_location(name):

    if not name:
        return None

    return " ".join(
        name.strip().lower().split()
    )


def get_coordinates(name):

    location = normalize_location(
        name
    )

    if location not in LOCATIONS:

        return None


    lat, lon = LOCATIONS[
        location
    ]


    return {

        "name":
            name.strip(),

        "latitude":
            lat,

        "longitude":
            lon
    }


# ============================================================
# WEATHER DESCRIPTION
# ============================================================

def weather_description(code):

    try:

        code = int(code)

    except:

        return "Unknown weather"


    descriptions = {

        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",

        45: "Fog",
        48: "Depositing rime fog",

        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",

        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",

        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",

        66: "Light freezing rain",
        67: "Heavy freezing rain",

        71: "Slight snowfall",
        73: "Moderate snowfall",
        75: "Heavy snowfall",

        77: "Snow grains",

        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",

        85: "Slight snow showers",
        86: "Heavy snow showers",

        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }


    return descriptions.get(
        code,
        "Unknown weather"
    )


# ============================================================
# OPEN-METEO WEATHER
# ============================================================

# Small in-memory cache so repeated requests from the same browser
# do not repeatedly hit Open-Meteo. This is especially important on
# Render free instances where several frontend requests can arrive
# together.
WEATHER_CACHE = {}
WEATHER_CACHE_SECONDS = 600


def build_weather_fallback(latitude, longitude, reason="Weather service unavailable"):
    """Always return a complete numeric weather object."""
    return {
        "temperature": 32.0,
        "apparent_temperature": 32.0,
        "humidity": 55.0,
        "precipitation": 0.0,
        "rain": 0.0,
        "showers": 0.0,
        "weather_code": 0,
        "weather_description": "Weather data temporarily unavailable",
        "cloud_cover": 0.0,
        "wind_speed": 0.0,
        "time": None,
        "timezone": "auto",
        "source": "Fallback",
        "live": False,
        "fallback": True,
        "fallback_reason": str(reason),
        "latitude": safe_float(latitude, 0),
        "longitude": safe_float(longitude, 0),
    }


def get_current_weather(
    latitude,
    longitude
):
    latitude = safe_float(latitude, 0)
    longitude = safe_float(longitude, 0)

    cache_key = (round(latitude, 4), round(longitude, 4))
    cached = WEATHER_CACHE.get(cache_key)

    if cached:
        cached_at, cached_weather = cached
        if time.time() - cached_at < WEATHER_CACHE_SECONDS:
            print(f"Weather cache hit for {cache_key}")
            return dict(cached_weather)

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )


    params = {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "current":
            ",".join([

                "temperature_2m",

                "relative_humidity_2m",

                "apparent_temperature",

                "precipitation",

                "rain",

                "showers",

                "weather_code",

                "cloud_cover",

                "wind_speed_10m"
            ]),

        "timezone":
            "auto"
    }


    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        if response.status_code == 429:
            print(
                f"Open-Meteo returned HTTP 429 for {cache_key}. Using fallback weather."
            )
            return build_weather_fallback(
                latitude,
                longitude,
                "Open-Meteo rate limit (HTTP 429)"
            )

        response.raise_for_status()
        data = response.json()

    except Exception as weather_error:
        print(
            "Open-Meteo request failed. Using fallback weather:",
            weather_error
        )
        return build_weather_fallback(
            latitude,
            longitude,
            weather_error
        )


    current = data.get(
        "current",
        {}
    )


    if not current:

        raise Exception(
            "Weather API returned no current data."
        )


    weather_code = current.get(
        "weather_code",
        0
    )


    weather_result = {

        "temperature":
            round(
                safe_float(
                    current.get(
                        "temperature_2m"
                    ),
                    32
                ),
                1
            ),

        "apparent_temperature":
            round(
                safe_float(
                    current.get(
                        "apparent_temperature"
                    ),
                    32
                ),
                1
            ),

        "humidity":
            round(
                safe_float(
                    current.get(
                        "relative_humidity_2m"
                    ),
                    50
                ),
                1
            ),

        "precipitation":
            round(
                safe_float(
                    current.get(
                        "precipitation"
                    ),
                    0
                ),
                2
            ),

        "rain":
            round(
                safe_float(
                    current.get(
                        "rain"
                    ),
                    0
                ),
                2
            ),

        "showers":
            round(
                safe_float(
                    current.get(
                        "showers"
                    ),
                    0
                ),
                2
            ),

        "weather_code":
            weather_code,

        "weather_description":
            weather_description(
                weather_code
            ),

        "cloud_cover":
            round(
                safe_float(
                    current.get(
                        "cloud_cover"
                    ),
                    0
                ),
                1
            ),

        "wind_speed":
            round(
                safe_float(
                    current.get(
                        "wind_speed_10m"
                    ),
                    0
                ),
                1
            ),

        "time":
            current.get(
                "time"
            ),

        "timezone":
            data.get(
                "timezone"
            ),

        "source":
            "Open-Meteo",

        "live":
            True
    }

    WEATHER_CACHE[cache_key] = (time.time(), dict(weather_result))
    return weather_result


# ============================================================
# TRAFFIC LEVEL FROM GOOGLE
# ============================================================

def google_speed_to_level(
    speed
):

    speed = str(
        speed
    ).upper()


    if speed == "TRAFFIC_JAM":

        return "Severe"


    if speed == "SLOW":

        return "High"


    if speed == "NORMAL":

        return "Low"


    return "Moderate"


# ============================================================
# TRAFFIC LEVEL SCORE
# ============================================================

TRAFFIC_SCORE = {

    "Low": 25,

    "Moderate": 50,

    "High": 75,

    "Severe": 95
}


# ============================================================
# GOOGLE TRAFFIC ROUTING
# ============================================================

def decode_polyline(
    encoded
):

    """
    Decode Google's encoded polyline.

    Returns:
        [
            [latitude, longitude],
            ...
        ]
    """

    if not encoded:

        return []


    coordinates = []

    index = 0

    lat = 0

    lng = 0


    while index < len(encoded):

        result = 0

        shift = 0


        while True:

            byte = (
                ord(
                    encoded[index]
                )
                - 63
            )

            index += 1

            result |= (
                (byte & 0x1F)
                << shift
            )

            shift += 5


            if byte < 0x20:

                break


        if result & 1:

            lat_change = ~(
                result >> 1
            )

        else:

            lat_change = (
                result >> 1
            )


        lat += lat_change


        result = 0

        shift = 0


        while True:

            byte = (
                ord(
                    encoded[index]
                )
                - 63
            )

            index += 1

            result |= (
                (byte & 0x1F)
                << shift
            )

            shift += 5


            if byte < 0x20:

                break


        if result & 1:

            lng_change = ~(
                result >> 1
            )

        else:

            lng_change = (
                result >> 1
            )


        lng += lng_change


        coordinates.append([

            lat / 100000.0,

            lng / 100000.0
        ])


    return coordinates


def get_google_routes(
    source_lat,
    source_lon,
    destination_lat,
    destination_lon
):

    if not GOOGLE_MAPS_API_KEY:

        raise Exception(
            "GOOGLE_MAPS_API_KEY is missing."
        )


    url = (
        "https://routes.googleapis.com/"
        "directions/v2:computeRoutes"
    )


    headers = {

        "Content-Type":
            "application/json",

        "X-Goog-Api-Key":
            GOOGLE_MAPS_API_KEY,

        "X-Goog-FieldMask":
            ",".join([

                "routes.distanceMeters",

                "routes.duration",

                "routes.staticDuration",

                "routes.polyline.encodedPolyline",

                "routes.description",

                "routes.travelAdvisory.speedReadingIntervals",

                "routes.routeLabels",

                "fallbackInfo"
            ])
    }


    body = {

        "origin": {

            "location": {

                "latLng": {

                    "latitude":
                        source_lat,

                    "longitude":
                        source_lon
                }
            }
        },


        "destination": {

            "location": {

                "latLng": {

                    "latitude":
                        destination_lat,

                    "longitude":
                        destination_lon
                }
            }
        },


        "travelMode":
            "DRIVE",


        "routingPreference":
            "TRAFFIC_AWARE",


        "computeAlternativeRoutes":
            True,


        "extraComputations": [

            "TRAFFIC_ON_POLYLINE"

        ],


        "routeModifiers": {

            "avoidTolls":
                False,

            "avoidHighways":
                False,

            "avoidFerries":
                False
        },


        "languageCode":
            "en-US",


        "units":
            "METRIC"
    }


    response = requests.post(

        url,

        headers=headers,

        json=body,

        timeout=30
    )


    if response.status_code != 200:

        try:

            error_data = response.json()

        except:

            error_data = response.text


        raise Exception(
            f"Google Routes API error "
            f"{response.status_code}: "
            f"{error_data}"
        )


    data = response.json()


    routes = data.get(
        "routes",
        []
    )


    if not routes:

        raise Exception(
            "Google Routes API returned no routes."
        )


    return routes


# ============================================================
# ANALYZE GOOGLE TRAFFIC
# ============================================================

def analyze_google_traffic(
    route
):

    advisory = route.get(
        "travelAdvisory",
        {}
    )


    intervals = advisory.get(
        "speedReadingIntervals",
        []
    )


    counts = {

        "NORMAL": 0,

        "SLOW": 0,

        "TRAFFIC_JAM": 0
    }


    total_segments = 0


    normalized_intervals = []


    for interval in intervals:

        speed = interval.get(
            "speed"
        )


        if not speed:

            continue


        speed = str(
            speed
        ).upper()


        if speed not in counts:

            continue


        start_index = interval.get(
            "startPolylinePointIndex",
            0
        )


        end_index = interval.get(
            "endPolylinePointIndex",
            start_index
        )


        segment_count = max(
            1,
            end_index - start_index
        )


        counts[
            speed
        ] += segment_count


        total_segments += (
            segment_count
        )


        normalized_intervals.append({

            "start_index":
                start_index,

            "end_index":
                end_index,

            "speed":
                speed,

            "traffic_level":
                google_speed_to_level(
                    speed
                )
        })


    if total_segments == 0:

        traffic_level = "Moderate"

        traffic_confidence = 0


    else:

        severe = counts[
            "TRAFFIC_JAM"
        ]

        high = counts[
            "SLOW"
        ]

        normal = counts[
            "NORMAL"
        ]


        if severe > 0:

            traffic_level = "Severe"

        elif (
            high / total_segments
            >= 0.25
        ):

            traffic_level = "High"

        elif (
            high > 0
        ):

            traffic_level = "Moderate"

        else:

            traffic_level = "Low"


        traffic_confidence = round(

            (
                max(
                    normal,
                    high,
                    severe
                )
                /
                total_segments
            )
            * 100,

            2
        )


    traffic_score = TRAFFIC_SCORE.get(

        traffic_level,

        50
    )


    # --------------------------------------------------------
    # Calculate approximate traffic percentages
    # --------------------------------------------------------

    if total_segments > 0:

        normal_percentage = round(

            counts["NORMAL"]
            /
            total_segments
            *
            100,

            2
        )


        slow_percentage = round(

            counts["SLOW"]
            /
            total_segments
            *
            100,

            2
        )


        jam_percentage = round(

            counts["TRAFFIC_JAM"]
            /
            total_segments
            *
            100,

            2
        )

    else:

        normal_percentage = 0

        slow_percentage = 0

        jam_percentage = 0


    return {

        "congestion":
            traffic_level,

        "traffic_score":
            traffic_score,

        "google_confidence":
            traffic_confidence,

        "normal_percentage":
            normal_percentage,

        "slow_percentage":
            slow_percentage,

        "traffic_jam_percentage":
            jam_percentage,

        "segments_analyzed":
            total_segments,

        "speed_intervals":
            normalized_intervals,

        "data_source":
            "Google Routes API",

        "live":
            True
    }


# ============================================================
# CONVERT GOOGLE TRAFFIC TO ML FEATURES
# ============================================================

def create_ml_features_from_google(
    traffic_analysis,
    distance_km,
    weather
):

    congestion = traffic_analysis[
        "congestion"
    ]


    # --------------------------------------------------------
    # Traffic proxy values
    #
    # IMPORTANT:
    # These are derived proxy features because Google Routes
    # does not directly provide vehicle_count or occupancy.
    # --------------------------------------------------------

    if congestion == "Low":

        average_speed = 45

        occupancy = 30

        vehicle_count = 180


    elif congestion == "Moderate":

        average_speed = 32

        occupancy = 55

        vehicle_count = 300


    elif congestion == "High":

        average_speed = 22

        occupancy = 75

        vehicle_count = 500


    else:

        average_speed = 12

        occupancy = 92

        vehicle_count = 700


    # Weather impact

    rainfall = safe_float(
        weather.get("precipitation"),
        0
    )


    temperature = safe_float(
        weather.get("temperature"),
        32
    )


    if rainfall > 5:

        average_speed -= min(
            rainfall * 0.25,
            6
        )

        occupancy += min(
            rainfall * 0.4,
            8
        )


    average_speed = max(
        8,
        average_speed
    )


    occupancy = min(
        98,
        occupancy
    )


    return {

        "vehicle_count":
            vehicle_count,

        "average_speed":
            round(
                average_speed,
                1
            ),

        "road_occupancy":
            round(
                occupancy,
                1
            ),

        "rainfall":
            rainfall,

        "temperature":
            temperature
    }


# ============================================================
# ML TRAFFIC PREDICTION
# ============================================================

def predict_traffic(

    vehicle_count=350,

    average_speed=25,

    road_occupancy=80,

    rainfall=15,

    temperature=32,

    hour=18,

    day_of_week=2
):

    traffic_data = {

        "vehicle_count":
            safe_float(vehicle_count, 350),

        "average_speed":
            safe_float(average_speed, 25),

        "road_occupancy":
            safe_float(road_occupancy, 80),

        "rainfall":
            safe_float(rainfall, 0),

        "temperature":
            safe_float(temperature, 32),

        "hour":
            safe_int(hour, 18),

        "day_of_week":
            safe_int(day_of_week, 2)
    }


    if model is None:

        return {

            "prediction":
                "Moderate",

            "confidence":
                0,

            "probabilities":
                {},

            "traffic_data":
                traffic_data,

            "error":
                "ML model is not loaded."
        }


    try:

        # Use a DataFrame with the same feature names used during
        # training. This also prevents sklearn feature-name warnings.
        features = pd.DataFrame([[

            traffic_data["vehicle_count"],
            traffic_data["average_speed"],
            traffic_data["road_occupancy"],
            traffic_data["rainfall"],
            traffic_data["temperature"],
            traffic_data["hour"],
            traffic_data["day_of_week"]

        ]], columns=[

            "vehicle_count",
            "average_speed",
            "road_occupancy",
            "rainfall",
            "temperature",
            "hour",
            "day_of_week"
        ])


        prediction = model.predict(
            features
        )[0]


        prediction = str(
            prediction
        )


        probabilities = {}

        confidence = 0


        if hasattr(
            model,
            "predict_proba"
        ):

            probability_values = (
                model.predict_proba(
                    features
                )[0]
            )


            classes = getattr(
                model,
                "classes_",
                []
            )


            for class_name, probability in zip(

                classes,

                probability_values

            ):

                probabilities[
                    str(class_name)
                ] = round(

                    float(
                        probability
                    )
                    * 100,

                    2
                )


            confidence = round(

                max(
                    probability_values
                )
                * 100,

                2
            )


        return {

            "prediction":
                prediction,

            "confidence":
                confidence,

            "probabilities":
                probabilities,

            "traffic_data":
                traffic_data
        }


    except Exception as e:

        print(
            "Prediction error:",
            e
        )


        return {

            "prediction":
                "Moderate",

            "confidence":
                0,

            "probabilities":
                {},

            "traffic_data":
                traffic_data,

            "error":
                str(e)
        }


# ============================================================
# ROUTE TRAFFIC COMBINATION
# ============================================================

def generate_route_traffic(

    google_traffic,

    distance_km,

    duration_minutes,

    weather
):

    now = datetime.now()


    hour = now.hour

    day_of_week = now.weekday()


    ml_features = create_ml_features_from_google(

        google_traffic,

        distance_km,

        weather
    )


    ml_prediction = predict_traffic(

        vehicle_count=
            ml_features[
                "vehicle_count"
            ],

        average_speed=
            ml_features[
                "average_speed"
            ],

        road_occupancy=
            ml_features[
                "road_occupancy"
            ],

        rainfall=
            ml_features[
                "rainfall"
            ],

        temperature=
            ml_features[
                "temperature"
            ],

        hour=
            hour,

        day_of_week=
            day_of_week
    )


    # --------------------------------------------------------
    # Google is the primary real-time traffic source.
    # Random Forest provides the AI prediction layer.
    # --------------------------------------------------------

    google_level = google_traffic[
        "congestion"
    ]


    final_level = google_level


    return {

        "congestion":
            final_level,

        "traffic_score":
            google_traffic[
                "traffic_score"
            ],

        "confidence":
            google_traffic[
                "google_confidence"
            ],

        "google_traffic":
            google_traffic,

        "ml_prediction":
            ml_prediction[
                "prediction"
            ],

        "ml_confidence":
            ml_prediction[
                "confidence"
            ],

        "probabilities":
            ml_prediction[
                "probabilities"
            ],

        "traffic_data":
            ml_prediction[
                "traffic_data"
            ],

        "weather":
            weather,

        "data_source":
            "Google Routes API + Random Forest",

        "live":
            True,

        "model_used":
            model is not None
    }


# ============================================================
# ADAPTIVE ROUTE SCORE
# ============================================================

def calculate_adaptive_scores(
    routes
):

    if not routes:
        return []

    distances = [
        safe_float(route.get("distance_km"), 0)
        for route in routes
    ]

    durations = [
        safe_float(route.get("duration_minutes"), 0)
        for route in routes
    ]

    min_distance = min(distances)
    max_distance = max(distances)

    min_duration = min(durations)
    max_duration = max(durations)

    # Lower traffic risk = better.
    traffic_risk_map = {
        "Low": 0.00,
        "Moderate": 0.33,
        "High": 0.66,
        "Severe": 1.00
    }

    for route in routes:

        # --------------------------------------------------------
        # Distance score
        # Shorter distance = higher score
        # --------------------------------------------------------

        if max_distance == min_distance:
            distance_score = 1.0
        else:
            distance_norm = (
                route["distance_km"] - min_distance
            ) / (
                max_distance - min_distance
            )

            distance_score = 1.0 - distance_norm

        # --------------------------------------------------------
        # Travel-time score
        # Shorter time = higher score
        # --------------------------------------------------------

        if max_duration == min_duration:
            duration_score = 1.0
        else:
            duration_norm = (
                route["duration_minutes"] - min_duration
            ) / (
                max_duration - min_duration
            )

            duration_score = 1.0 - duration_norm

        # --------------------------------------------------------
        # Live traffic score
        # Lower congestion = higher score
        # --------------------------------------------------------

        congestion = (
            route.get("traffic", {})
            .get("congestion", "Moderate")
        )

        traffic_risk = traffic_risk_map.get(
            congestion,
            0.33
        )

        traffic_score = 1.0 - traffic_risk

        # --------------------------------------------------------
        # Weighted adaptive score
        #
        # Distance   = 40%
        # Time       = 30%
        # Traffic    = 30%
        #
        # Higher score = better route
        # --------------------------------------------------------

        base_score = (
            (0.40 * distance_score)
            + (0.30 * duration_score)
            + (0.30 * traffic_score)
        )

        # Keep two decimals internally so the frontend can show
        # meaningful differences instead of converting both routes
        # to the same whole-number score.
        route["adaptive_score"] = round(
            base_score * 100,
            2
        )

        route["score_breakdown"] = {
            "distance_weight": "40%",
            "time_weight": "30%",
            "traffic_weight": "30%",

            "distance_score": round(
                distance_score * 100,
                2
            ),

            "time_score": round(
                duration_score * 100,
                2
            ),

            "traffic_score": round(
                traffic_score * 100,
                2
            ),

            "base_score": round(
                base_score * 100,
                2
            )
        }

    # ------------------------------------------------------------
    # Rank routes by adaptive score.
    #
    # Higher adaptive score is always the primary criterion.
    # Only an exact score tie uses:
    #   1. lower live traffic
    #   2. faster travel time
    #   3. shorter distance
    #
    # This ensures that a route with 60.1 beats a route with 60.0.
    # ------------------------------------------------------------

    def traffic_priority(route):
        congestion = (
            route.get("traffic", {})
            .get("congestion", "Moderate")
        )

        priority = {
            "Low": 0,
            "Moderate": 1,
            "High": 2,
            "Severe": 3
        }

        return priority.get(congestion, 1)

    routes.sort(
        key=lambda route: (
            -float(route.get("adaptive_score", 0)),
            traffic_priority(route),
            float(route.get("duration_minutes", float("inf"))),
            float(route.get("distance_km", float("inf")))
        )
    )

    # ------------------------------------------------------------
    # Assign ranks and recommendation.
    # ------------------------------------------------------------

    for index, route in enumerate(routes):

        route["rank"] = index + 1

        route["recommended"] = (
            index == 0
        )

    # ------------------------------------------------------------
    # Explain why the first route was recommended.
    # ------------------------------------------------------------

    if routes:

        recommended = routes[0]

        traffic = (
            recommended.get("traffic", {})
            .get("congestion", "Moderate")
        )

        score = float(
            recommended.get(
                "adaptive_score",
                0
            )
        )

        second_score = (
            float(
                routes[1].get(
                    "adaptive_score",
                    0
                )
            )
            if len(routes) > 1
            else None
        )

        if second_score is not None and score == second_score:
            if traffic == "Low":
                reason = (
                    "Recommended because the adaptive scores are equal, "
                    "so lower live traffic is used as the tie-breaker."
                )
            elif traffic == "Moderate":
                reason = (
                    "Recommended because the adaptive scores are equal, "
                    "so the route provides the better overall balance."
                )
            else:
                reason = (
                    "Recommended using the adaptive score and "
                    "deterministic route tie-break evaluation."
                )

        elif second_score is not None and score > second_score:
            reason = (
                f"Recommended because it has the highest adaptive score "
                f"({score:.1f}) among the available routes."
            )

        elif traffic == "Low":
            reason = (
                "Recommended because of low live traffic and "
                "strong overall route performance."
            )
        elif traffic == "Moderate":
            reason = (
                "Recommended based on the best overall balance of "
                "distance, travel time and traffic."
            )
        elif traffic == "High":
            reason = (
                "Recommended because it provides the best "
                "overall adaptive score."
            )
        else:
            reason = (
                "Recommended based on the overall adaptive "
                "route evaluation."
            )

        recommended["recommendation_reason"] = reason

    return routes


# ============================================================
# OSRM FALLBACK
# ============================================================

def get_osrm_routes(

    source_lat,

    source_lon,

    destination_lat,

    destination_lon
):

    url = (

        "https://router.project-osrm.org/"
        "route/v1/driving/"

        f"{source_lon},{source_lat};"
        f"{destination_lon},{destination_lat}"
    )


    params = {

        "alternatives":
            "true",

        "steps":
            "false",

        "overview":
            "full",

        "geometries":
            "geojson"
    }


    response = requests.get(

        url,

        params=params,

        timeout=30
    )


    response.raise_for_status()


    data = response.json()


    if data.get(
        "code"
    ) != "Ok":

        raise Exception(

            data.get(
                "message",
                "No route found."
            )
        )


    return data.get(
        "routes",
        []
    )


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({

        "message":
            "AI Smart Traffic Intelligence API is running",

        "status":
            "success",

        "model_loaded":
            model is not None,

        "model_type":
            type(model).__name__
            if model is not None
            else None,

        "google_routes_api":
            bool(
                GOOGLE_MAPS_API_KEY
            ),

        "traffic_mode":
            "Google Traffic Aware"
            if GOOGLE_MAPS_API_KEY
            else "OSRM fallback"
    })


# ============================================================
# LOCATIONS
# ============================================================

@app.route(
    "/api/locations",
    methods=["GET"]
)
def locations():

    cities = sorted(
        LOCATIONS.keys()
    )


    return jsonify({

        "status":
            "success",

        "count":
            len(cities),

        "locations":
            cities
    })


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if value is None:
            return float(default)

        if isinstance(value, str) and not value.strip():
            return float(default)

        return float(value)

    except (TypeError, ValueError):

        return float(default)


def safe_int(value, default=0):

    try:

        if value is None:
            return int(default)

        if isinstance(value, str) and not value.strip():
            return int(default)

        return int(value)

    except (TypeError, ValueError):

        return int(default)


# ============================================================
# WEATHER
# ============================================================

@app.route(
    "/api/weather",
    methods=["POST"]
)
def weather_api():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )


        location = data.get(
            "location"
        )


        latitude = data.get(
            "latitude"
        )


        longitude = data.get(
            "longitude"
        )


        if location:

            location_data = get_coordinates(
                location
            )


            if location_data is None:

                return jsonify({

                    "status":
                        "error",

                    "message":
                        f"Location '{location}' was not found."

                }), 400


            latitude = location_data[
                "latitude"
            ]

            longitude = location_data[
                "longitude"
            ]


        if latitude is None or longitude is None:

            return jsonify({

                "status":
                    "error",

                "message":
                    "Provide location or latitude and longitude."

            }), 400


        weather = get_current_weather(

            latitude,

            longitude
        )


        return jsonify({

            "status":
                "success",

            "location":
                location,

            "latitude":
                latitude,

            "longitude":
                longitude,

            "weather":
                weather

        })


    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict_api():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )


        result = predict_traffic(

            vehicle_count=
                data.get(
                    "vehicle_count",
                    350
                ),

            average_speed=
                data.get(
                    "average_speed",
                    25
                ),

            road_occupancy=
                data.get(
                    "road_occupancy",
                    80
                ),

            rainfall=
                data.get(
                    "rainfall",
                    15
                ),

            temperature=
                data.get(
                    "temperature",
                    32
                ),

            hour=
                data.get(
                    "hour",
                    18
                ),

            day_of_week=
                data.get(
                    "day_of_week",
                    2
                )
        )


        result[
            "congestion"
        ] = result[
            "prediction"
        ]


        result[
            "traffic_score"
        ] = TRAFFIC_SCORE.get(

            result[
                "prediction"
            ],

            50
        )


        return jsonify({

            "status":
                "success",

            **result

        })


    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# ROUTE
# ============================================================

@app.route(
    "/api/route",
    methods=["POST"]
)
def route_api():

    try:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )


        source = (
            data.get(
                "source",
                ""
            )
            .strip()
        )


        destination = (
            data.get(
                "destination",
                ""
            )
            .strip()
        )


        if not source:

            return jsonify({

                "status":
                    "error",

                "message":
                    "Source is required."

            }), 400


        if not destination:

            return jsonify({

                "status":
                    "error",

                "message":
                    "Destination is required."

            }), 400


        # ====================================================
        # SOURCE
        # ====================================================

        source_location = get_coordinates(
            source
        )


        if source_location is None:

            return jsonify({

                "status":
                    "error",

                "message":
                    f"Location '{source}' was not found."

            }), 400


        # ====================================================
        # DESTINATION
        # ====================================================

        destination_location = get_coordinates(
            destination
        )


        if destination_location is None:

            return jsonify({

                "status":
                    "error",

                "message":
                    f"Location '{destination}' was not found."

            }), 400


        source_lat = source_location[
            "latitude"
        ]

        source_lon = source_location[
            "longitude"
        ]


        destination_lat = destination_location[
            "latitude"
        ]

        destination_lon = destination_location[
            "longitude"
        ]


        # ====================================================
        # WEATHER
        # ====================================================

        try:

            source_weather = get_current_weather(

                source_lat,

                source_lon
            )


            destination_weather = get_current_weather(

                destination_lat,

                destination_lon
            )


            route_temperature = round(

                (
                    safe_float(
                        source_weather.get("temperature"),
                        32
                    )

                    +

                    safe_float(
                        destination_weather.get("temperature"),
                        32
                    )
                ) / 2,

                1
            )


            route_precipitation = round(

                (
                    safe_float(
                        source_weather.get("precipitation"),
                        0
                    )

                    +

                    safe_float(
                        destination_weather.get("precipitation"),
                        0
                    )
                ) / 2,

                2
            )


            route_weather = {

                "temperature":
                    route_temperature,

                "precipitation":
                    route_precipitation,

                "source":
                    source_weather,

                "destination":
                    destination_weather,

                "data_source":
                    "Open-Meteo",

                "live":
                    True
            }


        except Exception as weather_error:

            print(
                "Weather API error:",
                weather_error
            )


            route_weather = {

                "temperature":
                    32.0,

                "precipitation":
                    0.0,

                "source":
                    build_weather_fallback(
                        source_lat,
                        source_lon,
                        weather_error
                    ),

                "destination":
                    build_weather_fallback(
                        destination_lat,
                        destination_lon,
                        weather_error
                    ),

                "data_source":
                    "Fallback",

                "live":
                    False,

                "error":
                    str(
                        weather_error
                    )
            }


        # Ensure weather values used by ML/routing are always numeric.
        route_weather["temperature"] = safe_float(
            route_weather.get("temperature"),
            32
        )

        route_weather["precipitation"] = safe_float(
            route_weather.get("precipitation"),
            0
        )


        # ====================================================
        # GOOGLE ROUTES
        # ====================================================

        routes = []


        google_used = False

        google_error = None


        try:

            google_routes = get_google_routes(

                source_lat,

                source_lon,

                destination_lat,

                destination_lon
            )


            google_used = True


            print(
                f"Google returned {len(google_routes)} route(s)."
            )


            for index, google_route in enumerate(
                google_routes
            ):

                distance_km = round(

                    safe_float(
                        google_route.get("distanceMeters"),
                        0
                    )
                    / 1000,

                    2
                )


                duration_string = (
                    google_route.get(
                        "duration",
                        "0s"
                    )
                )


                static_duration_string = (
                    google_route.get(
                        "staticDuration",
                        "0s"
                    )
                )


                def seconds_from_duration(
                    value
                ):

                    value = str(
                        value
                    )

                    if value.endswith("s"):

                        value = value[:-1]

                    try:

                        return float(
                            value
                        )

                    except:

                        return 0


                duration_seconds = (
                    seconds_from_duration(
                        duration_string
                    )
                )


                static_duration_seconds = (
                    seconds_from_duration(
                        static_duration_string
                    )
                )


                duration_minutes = round(

                    duration_seconds
                    / 60,

                    1
                )


                static_duration_minutes = round(

                    static_duration_seconds
                    / 60,

                    1
                )


                encoded_polyline = (

                    google_route
                    .get(
                        "polyline",
                        {}
                    )
                    .get(
                        "encodedPolyline",
                        ""
                    )
                )


                coordinates = decode_polyline(

                    encoded_polyline
                )


                google_traffic = (
                    analyze_google_traffic(
                        google_route
                    )
                )


                route_traffic = (
                    generate_route_traffic(

                        google_traffic,

                        distance_km,

                        duration_minutes,

                        route_weather
                    )
                )


                routes.append({

                    "route_id":
                        f"route_{index + 1}",

                    "route_name":
                        f"Route {index + 1}",

                    "description":
                        google_route.get(
                            "description",
                            ""
                        ),

                    "distance_km":
                        distance_km,

                    "duration_minutes":
                        duration_minutes,

                    "static_duration_minutes":
                        static_duration_minutes,

                    "traffic_delay_minutes":
                        round(

                            max(

                                0,

                                duration_minutes
                                -
                                static_duration_minutes
                            ),

                            1
                        ),

                    "coordinates":
                        coordinates,

                    "traffic":
                        route_traffic,

                    "routing_provider":
                        "Google Routes API",

                    "live_traffic":
                        True,

                    "adaptive_score":
                        None,

                    "recommended":
                        False,

                    "rank":
                        None
                })


        except Exception as google_error_exception:

            google_error = str(
                google_error_exception
            )


            print(
                "Google Routes API failed:"
            )

            print(
                google_error
            )


        # ====================================================
        # OSRM FALLBACK
        # ====================================================

        if not routes:

            print(
                "Using OSRM fallback."
            )


            osrm_routes = get_osrm_routes(

                source_lat,

                source_lon,

                destination_lat,

                destination_lon
            )


            for index, osrm_route in enumerate(
                osrm_routes
            ):

                distance_km = round(

                    safe_float(
                        osrm_route.get("distance"),
                        0
                    )
                    / 1000,

                    2
                )


                duration_minutes = round(

                    safe_float(
                        osrm_route.get("duration"),
                        0
                    )
                    / 60,

                    1
                )


                geometry = osrm_route.get(
                    "geometry",
                    {}
                )


                coordinates = geometry.get(
                    "coordinates",
                    []
                )


                leaflet_coordinates = [

                    [
                        coordinate[1],
                        coordinate[0]
                    ]

                    for coordinate in coordinates

                    if len(coordinate) >= 2
                ]


                # Fallback traffic estimate

                fallback_level = (
                    "Moderate"
                )


                fallback_google_traffic = {

                    "congestion":
                        fallback_level,

                    "traffic_score":
                        50,

                    "google_confidence":
                        0,

                    "normal_percentage":
                        0,

                    "slow_percentage":
                        0,

                    "traffic_jam_percentage":
                        0,

                    "segments_analyzed":
                        0,

                    "speed_intervals":
                        [],

                    "data_source":
                        "OSRM fallback",

                    "live":
                        False
                }


                route_traffic = (
                    generate_route_traffic(

                        fallback_google_traffic,

                        distance_km,

                        duration_minutes,

                        route_weather
                    )
                )


                routes.append({

                    "route_id":
                        f"route_{index + 1}",

                    "route_name":
                        f"Route {index + 1}",

                    "description":
                        "OSRM fallback route",

                    "distance_km":
                        distance_km,

                    "duration_minutes":
                        duration_minutes,

                    "static_duration_minutes":
                        duration_minutes,

                    "traffic_delay_minutes":
                        0,

                    "coordinates":
                        leaflet_coordinates,

                    "traffic":
                        route_traffic,

                    "routing_provider":
                        "OSRM fallback",

                    "live_traffic":
                        False,

                    "adaptive_score":
                        None,

                    "recommended":
                        False,

                    "rank":
                        None
                })


        # ====================================================
        # ADAPTIVE ROUTING
        # ====================================================

        routes = calculate_adaptive_scores(
            routes
        )


        recommended_route = routes[0]


        # ====================================================
        # RESPONSE
        # ====================================================

        return jsonify({

            "status":
                "success",


            "source": {

                "name":
                    source,

                "latitude":
                    source_lat,

                "longitude":
                    source_lon
            },


            "destination": {

                "name":
                    destination,

                "latitude":
                    destination_lat,

                "longitude":
                    destination_lon
            },


            "distance_km":
                recommended_route[
                    "distance_km"
                ],


            "duration_minutes":
                recommended_route[
                    "duration_minutes"
                ],


            "route":
                recommended_route[
                    "coordinates"
                ],


            "traffic":
                recommended_route[
                    "traffic"
                ],


            "weather":
                route_weather,


            "recommended_route": {

                "route_id":
                    recommended_route[
                        "route_id"
                    ],

                "route_name":
                    recommended_route[
                        "route_name"
                    ],

                "distance_km":
                    recommended_route[
                        "distance_km"
                    ],

                "duration_minutes":
                    recommended_route[
                        "duration_minutes"
                    ],

                "adaptive_score":
                    recommended_route[
                        "adaptive_score"
                    ],

                "traffic":
                    recommended_route[
                        "traffic"
                    ],

                "recommended":
                    True,

                "rank":
                    recommended_route[
                        "rank"
                    ],

                "recommendation_reason":
                    recommended_route.get(
                        "recommendation_reason",
                        ""
                    ),

                "score_breakdown":
                    recommended_route.get(
                        "score_breakdown",
                        {}
                    )
            },


            "routes":
                routes,


            "route_count":
                len(routes),


            "adaptive_routing": {

                "enabled":
                    True,

                "weights": {

                    "distance":
                        "40%",

                    "travel_time":
                        "30%",

                    "traffic":
                        "30%"
                },

                "routing_provider":
                    "Google Routes API"
                    if google_used
                    else "OSRM fallback",

                "traffic_data_type":
                    "Google live traffic"
                    if google_used
                    else "Fallback traffic estimate",

                "live_weather":
                    route_weather[
                        "live"
                    ],

                "live_traffic":
                    google_used,

                "ml_model_loaded":
                    model is not None,

                "ml_model_type":
                    type(model).__name__
                    if model is not None
                    else None,

                "weather_provider":
                    "Open-Meteo",

                "google_api_error":
                    google_error
            }

        })


    except requests.exceptions.Timeout:

        return jsonify({

            "status":
                "error",

            "message":
                "External routing service timed out."

        }), 504


    except requests.exceptions.RequestException as e:

        return jsonify({

            "status":
                "error",

            "message":
                f"External service error: {str(e)}"

        }), 502


    except Exception as e:

        print(
            "Route error:",
            e
        )


        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "status":
            "error",

        "message":
            "API endpoint not found."

    }), 404


@app.errorhandler(500)
def internal_error(error):

    return jsonify({

        "status":
            "error",

        "message":
            "Internal server error."

    }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)

    print(
        "AI SMART TRAFFIC INTELLIGENCE"
    )

    print("=" * 70)

    print(
        "Backend:"
        " http://127.0.0.1:5000"
    )

    print(
        "Route API:"
        " POST /api/route"
    )

    print(
        "Prediction API:"
        " POST /api/predict"
    )

    print(
        "Weather API:"
        " POST /api/weather"
    )

    print(
        "Locations API:"
        " GET /api/locations"
    )

    print(
        "Adaptive routing: ENABLED"
    )

    print(
        "ML model loaded:",
        model is not None
    )

    if model is not None:

        print(
            "ML model type:",
            type(model).__name__
        )

    print(
        "Google Routes API:",
        "ENABLED"
        if GOOGLE_MAPS_API_KEY
        else "NOT CONFIGURED"
    )

    print(
        "Traffic mode:",
        "Google TRAFFIC_AWARE"
        if GOOGLE_MAPS_API_KEY
        else "OSRM fallback"
    )

    print(
        "Weather provider: Open-Meteo"
    )

    print("=" * 70)


    # Works locally and on Render/other cloud platforms.
    # Render provides the PORT environment variable automatically.
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )