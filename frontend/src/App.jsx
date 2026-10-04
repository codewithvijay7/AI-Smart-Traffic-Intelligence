import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  MapContainer,
  TileLayer,
  Polyline,
  Marker,
  Tooltip,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE = "http://127.0.0.1:5000";

// ============================================================
// LEAFLET ICONS
// ============================================================

const sourceIcon = new L.Icon({
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

const destinationIcon = new L.Icon({
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
  popupAnchor: [1, -34],
  shadowSize: [41, 41],
});

// ============================================================
// SAFE VALUE HELPERS
// ============================================================

function safeText(value, fallback = "—") {
  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "number") {
    return String(value);
  }

  return fallback;
}

function safeNumber(value, fallback = 0) {
  const n = Number(value);

  return Number.isFinite(n) ? n : fallback;
}

function formatNumber(value, decimals = 1) {
  const n = Number(value);

  if (!Number.isFinite(n)) {
    return "—";
  }

  return n.toFixed(decimals);
}

// ============================================================
// TRAFFIC HELPERS
// ============================================================

function trafficColor(level) {
  const value = String(level || "").toLowerCase();

  if (value === "low") return "#16a34a";
  if (value === "moderate") return "#eab308";
  if (value === "high") return "#f97316";
  if (value === "severe") return "#dc2626";

  return "#64748b";
}

function trafficIcon(level) {
  const value = String(level || "").toLowerCase();

  if (value === "low") return "🟢";
  if (value === "moderate") return "🟡";
  if (value === "high") return "🟠";
  if (value === "severe") return "🔴";

  return "⚪";
}

// ============================================================
// IMPORTANT:
// route.traffic can be either:
//   "Low"
// OR
//   {
//      congestion: "Low",
//      confidence: 100,
//      google_traffic: {...}
//   }
//
// This function ALWAYS returns a string.
// ============================================================

function getTrafficLevel(value, fallback = "Low") {
  if (typeof value === "string") {
    return value;
  }

  if (typeof value === "number") {
    return String(value);
  }

  if (!value || typeof value !== "object") {
    return fallback;
  }

  if (typeof value.congestion === "string") {
    return value.congestion;
  }

  if (typeof value.traffic_level === "string") {
    return value.traffic_level;
  }

  if (typeof value.level === "string") {
    return value.level;
  }

  if (
    value.google_traffic &&
    typeof value.google_traffic === "object"
  ) {
    if (
      typeof value.google_traffic.congestion === "string"
    ) {
      return value.google_traffic.congestion;
    }

    if (
      typeof value.google_traffic.traffic_level ===
      "string"
    ) {
      return value.google_traffic.traffic_level;
    }
  }

  return fallback;
}

function getRouteTrafficLevel(route, fallback = "Low") {
  if (!route || typeof route !== "object") {
    return fallback;
  }

  if (typeof route.traffic_level === "string") {
    return route.traffic_level;
  }

  if (typeof route.congestion === "string") {
    return route.congestion;
  }

  if (typeof route.traffic === "string") {
    return route.traffic;
  }

  if (
    route.traffic &&
    typeof route.traffic === "object"
  ) {
    return getTrafficLevel(
      route.traffic,
      fallback
    );
  }

  return fallback;
}

// ============================================================
// WEATHER ICON
// ============================================================

function weatherIcon(description) {
  const value = String(description || "").toLowerCase();

  if (value.includes("thunder")) return "⛈️";
  if (value.includes("rain")) return "🌧️";
  if (value.includes("cloud")) return "☁️";
  if (value.includes("fog")) return "🌫️";

  return "☀️";
}

// ============================================================
// MAP CONTROLLER
// ============================================================

function MapController({
  routePoints,
  sourcePosition,
  destinationPosition,
}) {
  const map = useMap();

  useEffect(() => {
    if (routePoints.length > 1) {
      map.fitBounds(routePoints, {
        padding: [40, 40],
      });
    } else if (
      sourcePosition &&
      destinationPosition
    ) {
      map.fitBounds(
        [
          sourcePosition,
          destinationPosition,
        ],
        {
          padding: [50, 50],
        }
      );
    }
  }, [
    map,
    routePoints,
    sourcePosition,
    destinationPosition,
  ]);

  return null;
}

// ============================================================
// POSITION HELPER
// ============================================================

function extractPosition(value) {
  if (!value || typeof value !== "object") {
    return null;
  }

  const lat = Number(
    value.lat ?? value.latitude
  );

  const lon = Number(
    value.lon ??
      value.lng ??
      value.longitude
  );

  if (
    Number.isFinite(lat) &&
    Number.isFinite(lon)
  ) {
    return [lat, lon];
  }

  return null;
}

// ============================================================
// MAIN APP
// ============================================================

export default function App() {
  const [locations, setLocations] = useState([]);

  const [source, setSource] =
    useState("Kadapa");

  const [destination, setDestination] =
    useState("Tirupati");

  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [locationsLoading, setLocationsLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  // ==========================================================
  // LOAD LOCATIONS
  // ==========================================================

  useEffect(() => {
    async function loadLocations() {
      try {
        const response = await axios.get(
          `${API_BASE}/api/locations`
        );

        const data = response.data;

        let list = [];

        if (Array.isArray(data)) {
          list = data;
        } else if (
          Array.isArray(data?.locations)
        ) {
          list = data.locations;
        } else if (
          data &&
          typeof data === "object"
        ) {
          list = Object.keys(data);
        }

        // Keep only strings
        list = list
          .map((item) =>
            typeof item === "string"
              ? item
              : item?.name
          )
          .filter(Boolean);

        setLocations(list);

        // Force useful defaults if available
        const kadapa = list.find(
          (item) =>
            item.toLowerCase() === "kadapa"
        );

        const tirupati = list.find(
          (item) =>
            item.toLowerCase() ===
            "tirupati"
        );

        if (kadapa) {
          setSource(kadapa);
        }

        if (tirupati) {
          setDestination(tirupati);
        }
      } catch (err) {
        console.error(
          "Location API error:",
          err
        );
      } finally {
        setLocationsLoading(false);
      }
    }

    loadLocations();
  }, []);

  // ==========================================================
  // CALCULATE ROUTE
  // ==========================================================

  async function calculateRoute() {
    if (!source || !destination) {
      setError(
        "Please select source and destination."
      );
      return;
    }

    if (
      source.toLowerCase() ===
      destination.toLowerCase()
    ) {
      setError(
        "Source and destination cannot be the same."
      );
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await axios.post(
        `${API_BASE}/api/route`,
        {
          source,
          destination,
        }
      );

      console.log(
        "Route API Response:",
        response.data
      );

      setResult(response.data);
    } catch (err) {
      console.error(
        "Route API error:",
        err
      );

      const message =
        err?.response?.data?.error ||
        err?.response?.data?.message ||
        err?.message ||
        "Unable to calculate route.";

      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  // ==========================================================
  // ROUTES
  // ==========================================================

  const routes = useMemo(() => {
    if (!result) {
      return [];
    }

    if (Array.isArray(result.routes)) {
      return result.routes.filter(
        (route) =>
          route &&
          typeof route === "object"
      );
    }

    if (
      result.route &&
      typeof result.route === "object"
    ) {
      return [result.route];
    }

    return [];
  }, [result]);

  // ==========================================================
  // RECOMMENDED ROUTE
  // ==========================================================

  const recommendedIndex = useMemo(() => {
    if (!result) {
      return 0;
    }

    const value =
      result.recommended_route;

    if (typeof value === "number") {
      return Math.max(0, value - 1);
    }

    if (typeof value === "string") {
      const digits =
        value.replace(/\D/g, "");

      if (digits) {
        return Math.max(
          0,
          Number(digits) - 1
        );
      }
    }

    // Some APIs may return an object.
    if (
      value &&
      typeof value === "object"
    ) {
      if (
        typeof value.rank === "number"
      ) {
        return Math.max(
          0,
          value.rank - 1
        );
      }

      if (
        typeof value.index === "number"
      ) {
        return Math.max(
          0,
          value.index
        );
      }

      // Backend may return the recommended route_id.
      if (
        value.route_id !== undefined &&
        Array.isArray(result.routes)
      ) {
        const foundIndex =
          result.routes.findIndex(
            (route) =>
              String(route?.route_id) ===
              String(value.route_id)
          );

        if (foundIndex >= 0) {
          return foundIndex;
        }
      }
    }

    // The backend sorts routes with the recommended route first.
    return 0;
  }, [result]);

  // ==========================================================
  // TRAFFIC
  // ==========================================================

  const traffic =
    result?.traffic &&
    typeof result.traffic === "object"
      ? result.traffic
      : {};

  const googleTraffic =
    traffic.google_traffic &&
    typeof traffic.google_traffic ===
      "object"
      ? traffic.google_traffic
      : {};

  const congestion = getTrafficLevel(
    traffic,
    "Low"
  );

  const googleConfidence =
    googleTraffic.google_confidence ??
    traffic.confidence ??
    0;

  const liveTraffic =
    traffic.live === true ||
    googleTraffic.live === true ||
    result?.live_traffic === true;

  const trafficSource = safeText(
    googleTraffic.data_source ||
      traffic.data_source ||
      result?.routing_provider ||
      "Google Routes API",
    "Google Routes API"
  );

  const trafficScore =
    traffic.traffic_score ??
    googleTraffic.traffic_score ??
    0;

  const normalPercentage =
    googleTraffic.normal_percentage ??
    0;

  const slowPercentage =
    googleTraffic.slow_percentage ??
    0;

  const jamPercentage =
    googleTraffic.traffic_jam_percentage ??
    0;

  const segmentsAnalyzed =
    googleTraffic.segments_analyzed ??
    0;

  const trafficDelay =
    result?.traffic_delay_minutes ??
    traffic.traffic_delay_minutes ??
    0;

  // ==========================================================
  // ML
  // ==========================================================

  const mlPrediction = getTrafficLevel(
    traffic.ml_prediction,
    congestion
  );

  const mlConfidence =
    traffic.ml_confidence ?? 0;

  const mlProbabilities =
    traffic.ml_probabilities &&
    typeof traffic.ml_probabilities ===
      "object"
      ? traffic.ml_probabilities
      : null;

  // ==========================================================
  // WEATHER
  // ==========================================================

  const weather =
    result?.weather &&
    typeof result.weather === "object"
      ? result.weather
      : {};

  const sourceWeather =
    weather.source &&
    typeof weather.source === "object"
      ? weather.source
      : weather.source_weather &&
        typeof weather.source_weather ===
          "object"
      ? weather.source_weather
      : {};

  const destinationWeather =
    weather.destination &&
    typeof weather.destination ===
      "object"
      ? weather.destination
      : weather.destination_weather &&
        typeof weather.destination_weather ===
          "object"
      ? weather.destination_weather
      : {};

  const averageTemperature =
    weather.temperature ??
    weather.route_average?.temperature ??
    (
      safeNumber(
        sourceWeather.temperature
      ) +
      safeNumber(
        destinationWeather.temperature
      )
    ) /
      2;

  const averageRain =
    weather.precipitation ??
    weather.route_average?.precipitation ??
    0;

  // ==========================================================
  // ROUTE COORDINATES
  // ==========================================================

  // Google Routes API normally returns an encoded polyline.
  // This decoder also keeps support for the array formats used
  // by the previous backend/OSRM fallback.
  function decodePolyline(encoded) {
    if (typeof encoded !== "string" || !encoded.length) {
      return [];
    }

    const points = [];
    let index = 0;
    let lat = 0;
    let lng = 0;

    try {
      while (index < encoded.length) {
        let shift = 0;
        let resultValue = 0;
        let byte;

        do {
          byte = encoded.charCodeAt(index++) - 63;
          resultValue |= (byte & 0x1f) << shift;
          shift += 5;
        } while (byte >= 0x20 && index < encoded.length);

        const deltaLat =
          resultValue & 1
            ? ~(resultValue >> 1)
            : resultValue >> 1;

        lat += deltaLat;

        shift = 0;
        resultValue = 0;

        do {
          byte = encoded.charCodeAt(index++) - 63;
          resultValue |= (byte & 0x1f) << shift;
          shift += 5;
        } while (byte >= 0x20 && index < encoded.length);

        const deltaLng =
          resultValue & 1
            ? ~(resultValue >> 1)
            : resultValue >> 1;

        lng += deltaLng;

        points.push([
          lat / 100000,
          lng / 100000,
        ]);
      }
    } catch (decodeError) {
      console.error(
        "Polyline decode error:",
        decodeError
      );
      return [];
    }

    return points;
  }

  function normalizeRoutePoints(raw) {
    if (!Array.isArray(raw)) {
      return [];
    }

    return raw
      .map((point) => {
        if (Array.isArray(point)) {
          const lat = Number(point[0]);
          const lon = Number(point[1]);

          if (
            Number.isFinite(lat) &&
            Number.isFinite(lon)
          ) {
            return [lat, lon];
          }

          return null;
        }

        return extractPosition(point);
      })
      .filter(Boolean);
  }

  function getRoutePoints(route) {
    if (!route || typeof route !== "object") {
      return [];
    }

    // 1. Already-decoded coordinate arrays.
    if (Array.isArray(route.coordinates)) {
      const points = normalizeRoutePoints(
        route.coordinates
      );

      if (points.length > 1) {
        return points;
      }
    }

    if (Array.isArray(route.route)) {
      const points = normalizeRoutePoints(
        route.route
      );

      if (points.length > 1) {
        return points;
      }
    }

    if (Array.isArray(route.polyline)) {
      const points = normalizeRoutePoints(
        route.polyline
      );

      if (points.length > 1) {
        return points;
      }
    }

    // 2. Encoded polyline returned directly by the backend.
    const encodedCandidates = [
      route.encoded_polyline,
      route.encodedPolyline,
      route.polyline,
      route.route_polyline,
      route.routePolyline,
    ];

    for (const encoded of encodedCandidates) {
      if (typeof encoded === "string") {
        const points = decodePolyline(encoded);

        if (points.length > 1) {
          return points;
        }
      }
    }

    // 3. Google-style nested polyline object:
    // { polyline: { encodedPolyline: "..." } }
    if (
      route.polyline &&
      typeof route.polyline === "object"
    ) {
      const encoded =
        route.polyline.encodedPolyline ??
        route.polyline.encoded_polyline;

      if (typeof encoded === "string") {
        const points = decodePolyline(encoded);

        if (points.length > 1) {
          return points;
        }
      }
    }

    // 4. Some backend versions return:
    // { route: { encodedPolyline: "..." } }
    if (
      route.route &&
      typeof route.route === "object" &&
      !Array.isArray(route.route)
    ) {
      const encoded =
        route.route.encodedPolyline ??
        route.route.encoded_polyline;

      if (typeof encoded === "string") {
        const points = decodePolyline(encoded);

        if (points.length > 1) {
          return points;
        }
      }
    }

    return [];
  }

  const mapRoutes = routes.map(
    (route, index) => ({
      route,
      index,
      points: getRoutePoints(route),
    })
  );

  const allPoints =
    mapRoutes.flatMap(
      (item) => item.points
    );

  // ==========================================================
  // SOURCE / DESTINATION POSITIONS
  // ==========================================================

  const sourcePosition =
    extractPosition(
      result?.source_coordinates
    ) ||
    (
      Number.isFinite(
        Number(result?.source_latitude)
      ) &&
      Number.isFinite(
        Number(result?.source_longitude)
      )
        ? [
            Number(
              result.source_latitude
            ),
            Number(
              result.source_longitude
            ),
          ]
        : null
    );

  const destinationPosition =
    extractPosition(
      result?.destination_coordinates
    ) ||
    (
      Number.isFinite(
        Number(
          result?.destination_latitude
        )
      ) &&
      Number.isFinite(
        Number(
          result?.destination_longitude
        )
      )
        ? [
            Number(
              result.destination_latitude
            ),
            Number(
              result.destination_longitude
            ),
          ]
        : null
    );

  const defaultCenter = [
    15.9129,
    79.74,
  ];

  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="app-shell">

      {/* ====================================================
          HEADER
      ==================================================== */}

      <header className="top-header">

        <div>
          <div className="brand-title">
            🚦 AI Smart Traffic Intelligence
          </div>

          <div className="brand-subtitle">
            Real-Time Traffic Analysis &
            Adaptive Route Recommendation
          </div>
        </div>

        <div className="header-status">

          <span className="status-dot" />

          {liveTraffic
            ? "Google Live Traffic Connected"
            : "Traffic System Ready"}

        </div>

      </header>

      <div className="dashboard">

        {/* ==================================================
            SIDEBAR
        ================================================== */}

        <aside className="sidebar">

          {/* ROUTE PLANNING */}

          <div className="sidebar-section">

            <h3>
              🗺️ Route Planning
            </h3>

            <label>
              Source
            </label>

            <select
              value={source}
              onChange={(e) =>
                setSource(e.target.value)
              }
              disabled={
                locationsLoading
              }
            >

              {locations.length === 0 ? (
                <>
                  <option value="Kadapa">
                    Kadapa
                  </option>

                  <option value="Tirupati">
                    Tirupati
                  </option>
                </>
              ) : (
                locations.map(
                  (location, index) => (
                    <option
                      key={index}
                      value={location}
                    >
                      {location}
                    </option>
                  )
                )
              )}

            </select>

            <label>
              Destination
            </label>

            <select
              value={destination}
              onChange={(e) =>
                setDestination(
                  e.target.value
                )
              }
              disabled={
                locationsLoading
              }
            >

              {locations.length === 0 ? (
                <>
                  <option value="Tirupati">
                    Tirupati
                  </option>

                  <option value="Kadapa">
                    Kadapa
                  </option>
                </>
              ) : (
                locations.map(
                  (location, index) => (
                    <option
                      key={index}
                      value={location}
                    >
                      {location}
                    </option>
                  )
                )
              )}

            </select>

            <button
              className="calculate-btn"
              onClick={
                calculateRoute
              }
              disabled={loading}
            >
              {loading
                ? "⏳ Calculating..."
                : "🔍 Find Best Route"}
            </button>

            {error && (
              <div className="error-box">
                ⚠️ {error}
              </div>
            )}

          </div>

          {/* ==================================================
              LIVE TRAFFIC
          ================================================== */}

          {result && (
            <div className="sidebar-section">

              <div className="section-heading-row">

                <h3>
                  🚦 Live Traffic
                </h3>

                <span className="live-badge">
                  LIVE
                </span>

              </div>

              <div className="live-source">

                <span className="live-dot" />

                {trafficSource}

              </div>

              <div className="traffic-main-card">

                <div
                  className="traffic-big-icon"
                  style={{
                    color:
                      trafficColor(
                        congestion
                      ),
                  }}
                >
                  {trafficIcon(
                    congestion
                  )}
                </div>

                <div>

                  <div className="traffic-label">
                    Current Congestion
                  </div>

                  <div
                    className="traffic-value"
                    style={{
                      color:
                        trafficColor(
                          congestion
                        ),
                    }}
                  >
                    {congestion}
                  </div>

                </div>

              </div>

              <div className="mini-stat-grid">

                <div className="mini-stat">

                  <span>
                    Traffic Score
                  </span>

                  <strong>
                    {formatNumber(
                      trafficScore,
                      0
                    )}
                    /100
                  </strong>

                </div>

                <div className="mini-stat">

                  <span>
                    Delay
                  </span>

                  <strong>
                    {formatNumber(
                      trafficDelay
                    )}{" "}
                    min
                  </strong>

                </div>

              </div>

              <div className="confidence-row">

                <div>

                  <span>
                    Google Traffic Confidence
                  </span>

                  <strong>
                    {formatNumber(
                      googleConfidence
                    )}
                    %
                  </strong>

                </div>

                <div className="progress-track">

                  <div
                    className="progress-fill"
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(
                          0,
                          safeNumber(
                            googleConfidence
                          )
                        )
                      )}%`,
                    }}
                  />

                </div>

              </div>

              <div className="traffic-breakdown">

                <div>

                  <span>
                    Normal
                  </span>

                  <strong>
                    {formatNumber(
                      normalPercentage
                    )}
                    %
                  </strong>

                </div>

                <div>

                  <span>
                    Slow
                  </span>

                  <strong>
                    {formatNumber(
                      slowPercentage
                    )}
                    %
                  </strong>

                </div>

                <div>

                  <span>
                    Jam
                  </span>

                  <strong>
                    {formatNumber(
                      jamPercentage
                    )}
                    %
                  </strong>

                </div>

              </div>

              <div className="segments-info">

                📡{" "}
                {formatNumber(
                  segmentsAnalyzed,
                  0
                )}{" "}
                traffic segments analyzed

              </div>

            </div>
          )}

          {/* ==================================================
              AI PREDICTION
          ================================================== */}

          {result && (
            <div className="sidebar-section">

              <h3>
                🤖 AI Traffic Prediction
              </h3>

              <div className="ai-card">

                <div className="ai-model">

                  <span>
                    Model
                  </span>

                  <strong>
                    Random Forest
                  </strong>

                </div>

                <div className="ai-prediction">

                  <span>
                    Predicted Traffic
                  </span>

                  <strong
                    style={{
                      color:
                        trafficColor(
                          mlPrediction
                        ),
                    }}
                  >
                    {trafficIcon(
                      mlPrediction
                    )}{" "}
                    {mlPrediction}
                  </strong>

                </div>

                <div className="confidence-row">

                  <div>

                    <span>
                      ML Confidence
                    </span>

                    <strong>
                      {formatNumber(
                        mlConfidence
                      )}
                      %
                    </strong>

                  </div>

                  <div className="progress-track">

                    <div
                      className="progress-fill ai-progress"
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(
                            0,
                            safeNumber(
                              mlConfidence
                            )
                          )
                        )}%`,
                      }}
                    />

                  </div>

                </div>

              </div>

              {mlProbabilities && (
                <div className="probability-box">

                  <div className="probability-title">
                    Prediction Probabilities
                  </div>

                  {Object.entries(
                    mlProbabilities
                  ).map(
                    (
                      [
                        label,
                        probability,
                      ]
                    ) => {

                      const value =
                        safeNumber(
                          probability
                        );

                      return (
                        <div
                          className="probability-row"
                          key={label}
                        >

                          <span>
                            {safeText(
                              label
                            )}
                          </span>

                          <div className="probability-track">

                            <div
                              className="probability-fill"
                              style={{
                                width: `${Math.min(
                                  100,
                                  Math.max(
                                    0,
                                    value
                                  )
                                )}%`,
                              }}
                            />

                          </div>

                          <strong>
                            {formatNumber(
                              value
                            )}
                            %
                          </strong>

                        </div>
                      );
                    }
                  )}

                </div>
              )}

            </div>
          )}

        </aside>

        {/* ==================================================
            MAIN
        ================================================== */}

        <main className="main-content">

          {/* ==================================================
              SUMMARY
          ================================================== */}

          {result && (
            <div className="summary-grid">

              <div className="summary-card">

                <div className="summary-icon">
                  🛣️
                </div>

                <div>

                  <span>
                    Routes Available
                  </span>

                  <strong>
                    {result.route_count ??
                      routes.length}
                  </strong>

                </div>

              </div>

              <div className="summary-card">

                <div className="summary-icon">
                  📏
                </div>

                <div>

                  <span>
                    Recommended Distance
                  </span>

                  <strong>
                    {formatNumber(
                      routes[
                        recommendedIndex
                      ]?.distance_km ??
                        result.distance_km
                    )}{" "}
                    km
                  </strong>

                </div>

              </div>

              <div className="summary-card">

                <div className="summary-icon">
                  ⏱️
                </div>

                <div>

                  <span>
                    Recommended Time
                  </span>

                  <strong>
                    {formatNumber(
                      routes[
                        recommendedIndex
                      ]?.duration_minutes ??
                        result.duration_minutes
                    )}{" "}
                    min
                  </strong>

                </div>

              </div>

              <div className="summary-card recommended-summary">

                <div className="summary-icon">
                  ⭐
                </div>

                <div>

                  <span>
                    Recommended
                  </span>

                  <strong>
                    Route{" "}
                    {Math.min(
                      recommendedIndex +
                        1,
                      Math.max(
                        routes.length,
                        1
                      )
                    )}
                  </strong>

                </div>

              </div>

            </div>
          )}

          {/* ==================================================
              MAP
          ================================================== */}

          <section className="map-section">

            <div className="map-header">

              <div>

                <h2>
                  🗺️ India Traffic Map
                </h2>

                <p>
                  {result
                    ? `${source} → ${destination}`
                    : "Select locations to calculate an adaptive route"}
                </p>

              </div>

              {result && (
                <div className="map-live-indicator">

                  <span className="live-dot" />

                  Google Traffic-Aware Routing

                </div>
              )}

            </div>

            <div className="map-wrapper">

              <MapContainer
                center={defaultCenter}
                zoom={6}
                scrollWheelZoom
                className="traffic-map"
              >

                <TileLayer
                  attribution="&copy; OpenStreetMap contributors"
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />

                {/* ROUTES */}

                {mapRoutes.map(
                  ({
                    route,
                    index,
                    points,
                  }) => {

                    if (
                      points.length < 2
                    ) {
                      return null;
                    }

                    const recommended =
                      index ===
                      recommendedIndex;

                    const routeTraffic =
                      getRouteTrafficLevel(
                        route,
                        recommended
                          ? congestion
                          : "Low"
                      );

                    return (
                      <Polyline
                        key={`route-${index}`}
                        positions={points}
                        pathOptions={{
                          color:
                            recommended
                              ? "#16a34a"
                              : trafficColor(
                                  routeTraffic
                                ),
                          weight:
                            recommended
                              ? 8
                              : 5,
                          opacity:
                            recommended
                              ? 1
                              : 0.75,
                        }}
                      >

                        <Tooltip sticky>

                          <strong>
                            {recommended
                              ? "⭐ Recommended Route"
                              : `Route ${
                                  index + 1
                                }`}
                          </strong>

                          <br />

                          Traffic:{" "}
                          {routeTraffic}

                        </Tooltip>

                      </Polyline>
                    );
                  }
                )}

                {/* SOURCE */}

                {sourcePosition && (
                  <Marker
                    position={
                      sourcePosition
                    }
                    icon={sourceIcon}
                  >
                    <Tooltip permanent>
                      📍 {source}
                    </Tooltip>
                  </Marker>
                )}

                {/* DESTINATION */}

                {destinationPosition && (
                  <Marker
                    position={
                      destinationPosition
                    }
                    icon={
                      destinationIcon
                    }
                  >
                    <Tooltip permanent>
                      🏁 {destination}
                    </Tooltip>
                  </Marker>
                )}

                <MapController
                  routePoints={allPoints}
                  sourcePosition={
                    sourcePosition
                  }
                  destinationPosition={
                    destinationPosition
                  }
                />

              </MapContainer>

              {/* LEGEND */}

              <div className="map-legend">

                <div className="legend-title">
                  Traffic
                </div>

                <div>
                  <span
                    className="legend-line"
                    style={{
                      background:
                        "#16a34a",
                    }}
                  />
                  Low
                </div>

                <div>
                  <span
                    className="legend-line"
                    style={{
                      background:
                        "#eab308",
                    }}
                  />
                  Moderate
                </div>

                <div>
                  <span
                    className="legend-line"
                    style={{
                      background:
                        "#f97316",
                    }}
                  />
                  High
                </div>

                <div>
                  <span
                    className="legend-line"
                    style={{
                      background:
                        "#dc2626",
                    }}
                  />
                  Severe
                </div>

              </div>

            </div>

          </section>

          {/* ==================================================
              WEATHER
          ================================================== */}

          {result && (
            <section className="weather-section">

              <div className="section-title">

                <div>

                  <h2>
                    🌦️ Live Weather Conditions
                  </h2>

                  <p>
                    Current weather from
                    Open-Meteo
                  </p>

                </div>

                <span className="live-badge">
                  LIVE
                </span>

              </div>

              <div className="weather-grid">

                {/* SOURCE */}

                <div className="weather-card">

                  <div className="weather-card-header">

                    <span>
                      📍 {source}
                    </span>

                    <span>
                      {weatherIcon(
                        sourceWeather.weather_description
                      )}
                    </span>

                  </div>

                  <div className="weather-temperature">

                    {formatNumber(
                      sourceWeather.temperature
                    )}
                    °C

                  </div>

                  <div className="weather-description">

                    {safeText(
                      sourceWeather.weather_description,
                      "Weather data"
                    )}

                  </div>

                  <div className="weather-details">

                    <span>
                      💧{" "}
                      {formatNumber(
                        sourceWeather.humidity,
                        0
                      )}
                      %
                    </span>

                    <span>
                      🌧️{" "}
                      {formatNumber(
                        sourceWeather.precipitation
                      )}{" "}
                      mm
                    </span>

                    <span>
                      💨{" "}
                      {formatNumber(
                        sourceWeather.wind_speed
                      )}{" "}
                      km/h
                    </span>

                  </div>

                </div>

                {/* DESTINATION */}

                <div className="weather-card">

                  <div className="weather-card-header">

                    <span>
                      🏁 {destination}
                    </span>

                    <span>
                      {weatherIcon(
                        destinationWeather.weather_description
                      )}
                    </span>

                  </div>

                  <div className="weather-temperature">

                    {formatNumber(
                      destinationWeather.temperature
                    )}
                    °C

                  </div>

                  <div className="weather-description">

                    {safeText(
                      destinationWeather.weather_description,
                      "Weather data"
                    )}

                  </div>

                  <div className="weather-details">

                    <span>
                      💧{" "}
                      {formatNumber(
                        destinationWeather.humidity,
                        0
                      )}
                      %
                    </span>

                    <span>
                      🌧️{" "}
                      {formatNumber(
                        destinationWeather.precipitation
                      )}{" "}
                      mm
                    </span>

                    <span>
                      💨{" "}
                      {formatNumber(
                        destinationWeather.wind_speed
                      )}{" "}
                      km/h
                    </span>

                  </div>

                </div>

                {/* ROUTE */}

                <div className="weather-card route-weather-card">

                  <div className="weather-card-header">

                    <span>
                      🛣️ Route Conditions
                    </span>

                    <span>
                      🌤️
                    </span>

                  </div>

                  <div className="weather-temperature">

                    {formatNumber(
                      averageTemperature
                    )}
                    °C

                  </div>

                  <div className="weather-description">
                    Average route temperature
                  </div>

                  <div className="weather-details">

                    <span>
                      🌧️ Rain{" "}
                      {formatNumber(
                        averageRain
                      )}{" "}
                      mm
                    </span>

                    <span>
                      ☀️ Live weather
                    </span>

                  </div>

                </div>

              </div>

            </section>
          )}

          {/* ==================================================
              ROUTE OPTIONS
          ================================================== */}

          {result &&
            routes.length > 0 && (
              <section className="routes-section">

                <div className="section-title">

                  <div>

                    <h2>
                      🛣️ Adaptive Route Options
                    </h2>

                    <p>
                      Ranked using distance,
                      travel time and live
                      traffic
                    </p>

                  </div>

                </div>

                <div className="route-cards">

                  {routes.map(
                    (
                      route,
                      index
                    ) => {

                      const recommended =
                        index ===
                        recommendedIndex;

                      const routeTraffic =
                        getRouteTrafficLevel(
                          route,
                          recommended
                            ? congestion
                            : "Moderate"
                        );

                      const routeScore =
                        safeNumber(
                          route.adaptive_score ??
                            route.score ??
                            route.traffic_score,
                          0
                        );

                      const routeDistance =
                        safeNumber(
                          route.distance_km ??
                            route.distance,
                          0
                        );

                      const routeDuration =
                        safeNumber(
                          route.duration_minutes ??
                            route.duration,
                          0
                        );

                      const routePrediction =
                        getTrafficLevel(
                          route.ml_prediction,
                          mlPrediction
                        );

                      const routeConfidence =
                        safeNumber(
                          route.ml_confidence ??
                            mlConfidence
                        );

                      return (
                        <div
                          className={`route-card ${
                            recommended
                              ? "recommended-route"
                              : ""
                          }`}
                          key={index}
                        >

                          {recommended && (
                            <div className="recommended-badge">
                              ⭐ AI RECOMMENDED
                            </div>
                          )}

                          <div className="route-card-top">

                            <div>

                              <span className="route-rank">
                                ROUTE{" "}
                                {index + 1}
                              </span>

                              <h3>
                                {trafficIcon(
                                  routeTraffic
                                )}{" "}
                                {routeTraffic}
                              </h3>

                            </div>

                            <div
                              className="route-score"
                              style={{
                                borderColor:
                                  trafficColor(
                                    routeTraffic
                                  ),
                              }}
                            >

                              <span>
                                Score
                              </span>

                              <strong>
                                {formatNumber(
                                  routeScore,
                                  1
                                )}
                              </strong>

                            </div>

                          </div>

                          <div className="route-stats">

                            <div>

                              <span>
                                📏 Distance
                              </span>

                              <strong>
                                {formatNumber(
                                  routeDistance
                                )}{" "}
                                km
                              </strong>

                            </div>

                            <div>

                              <span>
                                ⏱️ Time
                              </span>

                              <strong>
                                {formatNumber(
                                  routeDuration
                                )}{" "}
                                min
                              </strong>

                            </div>

                            <div>

                              <span>
                                🚦 Traffic
                              </span>

                              <strong
                                style={{
                                  color:
                                    trafficColor(
                                      routeTraffic
                                    ),
                                }}
                              >
                                {routeTraffic}
                              </strong>

                            </div>

                          </div>

                          <div className="route-ai">

                            <div>

                              <span>
                                🤖 AI Prediction
                              </span>

                              <strong>
                                {routePrediction}
                              </strong>

                            </div>

                            <div>

                              <span>
                                AI Confidence
                              </span>

                              <strong>
                                {formatNumber(
                                  routeConfidence
                                )}
                                %
                              </strong>

                            </div>

                          </div>

                          <div className="route-bar">

                            <div
                              style={{
                                width: `${Math.min(
                                  100,
                                  Math.max(
                                    0,
                                    routeScore
                                  )
                                )}%`,
                                background:
                                  trafficColor(
                                    routeTraffic
                                  ),
                              }}
                            />

                          </div>

                          {recommended &&
                            route.recommendation_reason && (
                              <div className="recommendation-reason">
                                <strong>
                                  Why this route?
                                </strong>
                                <span>
                                  {route.recommendation_reason}
                                </span>
                              </div>
                            )}

                        </div>
                      );
                    }
                  )}

                </div>

              </section>
            )}

          {/* ==================================================
              SYSTEM INFORMATION
          ================================================== */}

          {result && (
            <section className="system-info">

              <div className="system-info-item">

                <span>
                  🛰️ Routing
                </span>

                <strong>
                  {safeText(
                    result.routing_provider,
                    trafficSource
                  )}
                </strong>

              </div>

              <div className="system-info-item">

                <span>
                  🚦 Traffic
                </span>

                <strong>
                  {liveTraffic
                    ? "Google Live"
                    : "Unavailable"}
                </strong>

              </div>

              <div className="system-info-item">

                <span>
                  🤖 AI Model
                </span>

                <strong>
                  Random Forest
                </strong>

              </div>

              <div className="system-info-item">

                <span>
                  🌦️ Weather
                </span>

                <strong>
                  Open-Meteo
                </strong>

              </div>

            </section>
          )}

          {/* ==================================================
              EMPTY STATE
          ================================================== */}

          {!result &&
            !loading && (
              <div className="empty-state">

                <div className="empty-icon">
                  🚦
                </div>

                <h2>
                  AI-Powered Smart Traffic
                  Intelligence
                </h2>

                <p>
                  Select your source and
                  destination to analyze
                  live traffic, predict
                  congestion using AI and
                  find the best adaptive
                  route.
                </p>

                <button
                  className="calculate-btn empty-button"
                  onClick={
                    calculateRoute
                  }
                >
                  🗺️ Analyze Route
                </button>

              </div>
            )}

          {/* ==================================================
              FOOTER
          ================================================== */}

          <footer className="footer">

            <span>
              AI Smart Traffic Intelligence
            </span>

            <span>
              Google Routes API • Random
              Forest • Open-Meteo •
              OpenStreetMap
            </span>

          </footer>

        </main>

      </div>

    </div>
  );
}