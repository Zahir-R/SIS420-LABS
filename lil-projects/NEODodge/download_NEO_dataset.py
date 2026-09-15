import csv
import os, socket, time
import pandas as pd, requests
from dotenv import load_dotenv

def safe_float(val, default=float("nan")):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

curr_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(curr_dir, ".env"))
API_KEY = os.getenv("NASA_API_KEY") or input("NASA API KEY: ")

base = "https://api.nasa.gov/neo/rest/v1/neo/browse"
csv_path = os.path.join(curr_dir, "neo.csv")
progress_path = os.path.join(curr_dir, ".progress")

cols = ["name", "neo_reference_id", "magnitude", "diam_min_m", "diam_max_m",
        "hazardous", "is_sentry", "date", "date_full", "epoch_date",
        "vel_km_h", "vel_km_s", "miss_km", "miss_au", "miss_lunar", "orbiting",
        "eccentricity", "semi_major_axis", "inclination",
        "ascending_node_longitude", "orbital_period", "perihelion_distance",
        "aphelion_distance", "mean_anomaly", "mean_motion",
        "perihelion_argument", "tisserand_inv", "moid_au",
        "orbit_uncertainty", "obs_used", "data_arc_days", "orbit_class_type"]
MAX_PAGES = 600
size = 20
retries = 5
backoff = [2, 4, 8, 16]
timeout = 10
chunk = 50


class InvalidKeyError(Exception):
    pass


def load_progress():
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            return int(f.read().strip())
    return None


def save_progress(page):
    with open(progress_path, "w") as f:
        f.write(str(page))


orbital_src = {"tisserand_inv": "jupiter_tisserand_invariant",
               "moid_au": "minimum_orbit_intersection",
               "obs_used": "observations_used",
               "data_arc_days": "data_arc_in_days",
               "orbit_uncertainty": "orbit_uncertainty",
               "orbit_class_type": ("orbit_class", "orbit_class_type")}


def rows_from_neo(neo):
    meters = (neo.get("estimated_diameter") or {}).get("meters") or {}
    od = neo.get("orbital_data") or {}
    asteroid = {
        "name": neo.get("name"),
        "neo_reference_id": neo.get("neo_reference_id"),
        "magnitude": neo.get("absolute_magnitude_h"),
        "diam_min_m": meters.get("estimated_diameter_min"),
        "diam_max_m": meters.get("estimated_diameter_max"),
        "hazardous": int(bool(neo.get("is_potentially_hazardous_asteroid"))),
        "is_sentry": int(bool(neo.get("is_sentry_object"))),
    }
    for out, src in orbital_src.items():
        if isinstance(src, tuple):
            obj = od.get(src[0]) or {}
            asteroid[out] = obj.get(src[1])
        else:
            asteroid[out] = safe_float(od.get(src))
    for f in ["eccentricity", "semi_major_axis", "inclination",
              "ascending_node_longitude", "orbital_period",
              "perihelion_distance", "aphelion_distance", "mean_anomaly",
              "mean_motion", "perihelion_argument"]:
        asteroid[f] = safe_float(od.get(f))
    rows = []
    for cad in neo.get("close_approach_data", []):
        row = dict(asteroid)
        rv = cad.get("relative_velocity") or {}
        md = cad.get("miss_distance") or {}
        row.update({
            "date": cad.get("close_approach_date"),
            "date_full": cad.get("close_approach_date_full"),
            "epoch_date": safe_float(cad.get("epoch_date_close_approach")),
            "vel_km_h": rv.get("kilometers_per_hour"),
            "vel_km_s": rv.get("kilometers_per_second"),
            "miss_km": md.get("kilometers"),
            "miss_au": md.get("astronomical"),
            "miss_lunar": md.get("lunar"),
            "orbiting": cad.get("orbiting_body"),
        })
        rows.append(row)
    return rows


def fetch_page(session, page):
    params = {"page": page, "size": size, "api_key": API_KEY}
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(base, params=params, timeout=timeout)
            if resp.status_code in (403, 429):
                if "invalid" in resp.text.lower():
                    raise InvalidKeyError()
                return None
            if resp.status_code >= 500:
                raise requests.ConnectionError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp
        except InvalidKeyError:
            raise
        except (requests.ConnectionError, requests.Timeout, socket.gaierror) as exc:
            print(f"  fallo de red en página {page} ({type(exc).__name__}): "
                  f"reintento {attempt}/{retries}")
            if attempt < retries:
                time.sleep(backoff[attempt - 1])
    return False


def remaining_quota(resp):
    try:
        return int(resp.headers.get("X-RateLimit-Remaining", ""))
    except (TypeError, ValueError):
        return None


def frame(rows):
    return pd.DataFrame(rows).reindex(columns=cols)


def main():
    progress = load_progress()
    fresh = progress is None or not os.path.exists(csv_path)
    page = 0 if fresh else progress + 1

    if fresh:
        pd.DataFrame(columns=cols).to_csv(csv_path, index=False)
        print("arrancando de cero")
    else:
        print(f"retomando desde la página {page}")

    session = requests.Session()
    if fresh:
        first = fetch_page(session, 0)
        if first is None:
            print("cuota agotada: vuelve en 1 hora.")
            return
        if first is False:
            print("API no responde")
            return
        rows = []
        for neo in first.json()["near_earth_objects"]:
            rows.extend(rows_from_neo(neo))
        frame(rows).to_csv(csv_path, mode="a", header=False, index=False)
        save_progress(0)
        rem = remaining_quota(first)
        extra = f" | remaining: {rem}" if rem is not None else ""
        print(f"page 0: +{len(rows)} filas{extra}")
        page = 1

    try:
        while page < MAX_PAGES:
            resp = fetch_page(session, page)
            if resp is None:
                save_progress(page - 1)
                print(f"cuota agotada en página {page}: Vuelve en 1 hora aprox.")
                break
            if resp is False:
                save_progress(page - 1)
                print(f"abortado por fallos de red repetidos en página {page}: ")
                break

            rows = []
            for neo in resp.json()["near_earth_objects"]:
                rows.extend(rows_from_neo(neo))
            frame(rows).to_csv(csv_path, mode="a", header=False, index=False)

            save_progress(page)
            if page % chunk == 0 or page == MAX_PAGES - 1:
                rem = remaining_quota(resp)
                extra = f" | remaining: {rem}" if rem is not None else ""
                print(f"page {page}: +{len(rows)} filas{extra}")
            page += 1
            time.sleep(0.1)
    except InvalidKeyError:
        save_progress(page - 1)
        print("la API rechazó la key: revisa la NASA_API_KEY")
    except KeyboardInterrupt:
        save_progress(page - 1)
        print(f"\ndetenido en página {page - 1}")

    df = pd.read_csv(csv_path)
    print(f"OK: {df.shape[0]} filas y {df.shape[1]} columnas en neo.csv")


if __name__ == "__main__":
    main()
