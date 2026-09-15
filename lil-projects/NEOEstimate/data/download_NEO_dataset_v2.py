import os, socket, time
import pandas as pd, requests
from dotenv import load_dotenv

curr_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(curr_dir, ".env"))
load_dotenv(os.path.join(os.path.dirname(curr_dir), "NEODodge", ".env"))
API_KEY = os.getenv("NASA_API_KEY") or input("NASA API KEY: ")

base = "https://api.nasa.gov/neo/rest/v1/neo/browse"
csv_path = os.path.join(curr_dir, "neo.csv")
objects_path = os.path.join(curr_dir, "neo_objects.csv")
progress_path = os.path.join(curr_dir, ".progress")

APPROACH_COLS = ["name", "magnitude", "diam_min_m", "diam_max_m",
                 "hazardous", "date", "vel_km_h", "miss_km", "orbiting"]
ORBITAL_NUM = ["eccentricity", "semi_major_axis", "inclination",
               "perihelion_distance", "aphelion_distance",
               "minimum_orbit_intersection", "orbital_period",
               "jupiter_tisserand_invariant", "data_arc_in_days",
               "observations_used", "orbit_uncertainty"]
OBJECT_COLS = (["name", "magnitude", "diam_min_m", "diam_max_m", "hazardous",
                "is_sentry_object", "orbit_class_type"] + ORBITAL_NUM)
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


def rows_from_neo(neo):
    meters = (neo.get("estimated_diameter") or {}).get("meters") or {}
    orbit = neo.get("orbital_data") or {}
    orbit_class = orbit.get("orbit_class") or {}
    base_obj = {
        "name": neo.get("name"),
        "magnitude": neo.get("absolute_magnitude_h"),
        "diam_min_m": meters.get("estimated_diameter_min"),
        "diam_max_m": meters.get("estimated_diameter_max"),
        "hazardous": int(bool(neo.get("is_potentially_hazardous_asteroid"))),
    }

    obj = dict(base_obj)
    obj["is_sentry_object"] = int(bool(neo.get("is_sentry_object")))
    obj["orbit_class_type"] = orbit_class.get("orbit_class_type")
    for k in ORBITAL_NUM:
        obj[k] = orbit.get(k)

    rows = []
    for cad in neo.get("close_approach_data", []):
        row = dict(base_obj)
        rv = cad.get("relative_velocity") or {}
        md = cad.get("miss_distance") or {}
        row.update({
            "date": cad.get("close_approach_date"),
            "vel_km_h": rv.get("kilometers_per_hour"),
            "miss_km": md.get("kilometers"),
            "orbiting": cad.get("orbiting_body"),
        })
        rows.append(row)
    return obj, rows


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


def main():
    progress = load_progress()
    fresh = (progress is None
             or not (os.path.exists(csv_path)
                     and os.path.exists(objects_path)))
    page = 0 if fresh else progress + 1

    if fresh:
        pd.DataFrame(columns=APPROACH_COLS).to_csv(csv_path, index=False)
        pd.DataFrame(columns=OBJECT_COLS).to_csv(objects_path, index=False)
        print("arrancando de cero (descarga completa)")
    else:
        print(f"retomando desde la página {page}")

    session = requests.Session()
    try:
        while page < MAX_PAGES:
            resp = fetch_page(session, page)
            if resp is None:
                save_progress(page - 1)
                print("cuota agotada papu")
                break
            if resp is False:
                save_progress(page - 1)
                print("abortado por fallos de red repetidos")
                break

            obj_rows, app_rows = [], []
            for neo in resp.json()["near_earth_objects"]:
                obj, rows = rows_from_neo(neo)
                obj_rows.append(obj)
                app_rows.extend(rows)
            if obj_rows:
                pd.DataFrame(obj_rows).to_csv(objects_path, mode="a",
                                              header=False, index=False)
            if app_rows:
                pd.DataFrame(app_rows).to_csv(csv_path, mode="a",
                                              header=False, index=False)

            save_progress(page)
            if page % chunk == 0 or page == MAX_PAGES - 1:
                rem = remaining_quota(resp)
                extra = f" | remanentes: {rem}" if rem is not None else ""
                print(f"page {page}: +{len(app_rows)} aprox, +{len(obj_rows)} objetos{extra}")
            page += 1
            time.sleep(0.1)
    except InvalidKeyError:
        save_progress(page - 1)
        print("la API rechazó la key")
    except KeyboardInterrupt:
        save_progress(page - 1)
        print(f"\ndetenido en página {page - 1}")

    df = pd.read_csv(csv_path)
    objects = pd.read_csv(objects_path)
    print(f"OK: {df.shape[0]} aproximaciones x {df.shape[1]} en neo.csv | "
          f"{objects.shape[0]} objetos x {objects.shape[1]} en neo_objects.csv")


if __name__ == "__main__":
    main()
