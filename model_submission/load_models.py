import os
import joblib

for case in ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]:
    path = os.path.join("saved_models", case + ".joblib")
    artifact = joblib.load(path)

    print("\nLoaded:", path)
    print("Targets:", sorted(artifact.keys()))

    for target, entry in artifact.items():
        print(
            "%s: model=%s, reg=%s, clf=%s" % (
                target,
                entry.get("model_name", "unknown"),
                type(entry["reg"]).__name__,
                type(entry["clf"]).__name__,
            )
        )
