from pathlib import Path
import yaml

from wflow_v2.wflow_boundary import build_base_models, plan_domain, prepare_states, require_event_boundary, run_event_boundary


def main(location_root=Path("locations/example"), event_id="E0001"):
    location_root = Path(location_root)
    config = yaml.safe_load((location_root / "config.yaml").read_text())

    submodels = plan_domain(config, location_root)
    print(f"planned {len(submodels)} Wflow submodel(s)")

    print(build_base_models(config, location_root, execute=False).to_string(index=False))
    print(prepare_states(config, location_root).to_string(index=False))

    run = run_event_boundary(config, location_root, event_id, execute=True)
    print(run.to_series().to_string())

    accepted = require_event_boundary(config, location_root, event_id)
    print(accepted.to_string())


if __name__ == "__main__":
    main()
