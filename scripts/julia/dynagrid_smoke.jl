#!/usr/bin/env julia

using JSON

function parse_cli(args)
    if length(args) < 3
        error("usage: dynagrid_smoke.jl <network.dss> <settings.json> <output.json> [--events events.json]")
    end
    parsed = Dict{String,Any}(
        "network" => args[1],
        "settings" => args[2],
        "output" => args[3],
        "events" => nothing,
    )
    i = 4
    while i <= length(args)
        if args[i] == "--events"
            i += 1
            parsed["events"] = args[i]
        else
            error("unknown argument: $(args[i])")
        end
        i += 1
    end
    return parsed
end

function write_summary(path, summary)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, summary, 2)
        write(io, "\n")
    end
end

function load_dynagrid!()
    dynagrid_path = get(ENV, "NRELDYNAGRID_PATH", "")
    if isempty(dynagrid_path)
        error("NRELDYNAGRID_PATH must point at a NatLabRockies/NRELDynaGrid source checkout")
    end
    entrypoint = joinpath(dynagrid_path, "src", "OnlineOptDynaGrid.jl")
    if !isfile(entrypoint)
        error("NRELDYNAGRID_PATH does not contain src/OnlineOptDynaGrid.jl: $(dynagrid_path)")
    end
    Base.include(Main, entrypoint)
    return getfield(Main, :OnlineOptDynaGrid)
end

function count_nested_outputs(result, names)
    total = 0
    if result isa AbstractDict
        for name in names
            value = get(result, name, nothing)
            if value isa AbstractDict || value isa AbstractArray
                total += length(value)
            elseif value !== nothing
                total += 1
            end
        end
    end
    return total
end

function main()
    args = parse_cli(ARGS)
    summary = Dict{String,Any}(
        "status" => "error",
        "network" => args["network"],
        "settings" => args["settings"],
        "events" => args["events"],
    )

    try
        import PowerModelsONM as ONM
        import PowerModelsDistribution as PMD
        import HiGHS
        import InfrastructureModels as IM
        import JuMP

        OnlineOptDynaGrid = load_dynagrid!()

        eng, _mn_eng = ONM.parse_network(args["network"])
        _settings = ONM.parse_settings(args["settings"]; validate=true)
        PMD.apply_voltage_bounds!(eng; vm_lb=0.95, vm_ub=1.05)
        math = ONM.transform_data_model(eng)

        optimizer = JuMP.optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
        pm_base = ONM.instantiate_onm_model(math, PMD.LPUBFDiagPowerModel, ONM.build_block_mld)
        mld_result = IM.optimize_model!(pm_base; optimizer=optimizer)

        time_periods = parse(Int, get(ENV, "DYNAGRID_TIME_PERIODS", "1"))
        time_steps = parse(Int, get(ENV, "DYNAGRID_TIME_STEPS", "3"))
        opt_data = Dict{String,Any}(
            "pm_baseCase" => pm_base,
            "TimePeriods" => time_periods,
            "TimeSteps" => time_steps,
            "epsilon" => 1.0e-4,
            "alpha" => 1.0,
            "case_math" => math,
            "case_math_bounds" => deepcopy(math),
            "results" => Dict(1 => mld_result),
            "NL_solver" => optimizer,
        )

        opt_data = OnlineOptDynaGrid.GenerateLoadTimeSeriesMATH(opt_data)
        online_result = OnlineOptDynaGrid.runOptOnline(opt_data)

        summary["status"] = "ok"
        summary["time_periods"] = time_periods
        summary["time_steps"] = time_steps
        summary["mld_termination_status"] = string(get(mld_result, "termination_status", ""))
        summary["tracked_voltage_count"] = count_nested_outputs(
            online_result,
            ["Voltage", "voltage", "vm", "Voltages", "V"],
        )
        summary["der_setpoint_count"] = count_nested_outputs(
            online_result,
            ["DER", "der", "pg", "qg", "DERSetpoints", "setpoints"],
        )
    catch err
        summary["status"] = "error"
        summary["error_summary"] = sprint(showerror, err)
        summary["error_type"] = string(typeof(err))
    end

    write_summary(args["output"], summary)
end

main()
