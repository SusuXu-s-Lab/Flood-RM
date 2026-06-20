#!/usr/bin/env julia

using JSON

function parse_cli(args)
    if length(args) < 3
        error("usage: powermodels_onm_smoke.jl <network.dss> <settings.json> <output.json> [--events events.json] [--mld]")
    end
    parsed = Dict{String,Any}(
        "network" => args[1],
        "settings" => args[2],
        "output" => args[3],
        "events" => nothing,
        "mld" => false,
    )
    i = 4
    while i <= length(args)
        if args[i] == "--events"
            i += 1
            parsed["events"] = args[i]
        elseif args[i] == "--mld"
            parsed["mld"] = true
        else
            error("unknown argument: $(args[i])")
        end
        i += 1
    end
    return parsed
end

function count_section(data, key)
    value = get(data, key, Dict())
    return value isa AbstractDict ? length(value) : 0
end

function write_summary(path, summary)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, summary, 2)
        write(io, "\n")
    end
end

function main()
    args = parse_cli(ARGS)
    summary = Dict{String,Any}(
        "status" => "error",
        "network" => args["network"],
        "settings" => args["settings"],
        "events" => args["events"],
        "settings_validation" => Dict("strict" => true, "passed" => false),
    )

    try
        import PowerModelsONM as ONM
        import PowerModelsDistribution as PMD

        eng, mn_eng = ONM.parse_network(args["network"])
        settings = ONM.parse_settings(args["settings"]; validate=true)

        summary["status"] = "ok"
        summary["settings_validation"] = Dict(
            "strict" => true,
            "passed" => true,
            "section_count" => length(settings),
        )
        summary["bus_count"] = count_section(eng, "bus")
        summary["load_count"] = count_section(eng, "load")
        summary["line_count"] = count_section(eng, "line")
        summary["transformer_count"] = count_section(eng, "transformer")
        summary["switch_count"] = count_section(eng, "switch")
        summary["generator_count"] = count_section(eng, "generator")
        summary["storage_count"] = count_section(eng, "storage")

        if args["events"] !== nothing
            raw_events = ONM.parse_events(args["events"]; validate=true)
            summary["events_validation"] = Dict(
                "strict" => true,
                "passed" => true,
                "event_count" => length(raw_events),
            )
            try
                parsed_events = ONM.parse_events(raw_events, mn_eng)
                summary["events_validation"]["parsed_event_count"] = length(parsed_events)
            catch event_parse_error
                summary["events_validation"]["parsed_event_count"] = missing
                summary["events_validation"]["parse_warning"] = sprint(showerror, event_parse_error)
            end
        end

        if args["mld"]
            import HiGHS
            import InfrastructureModels as IM
            import JuMP

            optimizer = JuMP.optimizer_with_attributes(HiGHS.Optimizer, "output_flag" => false)
            math = ONM.transform_data_model(eng)
            model = ONM.instantiate_onm_model(math, PMD.LPUBFDiagPowerModel, ONM.build_block_mld)
            result = IM.optimize_model!(model; optimizer=optimizer)
            summary["mld"] = Dict(
                "attempted" => true,
                "solved" => true,
                "termination_status" => string(get(result, "termination_status", "")),
                "objective" => get(result, "objective", missing),
            )
        else
            summary["mld"] = Dict("attempted" => false, "solved" => false)
        end
    catch err
        summary["status"] = "error"
        summary["error_summary"] = sprint(showerror, err)
        summary["error_type"] = string(typeof(err))
    end

    write_summary(args["output"], summary)
end

main()
