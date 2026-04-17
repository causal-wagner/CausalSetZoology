function _validate_links_only(kind::AbstractString, links_only::Bool)::Nothing
    supported_kinds = (
        "grid",
        "manifoldlike_simply_connected",
        "manifoldlike_non_simply_connected",
        "minkowski_quasicrystal",
        "minkowski_sprinkling",
    )
    if links_only && !(kind in supported_kinds)
        throw(ArgumentError("links_only=true is only supported for grid, manifoldlike_simply_connected, manifoldlike_non_simply_connected, minkowski_quasicrystal, and minkowski_sprinkling"))
    end
    return nothing
end

function generate_batch(
    b::Int,
    batchsize::Int,
    N::Int,
    kind::AbstractString,
    seed::Int;

    cset_size::Union{Nothing,Int} = nothing,

    link_probability::Union{Nothing,Float64}=nothing,
    D::Union{Nothing,Int}=nothing,
    cut_restriction::Union{Nothing,String}=nothing,
    big_crystal=nothing,

    mink::Union{Nothing, CausalSets.MinkowskiManifold}=nothing,
    causal_diamond_boundary::Union{Nothing, CausalSets.CausalDiamondBoundary}=nothing,

    ndistr::Union{Nothing, Distributions.Distribution}=nothing,

    rdistr::Union{Nothing, Distributions.Distribution}=nothing,
    genus_distr::Union{Nothing, Distributions.Distribution}=nothing,
    num_boundary_cuts_distr::Union{Nothing, Distributions.Distribution}=nothing,

    lattice_distr::Union{Nothing, Distributions.Distribution}=nothing,
    lattices::Union{Nothing,Vector{String}}=nothing,
    segment_ratio_distr::Union{Nothing, Distributions.Distribution}=nothing,
    rotate_angle_distr::Union{Nothing, Distributions.Distribution}=nothing,
    oblique_angle_distr::Union{Nothing, Distributions.Distribution}=nothing,

    non_manifoldlikeness_distr::Union{Nothing, Distributions.Distribution}=nothing,

    layers_distr::Union{Nothing, Distributions.Distribution}=nothing,
    link_probability_distr::Union{Nothing, Distributions.Distribution}=nothing,

    connectivity_distr::Union{Nothing, Distributions.Distribution}=nothing,
    links_only::Bool=false,
)::NamedTuple

    _validate_links_only(kind, links_only)

    start_i = (b - 1) * batchsize + 1
    end_i = min(b * batchsize, N)

    csets_b = CausalSets.BitArrayCauset[]
    links_b = SparseLinksCauset[]
    r_b  = Float64[]
    order_b = Int[]
    num_boundary_cuts_b = Int[]
    genus_b = Int[]
    num_layers_b = Int[]
    std_b = Float64[]
    segment_ratio_b = Float64[]
    segment_angle_b = Float64[]
    rotation_angle_b = Float64[]
    rel_num_flips_b = Float64[]
    rel_size_KR_b = Float64[]
    link_probability_b = Float64[]
    lattice_b = String[]
    trans_in_b  = Float64[]
    trans_out_b = Float64[]

    for i in start_i:end_i
        rng = Random.MersenneTwister(seed + i)
        cset_size_i = if isnothing(cset_size)
            if isnothing(ndistr)
                ArgumentError("Either cset_size or ndistr must be provided")
            end
            ndistr = ndistr
            rand(rng, ndistr)
        else
            cset_size
        end

        if kind == "minkowski_sprinkling"
            sprinkling = CausalSets.generate_sprinkling(mink, causal_diamond_boundary, cset_size_i; rng = rng)
            causet = links_only ? SparseLinksCauset(mink, sprinkling) : CausalSets.BitArrayCauset(mink, sprinkling)

        elseif kind == "minkowski_quasicrystal"
            if big_crystal === nothing
                throw(ArgumentError("big_crystal must be provided for kind = \"minkowski_quasicrystal\""))
            end
            ϵ = sqrt(cset_size_i / length(big_crystal[1]))
            trans_distr = Distributions.Uniform(ϵ, 1 - ϵ)
            αin = rand(rng, trans_distr)
            αout = rand(rng, trans_distr)
            causet = create_Minkowski_quasicrystal_cset(
                cset_size_i,
                (αin, αout);
                crystal = big_crystal,
                exact_size = true,
                deviation_from_mean_size = .1,
                max_iter = 100,
                links = links_only,
            )
            push!(trans_in_b, αin)
            push!(trans_out_b, αout)

        elseif kind == "manifoldlike_simply_connected"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            causet, _, __ = make_polynomial_manifold_cset(
                cset_size_i,
                rng,
                order,
                Float64(r);
                d = D === nothing ? 2 : D,
                links = links_only,
            )
            push!(r_b, r)
            push!(order_b, order)

        elseif kind == "manifoldlike_non_simply_connected"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            num_boundary_cuts = cut_restriction == "free_cuts" ? 0 : rand(rng, num_boundary_cuts_distr)
            genus = cut_restriction == "boundary_cuts" ? 0 : rand(rng, genus_distr)
            causet, _, _, _ = make_polynomial_manifold_cset_with_nontrivial_topology(
                cset_size_i,
                num_boundary_cuts,
                genus,
                rng,
                order,
                r;
                links = links_only,
            )
            push!(r_b, r)
            push!(order_b, order)
            push!(num_boundary_cuts_b, num_boundary_cuts)
            push!(genus_b, genus)

        elseif kind == "destroyed"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            num_flips = Int64(ceil(cset_size_i * (cset_size_i - 1) /2 * rand(rng, non_manifoldlikeness_distr)))
            causet, _, _ = QuantumGrav.destroy_manifold_cset(cset_size_i, num_flips, rng, order, r)
            push!(r_b, r)
            push!(order_b, order)
            push!(rel_num_flips_b, num_flips / (cset_size_i * (cset_size_i - 1) / 2 ))

        elseif kind == "merged"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            link_probability_value = link_probability === nothing ? rand(rng, link_probability_distr) : link_probability
            n2_rel = rand(rng, non_manifoldlikeness_distr)
            causet, _, _ = QuantumGrav.insert_KR_into_manifoldlike(cset_size_i, order, r, link_probability_value; rng = rng, n2_rel = n2_rel)
            push!(r_b, r)
            push!(order_b, order)
            push!(rel_size_KR_b, n2_rel)
            push!(link_probability_b, link_probability_value)

        elseif kind == "grid"
            max_grid_tries = 20
            grid_ok = false
            lattice = ""
            segment_ratio = 0.0
            rotate_angle = 0.0
            oblique_angle = 0.0
            for _ in 1:max_grid_tries
                lattice = lattices[rand(rng, lattice_distr)]
                segment_ratio = rand(rng, segment_ratio_distr)
                rotate_angle = rand(rng, rotate_angle_distr)
                oblique_angle = rand(rng, oblique_angle_distr)
                try
                    causet, _, _ = create_grid_causet_in_boundary_2D(
                        cset_size_i,
                        lattice,
                        CausalSets.BoxBoundary{2}(((0.0, -0.5), (1.0, 0.5))),
                        CausalSets.MinkowskiManifold{2}();
                        b = segment_ratio,
                        gamma_deg = oblique_angle,
                        rotate_deg = rotate_angle,
                        links = links_only,
                    )
                    grid_ok = true
                    break
                catch err
                    if !(err isa ArgumentError && occursin("Boundary shell too small", sprint(showerror, err)))
                        rethrow()
                    end
                end
            end
            grid_ok || error(
                "Grid generation failed after $(max_grid_tries) retries (boundary shell too small). " *
                "Consider narrowing segment_ratio/oblique_angle ranges."
            )
            push!(segment_ratio_b, segment_ratio)
            push!(segment_angle_b, oblique_angle)
            push!(rotation_angle_b, rotate_angle)
            push!(lattice_b, lattice)

        elseif kind == "random"
            connectivity_goal = rand(rng, connectivity_distr)
            abs_tol = 1e-2
            converged = false
            tries = 0
            num_new_goals = 1
            cset_try = nothing
            while num_new_goals < 5
                while !converged && tries ≤ 100
                    cset_try, converged = QuantumGrav.sample_bitarray_causet_by_connectivity(
                        cset_size_i,
                        connectivity_goal,
                        20,
                        rng;
                        abs_tol = abs_tol,
                    )
                    tries += 1
                end
                if converged
                    break
                else
                    @warn "Skipping causet after 100 failed attempts"
                end

                num_new_goals += 1
                if num_new_goals == 5
                    error("Did not converge 5 times after 100 failed attempts. Maybe change connectivity_distr")
                end
            end
            causet = cset_try

        elseif kind == "layered"
            num_layers = rand(rng, layers_distr)
            link_probability = rand(rng, link_probability_distr)
            std_distr = Distributions.Uniform(0., Float64(cset_size_i / (2 * num_layers)))
            std = rand(rng, std_distr)
            causet, _ = QuantumGrav.create_random_layered_causet(cset_size_i, num_layers; p = link_probability, rng = rng, standard_deviation = std)
            push!(num_layers_b, num_layers)
            push!(std_b, std)
        end

        if links_only
            links = causet
        else
            cset = causet
            links = SparseLinksCauset(cset)
        end

        if !links_only
            push!(csets_b, cset)
        end
        push!(links_b, links)
    end

    return (
        csets_b = csets_b,
        links_b = links_b,
        r_b = r_b,
        order_b = order_b,
        num_boundary_cuts_b = num_boundary_cuts_b,
        genus_b = genus_b,
        num_layers_b = num_layers_b,
        std_b = std_b,
        segment_ratio_b = segment_ratio_b,
        segment_angle_b = segment_angle_b,
        rotation_angle_b = rotation_angle_b,
        rel_num_flips_b = rel_num_flips_b,
        rel_size_KR_b = rel_size_KR_b,
        link_probability_b = link_probability_b,
        lattice_b = lattice_b,
        trans_in_b = trans_in_b,
        trans_out_b = trans_out_b,
    )
end

function create_dataset_and_save(
    out_path::String,
    kind::AbstractString,
    num_workers::Int,
    batchsize::Int,
    nbatches::Int,
    N::Int,
    config::Dict,
    seed::Int;

    cset_size::Union{Nothing,Int}=nothing,

    link_probability::Union{Nothing,Float64}=nothing,
    D::Union{Nothing,Int}=nothing,
    cut_restriction::Union{Nothing,String}=nothing,
    big_crystal=nothing,

    ndistr::Union{Nothing, Distributions.Distribution}=nothing,

    rdistr::Union{Nothing, Distributions.Distribution}=nothing,
    genus_distr::Union{Nothing, Distributions.Distribution}=nothing,
    num_boundary_cuts_distr::Union{Nothing, Distributions.Distribution}=nothing,

    lattice_distr::Union{Nothing, Distributions.Distribution}=nothing,
    lattices::Union{Nothing,Vector{String}}=nothing,
    segment_ratio_distr::Union{Nothing, Distributions.Distribution}=nothing,
    rotate_angle_distr::Union{Nothing, Distributions.Distribution}=nothing,
    oblique_angle_distr::Union{Nothing, Distributions.Distribution}=nothing,

    non_manifoldlikeness_distr::Union{Nothing, Distributions.Distribution}=nothing,

    layers_distr::Union{Nothing, Distributions.Distribution}=nothing,
    link_probability_distr::Union{Nothing, Distributions.Distribution}=nothing,

    connectivity_distr::Union{Nothing, Distributions.Distribution}=nothing,

    mink::Union{Nothing, CausalSets.MinkowskiManifold}=nothing,
    causal_diamond_boundary::Union{Nothing, CausalSets.CausalDiamondBoundary}=nothing,
    links_only::Bool=false,

)::Nothing

    _validate_links_only(kind, links_only)

    JLD2.jldopen(out_path, "w") do fout
        fout["meta/batchsize"] = batchsize
        fout["meta/nbatches"]  = nbatches
        fout["meta/N"]         = N
        fout["meta/config"]    = config

        p = ProgressMeter.Progress(N; desc = "Creating causal sets")
        pending = Dict{Int,Any}()
        next_b = 1
        @info "Assigning batches to workers" nbatches=nbatches num_workers=num_workers

        if num_workers == 1
            @info "Running sequential dataset generation (no worker serialization)"
            for b in 1:nbatches
                data = generate_batch(
                    b,
                    batchsize,
                    N,
                    kind,
                    seed;
                    link_probability = link_probability,
                    cset_size = cset_size,
                    D = D,
                    cut_restriction = cut_restriction,
                    big_crystal = big_crystal,

                    mink = mink,
                    causal_diamond_boundary = causal_diamond_boundary,

                    ndistr = ndistr,

                    rdistr = rdistr,
                    genus_distr = genus_distr,
                    num_boundary_cuts_distr = num_boundary_cuts_distr,

                    lattice_distr = lattice_distr,
                    lattices = lattices,
                    segment_ratio_distr = segment_ratio_distr,
                    rotate_angle_distr = rotate_angle_distr,
                    oblique_angle_distr = oblique_angle_distr,

                    non_manifoldlikeness_distr = non_manifoldlikeness_distr,

                    layers_distr = layers_distr,
                    link_probability_distr = link_probability_distr,
                
                    connectivity_distr = connectivity_distr,
                    links_only = links_only,
                )
                pending[b] = data
                while haskey(pending, next_b)
                    data = pending[next_b]
                    b = next_b
                    delete!(pending, next_b)
                    next_b += 1

                    if !links_only
                        fout["batches/$b/csets"] = data.csets_b
                    end
                    fout["batches/$b/links"] = data.links_b

                    if kind == "minkowski_quasicrystal"
                        fout["batches/$b/trans_in"]  = data.trans_in_b
                        fout["batches/$b/trans_out"] = data.trans_out_b
                    end

                    if kind == "manifoldlike_simply_connected"
                        fout["batches/$b/r"] = data.r_b
                        fout["batches/$b/order"] = data.order_b
                    end

                    if kind == "manifoldlike_non_simply_connected"
                        fout["batches/$b/r"] = data.r_b
                        fout["batches/$b/order"] = data.order_b
                        fout["batches/$b/num_boundary_cuts"] = data.num_boundary_cuts_b
                        fout["batches/$b/genus"] = data.genus_b
                    end

                    if kind == "destroyed"
                        fout["batches/$b/r"] = data.r_b
                        fout["batches/$b/order"] = data.order_b
                        fout["batches/$b/rel_num_flips"] = data.rel_num_flips_b
                    end

                    if kind == "merged"
                        fout["batches/$b/r"] = data.r_b
                        fout["batches/$b/order"] = data.order_b
                        fout["batches/$b/rel_size_KR"] = data.rel_size_KR_b
                        fout["batches/$b/link_probability"] = data.link_probability_b
                    end

                    if kind == "grid"
                        fout["batches/$b/segment_ratio"] = data.segment_ratio_b
                        fout["batches/$b/segment_angle"] = data.segment_angle_b
                        fout["batches/$b/rotation_angle"] = data.rotation_angle_b
                        fout["batches/$b/lattice"] = data.lattice_b
                    end

                    if kind == "layered"
                        fout["batches/$b/num_layers"] = data.num_layers_b
                        fout["batches/$b/std"] = data.std_b
                    end

                    ProgressMeter.next!(p; step=length(data.links_b))
                end
            end
        else
            workers_list = Distributed.workers()
            if num_workers < 1
                error("num_workers must be >= 1")
            end
            if length(workers_list) < num_workers
                error("num_workers=$num_workers but only $(length(workers_list)) workers available")
            end
            workers_list = workers_list[1:num_workers]

            batch_map = [collect(w:num_workers:nbatches) for w in 1:num_workers]
            results = Distributed.RemoteChannel(() -> Channel{Any}(max(num_workers, nbatches)))
            worker_tasks = Distributed.Future[]
            worker_checked = Bool[]

            @info "Launching worker tasks"
            for (idx, w) in enumerate(workers_list)
                let worker_id = w, worker_batches = batch_map[idx]
                    fut = Distributed.@spawnat worker_id begin
                        try
                            for b in worker_batches
                                data = generate_batch(
                                    b,
                                    batchsize,
                                    N,
                                    kind,
                                    seed;
                                    link_probability = link_probability,
                                    cset_size = cset_size,
                                    D = D,
                                    cut_restriction = cut_restriction,
                                    big_crystal = big_crystal,

                                    mink = mink,
                                    causal_diamond_boundary = causal_diamond_boundary,

                                    ndistr = ndistr,

                                    rdistr = rdistr,
                                    genus_distr = genus_distr,
                                    num_boundary_cuts_distr = num_boundary_cuts_distr,

                                    lattice_distr = lattice_distr,
                                    lattices = lattices,
                                    segment_ratio_distr = segment_ratio_distr,
                                    rotate_angle_distr = rotate_angle_distr,
                                    oblique_angle_distr = oblique_angle_distr,

                                    non_manifoldlikeness_distr = non_manifoldlikeness_distr,

                                    layers_distr = layers_distr,
                                    link_probability_distr = link_probability_distr,
                                
                                    connectivity_distr = connectivity_distr,
                                    links_only = links_only,
                                )
                                put!(results, (:ok, b, data))
                            end
                        catch e
                            bt = catch_backtrace()
                            put!(results, (:err, Distributed.myid(), sprint((io, ex) -> showerror(io, ex, bt), e)))
                        end
                    end
                    push!(worker_tasks, fut)
                    push!(worker_checked, false)
                end
            end

            @info "Collecting and writing batches" total_batches=nbatches
            received = 0
            while received < nbatches
                if !isready(results)
                    for i in eachindex(worker_tasks)
                        worker_checked[i] && continue
                        if isready(worker_tasks[i])
                            try
                                fetch(worker_tasks[i])
                                worker_checked[i] = true
                            catch e
                                bt = catch_backtrace()
                                error(
                                    "Worker task failed before sending batch result:\n" *
                                    sprint((io, ex) -> showerror(io, ex, bt), e),
                                )
                            end
                        end
                    end
                    sleep(0.05)
                    continue
                end
                msg = take!(results)
                if msg[1] === :err
                    error("Worker $(msg[2]) failed while generating batch data:\n$(msg[3])")
                end
                _, b, data = msg
                received += 1
                pending[b] = data
                while haskey(pending, next_b)
                    data = pending[next_b]
                    b = next_b
                    delete!(pending, next_b)
                    next_b += 1

                    if !links_only
                        fout["batches/$b/csets"] = data.csets_b
                    end
                    fout["batches/$b/links"] = data.links_b

                    if kind == "minkowski_quasicrystal"
                        fout["batches/$b/trans_in"]  = data.trans_in_b
                        fout["batches/$b/trans_out"] = data.trans_out_b
                    end

                    if kind == "manifoldlike_simply_connected"
                        fout["batches/$b/r"] = data.r_b
                        fout["batches/$b/order"] = data.order_b
                    end

                    if kind == "manifoldlike_non_simply_connected"
                        fout["batches/$b/r"] = data.r_b
                        fout["batches/$b/order"] = data.order_b
                        fout["batches/$b/num_boundary_cuts"] = data.num_boundary_cuts_b
                        fout["batches/$b/genus"] = data.genus_b
                    end

                    if kind == "destroyed"
                        fout["batches/$b/r"] = data.r_b
                        fout["batches/$b/order"] = data.order_b
                        fout["batches/$b/rel_num_flips"] = data.rel_num_flips_b
                    end

                    if kind == "merged"
                        fout["batches/$b/r"] = data.r_b
                        fout["batches/$b/order"] = data.order_b
                        fout["batches/$b/rel_size_KR"] = data.rel_size_KR_b
                        fout["batches/$b/link_probability"] = data.link_probability_b
                    end

                    if kind == "grid"
                        fout["batches/$b/segment_ratio"] = data.segment_ratio_b
                        fout["batches/$b/segment_angle"] = data.segment_angle_b
                        fout["batches/$b/rotation_angle"] = data.rotation_angle_b
                        fout["batches/$b/lattice"] = data.lattice_b
                    end

                    if kind == "layered"
                        fout["batches/$b/num_layers"] = data.num_layers_b
                        fout["batches/$b/std"] = data.std_b
                    end

                    ProgressMeter.next!(p; step=length(data.links_b))
                end
            end

            for i in eachindex(worker_tasks)
                worker_checked[i] && continue
                fetch(worker_tasks[i])
            end
        end
    end
end
