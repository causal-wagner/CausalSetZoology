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
)::NamedTuple

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
            sprinkling = CausalSets.generate_sprinkling(mink, causal_diamond_boundary, cset_size_i)
            cset = CausalSets.BitArrayCauset(mink, sprinkling)

        elseif kind == "minkowski_quasicrystal"
            if big_crystal === nothing
                throw(ArgumentError("big_crystal must be provided for kind = \"minkowski_quasicrystal\""))
            end
            ϵ = sqrt(cset_size_i / length(big_crystal[1]))
            trans_distr = Distributions.Uniform(ϵ, 1 - ϵ)
            αin = rand(rng, trans_distr)
            αout = rand(rng, trans_distr)
            cset = QuantumGrav.create_Minkowski_quasicrystal_cset(
                cset_size_i,
                (αin, αout);
                crystal = big_crystal,
                exact_size = true,
                deviation_from_mean_size = .1,
                max_iter = 100,
            )
            push!(trans_in_b, αin)
            push!(trans_out_b, αout)

        elseif kind == "manifoldlike_simply_connected"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            cset, _, __ = QuantumGrav.make_polynomial_manifold_cset(cset_size_i, rng, order, Float64(r); d = D === nothing ? 2 : D)
            push!(r_b, r)
            push!(order_b, order)

        elseif kind == "manifoldlike_non_simply_connected"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            num_boundary_cuts = cut_restriction == "free_cuts" ? 0 : rand(rng, num_boundary_cuts_distr)
            genus = cut_restriction == "boundary_cuts" ? 0 : rand(rng, genus_distr)
            cset, _, _, _  = QuantumGrav.make_polynomial_manifold_cset_with_nontrivial_topology(cset_size_i, num_boundary_cuts, genus, rng, order, r)
            push!(r_b, r)
            push!(order_b, order)
            push!(num_boundary_cuts_b, num_boundary_cuts)
            push!(genus_b, genus)

        elseif kind == "destroyed"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            num_flips = Int64(ceil(cset_size_i * (cset_size_i - 1) /2 * rand(rng, non_manifoldlikeness_distr)))
            cset, _, _ = QuantumGrav.destroy_manifold_cset(cset_size_i, num_flips, rng, order, r)
            push!(r_b, r)
            push!(order_b, order)
            push!(rel_num_flips_b, num_flips / (cset_size_i * (cset_size_i - 1) / 2 ))

        elseif kind == "merged"
            r = rand(rng, rdistr)
            order = Int(ceil(2 * log(cset_size_i) / log(r) + 1))
            link_probability_value = link_probability === nothing ? rand(rng, link_probability_distr) : link_probability
            n2_rel = rand(rng, non_manifoldlikeness_distr)
            cset, _, _ = QuantumGrav.insert_KR_into_manifoldlike(cset_size_i, order, r, link_probability_value; rng = rng, n2_rel = n2_rel)
            push!(r_b, r)
            push!(order_b, order)
            push!(rel_size_KR_b, n2_rel)
            push!(link_probability_b, link_probability_value)

        elseif kind == "grid"
            lattice = lattices[rand(rng, lattice_distr)]
            segment_ratio = rand(rng, segment_ratio_distr)
            rotate_angle = rand(rng, rotate_angle_distr)
            oblique_angle = rand(rng, oblique_angle_distr)
            cset, _, _ = QuantumGrav.create_grid_causet_in_boundary_2D(
                cset_size_i,
                lattice,
                CausalSets.CausalDiamondBoundary{2}(1.0),
                CausalSets.MinkowskiManifold{2}();
                b = segment_ratio,
                gamma_deg = oblique_angle,
                rotate_deg = rotate_angle
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
            cset = cset_try

        elseif kind == "layered"
            num_layers = rand(rng, layers_distr)
            link_probability = rand(rng, link_probability_distr)
            std_distr = Distributions.Uniform(0., Float64(cset_size_i / (2 * num_layers)))
            std = rand(rng, std_distr)
            cset, n_per_layer = QuantumGrav.create_random_layered_causet(cset_size_i, num_layers; p = link_probability, rng = rng, standard_deviation = std)
            push!(num_layers_b, num_layers)
            push!(std_b, std)
        end

        links = SparseLinksCauset(cset)

        push!(csets_b, cset)
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

)::Nothing

    JLD2.jldopen(out_path, "w") do fout
        fout["meta/batchsize"] = batchsize
        fout["meta/nbatches"]  = nbatches
        fout["meta/N"]         = N
        fout["meta/config"]    = config

        @info "Assigning batches to workers" nbatches=nbatches num_workers=num_workers
        workers_list = Distributed.workers()
        if num_workers < 1
            error("num_workers must be >= 1")
        end
        if length(workers_list) < num_workers
            error("num_workers=$num_workers but only $(length(workers_list)) workers available")
        end
        workers_list = workers_list[1:num_workers]

        batch_map = [collect(w:num_workers:nbatches) for w in 1:num_workers]
        results = Distributed.RemoteChannel(() -> Channel{Tuple{Int,Any}}(num_workers))

        @info "Launching worker tasks"
        for (idx, w) in enumerate(workers_list)
            Distributed.@spawnat w begin
                for b in batch_map[idx]
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
                    )
                    put!(results, (b, data))
                end
            end
        end

        p = ProgressMeter.Progress(N; desc = "Creating causal sets")
        pending = Dict{Int,Any}()
        next_b = 1
        @info "Collecting and writing batches" total_batches=nbatches
        for _ in 1:nbatches
            b, data = take!(results)
            pending[b] = data
            while haskey(pending, next_b)
                data = pending[next_b]
                b = next_b
                delete!(pending, next_b)
                next_b += 1

            fout["batches/$b/csets"] = data.csets_b
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

                ProgressMeter.next!(p; step=length(data.csets_b))
            end
        end
    end
end
