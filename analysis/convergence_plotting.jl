"""
   convergence_plots_std_change(
        hists::Vector{<:AbstractDict};
        batchsize::Int = 1,
        bin_average::Int = 1,
        xlim = nothing,
        ylim1 = nothing,
        ylim2 = nothing,
        xlabel::AbstractString = "histogram bins",
        n_Legend_columns::Int = 1,
        double_column::Bool = false,
        magnification::Real = 1,
    )::Figure

Compute convergence of histogram standard deviations.

# Returns
- `Figure` with:
  - top axis: `Δσₖ(N)` vs bin index (log-y), one line per sample-size step.
  - bottom axis: `⟨Δσ⟩` vs sample size `N` (log-log).
  - colorbar indicating sample size `N`.

# Arguments
- `hists`: Histogram input data.

# Keyword Arguments
- `batchsize`: Numeric control parameter for fitting/sampling resolution.
- `bin_average`: Bin selection or binning control parameter.
- `xlim`: Axis limits for plotting.
- `ylim1`: Keyword option `ylim1` controlling this method's behavior.
- `ylim2`: Keyword option `ylim2` controlling this method's behavior.
- `xlabel`: Text label shown in the plot output.
- `n_Legend_columns`: Keyword option `n_Legend_columns` controlling this method's behavior.
- `double_column`: Boolean toggle controlling output or execution behavior.
- `magnification`: Keyword option `magnification` controlling this method's behavior.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function convergence_plots_std_change(
    hists::Vector{<:AbstractDict};
    batchsize::Int = 1,
    bin_average::Int = 1,
    xlim = nothing,
    ylim1 = nothing,
    ylim2 = nothing,
    xlabel::AbstractString = "histogram bins",
    n_Legend_columns::Int = 1,
    double_column::Bool=false,
    magnification::Real=1,
)::Figure

    X = densify_hists(hists)
    nbins = size(X, 2)

    Ns, σ = compute_sigma_evolution(
        X;
        batchsize = batchsize,
        bin_average = bin_average,
    )

    nsteps, nbins_eff = size(σ)

    Δσ = Vector{Vector{Float64}}(undef, nsteps)
    Δσ_avg = zeros(Float64, nsteps)

    for j in 1:nsteps
        if j == 1
            Δσ[j] = σ[j, :]
        else
            Δσ[j] = abs.(σ[j, :] .- σ[j-1, :])
        end
        Δσ_avg[j] = mean(Δσ[j])
    end
    # Set up continuous colormap for sample size
    cmap = :viridis
    Ns_min, Ns_max = minimum(Ns), maximum(Ns)
    normN(N) = (N - Ns_min) / (Ns_max - Ns_min)

    figsize = apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_y = true,
        n_Legend_columns = n_Legend_columns,
    )

    fig = Figure(size=figsize)

    ax1 = Axis(
        fig[1, 1];
        yscale = log10,
    )

    for (i, dσ) in enumerate(Δσ)
        y = dσ
        x = if bin_average == 1
            collect(1:length(y))
        else
            centers = Vector{Float64}(undef, length(y))
            for k in 1:length(y)
                lo = (k - 1) * bin_average + 1
                hi = min(k * bin_average, nbins)
                centers[k] = (lo + hi) / 2
            end
            centers
        end
        mask = y .> 0
        lines!(ax1, x[mask], y[mask];
               color = get(ColorSchemes.viridis, normN(Ns[i])))
    end

    ax1.xlabel = xlabel
    ax1.ylabel = bin_average > 1 ? "Δσ (bin-averaged)" : "Δσₖ"
    xlim !== nothing && xlims!(ax1, xlim...)
    ylim1 !== nothing && ylims!(ax1, ylim1...)

    apply_paper_theme!(
        double_column = false,
        magnification = magnification,
        logscale_x = true,
        logscale_y = true,
        n_Legend_columns = n_Legend_columns,
    )

    ax2 = Axis(
        fig[2, 1];
        xscale = log10,
        yscale = log10,
    )
    mask = Δσ_avg .> 0
    lines!(ax2, Ns[mask], Δσ_avg[mask];
           color = get.(Ref(ColorSchemes.viridis), normN.(Ns[mask])))

    ax2.xlabel = "sample size"
    ax2.ylabel = "⟨Δσ⟩"
    ylim2 !== nothing && ylims!(ax2, ylim2...)

    # Add a colorbar legend for sample size N
    Colorbar(fig[1:2, 2];
        colormap = cmap,
        limits = (Ns_min, Ns_max),
        label = "sample size"
    )

    return fig
end

"""
    plot_alpha_bins(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
        bounds_σinf = nothing,
        bounds_α = (1e-3, 5.0),
        σinf_init::Union{Nothing,Real} = nothing,
        α_init = 0.5,
        N0::Union{Real,Nothing} = nothing,
        bin_plot::Union{Int,Nothing} = nothing,
        xlabel::AbstractString = "bin index",
        ylabel::AbstractString = "α",
        double_column::Bool = false,
        magnification::Real = 1.0,
        n_Legend_columns::Int = 1,
        fix_sigma_inf::Bool = true,
        flag_zero_frac::Real = 0.05,
        legendpos = :rt,
        legendpadding = nothing,
        legendmargin = nothing,
        legend::Bool = false,
        plot_mean::Bool = false,
    )::Figure

Plot fitted α for all (possibly bin-averaged) histogram bins, together with:
- a horizontal line at α = 1/2
- a horizontal line at α from the mean σ(n)

Bins that do not admit a fit are skipped. If `N0` is set, fits are recomputed
using only points with `N >= N0`. If `bin_plot` is set, a second panel is added
showing σ(N) and fitted envelope for that bin.

# Arguments
- `X`: Input coordinate/data values used in the computation.

# Keyword Arguments
- `batchsize`: Numeric control parameter for fitting/sampling resolution.
- `bin_average`: Bin selection or binning control parameter.
- `N0`: Keyword option `N0` controlling this method's behavior.
- `bin_plot`: Bin selection or binning control parameter.
- `xlabel`: Text label shown in the plot output.
- `ylabel`: Text label shown in the plot output.
- `double_column`: Boolean toggle controlling output or execution behavior.
- `magnification`: Keyword option `magnification` controlling this method's behavior.
- `n_Legend_columns`: Keyword option `n_Legend_columns` controlling this method's behavior.
- `fix_sigma_inf`: Keyword option `fix_sigma_inf` controlling this method's behavior.
- `flag_zero_frac`: Keyword option `flag_zero_frac` controlling this method's behavior.
- `legendpos`: Keyword option `legendpos` controlling this method's behavior.
- `legendpadding`: Keyword option `legendpadding` controlling this method's behavior.
- `legendmargin`: Keyword option `legendmargin` controlling this method's behavior.
- `legend`: Boolean toggle controlling output or execution behavior.
- `plot_mean`: Keyword option `plot_mean` controlling this method's behavior.

# Returns
- `result::Figure`: Output of `plot_alpha_bins` with type annotation `Figure`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function plot_alpha_bins(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
    bounds_σinf = nothing,
    bounds_α = (1e-3, 5.0),
    σinf_init::Union{Nothing,Real} = nothing,
    α_init = 0.5,
    N0::Union{Real,Nothing} = nothing,
    bin_plot::Union{Int,Nothing} = nothing,
    xlabel::AbstractString = "histogram bins",
    ylabel::AbstractString = "α",
    double_column::Bool = false,
    magnification::Real = 1.0,
    n_Legend_columns::Int = 1,
    fix_sigma_inf::Bool = true,
    flag_zero_frac::Real = 0.05,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    legend::Bool = false,
    plot_mean::Bool = false,
)::Figure

    # --- compute fits ------------------------------------------------------
    nbins_orig = size(X, 2)
    fit = fit_sigma_convergence(
        X;
        batchsize = batchsize,
        bin_average = bin_average,
        bounds_σinf = bounds_σinf,
        bounds_α = bounds_α,
        α_init = α_init,
        fix_sigma_inf = fix_sigma_inf,
    )

    σ = fit.σ
    Ns = fit.Ns
    ns = Float64.(Ns)

    if N0 === nothing
        bin_fits = fit.bin_fits
        mean_fit = fit.mean_fit
    else
        nsteps, nbins = size(σ)
        bin_fits = Vector{Union{NamedTuple,Missing}}(undef, nbins)

        for k in 1:nbins
            σn = view(σ, :, k)
            mask = (σn .> 0) .& (ns .>= N0)
            ns_k = ns[mask]
            σn_k = σn[mask]

            if length(σn_k) < 3
                bin_fits[k] = missing
            else
                bin_fits[k] = fit_sigma_infty_alpha(
                    σn_k,
                    ns_k;
                    σinf_init = σinf_init,
                    bounds_σinf = bounds_σinf,
                    bounds_α = bounds_α,
                    α_init = α_init,
                    fix_sigma_inf = fix_sigma_inf,
                )
            end
        end

        σmean = vec(mean(σ; dims=2))
        mask = (σmean .> 0) .& (ns .>= N0)
        ns_m = ns[mask]
        σm = σmean[mask]
        mean_fit = fit_sigma_infty_alpha(
            σm,
            ns_m;
            σinf_init = σm[end],
            bounds_σinf = bounds_σinf,
            bounds_α = bounds_α,
            α_init = α_init,
            fix_sigma_inf = fix_sigma_inf,
        )
    end
    
    # --- collect valid bin results ----------------------------------------
    αs = Float64[]
    bins = Int[]
    flagged = Bool[]

    for (k, f) in enumerate(bin_fits)
        f === missing && continue
        push!(bins, k)
        push!(αs, f.α)
        σn = view(σ, :, k)
        mask = isfinite.(σn)
        if N0 !== nothing
            mask = mask .& (Ns .>= N0)
        end
        σn_k = σn[mask]
        ns_k = ns[mask]
        if length(σn_k) < 3
            push!(flagged, true)
        else
            frac_zero = mean(σn_k .== 0)
            push!(flagged, frac_zero > flag_zero_frac)
        end
    end

    @assert !isempty(αs) "No bins with valid α fits"

    # --- theme -------------------------------------------------------------
    figsize = apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_x = false,
        logscale_y = false,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
    )

    fig_height = bin_plot === nothing ? figsize[2] : 1.6 * figsize[2]
    fig = Figure(size = (figsize[1], fig_height))
    ax  = Axis(fig[1, 1])

    ax.xlabel = xlabel
    ax.ylabel = ylabel

    # --- plot per-bin α ----------------------------------------------------
    xbins = if bin_average == 1
        bins
    else
        centers = Vector{Float64}(undef, length(bins))
        for (i, k) in enumerate(bins)
            lo = (k - 1) * bin_average + 1
            hi = min(k * bin_average, nbins_orig)
            centers[i] = (lo + hi) / 2
        end
        centers
    end

    #scatter!(ax, xbins, αs; markersize = 8)
    lines!(ax, xbins, αs)
    colors_obs = Makie.theme(:palette).color
    colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
    flag_color = colors[mod1(2, length(colors))]
    i = 1
    while i <= length(flagged)
        if flagged[i]
            j = i
            while j < length(flagged) && flagged[j+1]
                j += 1
            end
            lines!(ax, xbins[i:j], αs[i:j]; color = flag_color)
            i = j + 1
        else
            i += 1
        end
    end

    if bin_plot !== nothing
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        idx = findfirst(==(bin_plot_avg), bins)
        if idx !== nothing
            scatter!(ax, [xbins[idx]], [αs[idx]]; markersize = 8, color = :black)
        end
    end
    ylo = minimum(αs)
    yhi = maximum(αs)
    pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
    ylims!(ax, ylo - pad, yhi + pad)

    # --- reference lines ---------------------------------------------------
    hlines!(ax, [0.5];
        linestyle = :dash,
        linewidth = 2 * magnification,
        color = :black,
        label = L"\alpha = \frac{1}{2}",
    )
    hlines!(ax, [0.0];
        linestyle = :solid,
        linewidth = 2 * magnification,
        color = :black,
        label = L"\alpha = 0",
    )

    mean_color = colorant"#D12771"
    if plot_mean
        hlines!(ax, [mean_fit.α];
            linestyle = :dot,
            linewidth = 2 * magnification,
            color = mean_color,
            label = L"\alpha_{\mathrm{mean}}",
        )
    end

    if legend
        legend_kwargs = (position = legendpos,)
        legendpadding !== nothing && (legend_kwargs = merge(legend_kwargs, (padding = legendpadding,)))
        legendmargin !== nothing && (legend_kwargs = merge(legend_kwargs, (margin = legendmargin,)))
        n_Legend_columns > 1 && (legend_kwargs = merge(legend_kwargs, (nbanks = n_Legend_columns,)))
        axislegend(ax; legend_kwargs...)
    end

    if bin_plot !== nothing
        @assert 1 <= bin_plot <= nbins_orig "bin_plot out of range"
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        @assert 1 <= bin_plot_avg <= length(bin_fits) "bin_plot out of range"
        fit = bin_fits[bin_plot_avg]
        @assert fit !== missing "bin_plot has no valid fit"

        σn = view(σ, :, bin_plot_avg)
        mask = isfinite.(σn)
        if N0 !== nothing
            mask = mask .& (Ns .>= N0)
        end
        ns = Float64.(Ns[mask])
        σn = σn[mask]

        σinf = fit.σinf
        σinf_rounded = round(σinf, sigdigits = 1)
        A = fit.A
        α = fit.α
        A_rounded = round(A, sigdigits = 1)
        upper = σinf .+ A .* ns.^(-α)
        lower = σinf .- A .* ns.^(-α)

        mask = isfinite.(σn) .& isfinite.(upper) .& isfinite.(lower)
        ns = ns[mask]
        σn = σn[mask]
        upper = upper[mask]
        lower = lower[mask]

        ax2 = Axis(fig[2, 1])

        ax2.xlabel = "sample size N"
        ax2.ylabel = "σ(N)"
        ax2.title = latexstring("\\mathrm{bin} = $(bin_plot),\\ A = $(A_rounded),\\ \\sigma_\\infty = $(σinf_rounded)")
        xlims!(ax2, 0, maximum(ns))
        ylo = minimum(σn)
        yhi = maximum(σn)
        pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
        ylims!(ax2, ylo - pad, yhi + pad)

        lines!(ax2, ns, σn; color = :black)
        scatter!(ax2, ns, σn; markersize = 6, color = :black)

        lines!(ax2, ns, upper; linestyle = :dash, color = mean_color)
        lines!(ax2, ns, lower; linestyle = :dash, color = mean_color)
        hlines!(ax2, [σinf]; linestyle = :dot, color = :black)
        band!(ax2, ns, lower, upper; color = (mean_color, 0.15))
    end

    return fig
end

"""
    plot_beta_bins(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
        bounds_μinf = nothing,
        bounds_β = (1e-3, 5.0),
        μinf_init::Union{Nothing,Real} = nothing,
        β_init = 0.5,
        N0::Union{Real,Nothing} = nothing,
        bin_plot::Union{Int,Nothing} = nothing,
        xlabel::AbstractString = "bin index",
        ylabel::AbstractString = "β",
        double_column::Bool = false,
        magnification::Real = 1.0,
        n_Legend_columns::Int = 1,
        fix_sigma_inf::Bool = true,
        flag_zero_frac::Real = 0.05,
        legendpos = :rt,
        legendpadding = nothing,
        legendmargin = nothing,
        legend::Bool = false,
        plot_mean::Bool = false,
    )::Figure

Plot fitted β for all (possibly bin-averaged) histogram bins, together with:
- a horizontal line at β = 1/2
- a horizontal line at β from the mean μ(n)

Bins that do not admit a fit are skipped. If `N0` is set, fits are recomputed
using only points with `N >= N0`. If `bin_plot` is set, a second panel is added
showing μ(N) and fitted envelope for that bin.

# Arguments
- `X`: Input coordinate/data values used in the computation.

# Keyword Arguments
- `batchsize`: Numeric control parameter for fitting/sampling resolution.
- `bin_average`: Bin selection or binning control parameter.
- `N0`: Keyword option `N0` controlling this method's behavior.
- `bin_plot`: Bin selection or binning control parameter.
- `xlabel`: Text label shown in the plot output.
- `ylabel`: Text label shown in the plot output.
- `double_column`: Boolean toggle controlling output or execution behavior.
- `magnification`: Keyword option `magnification` controlling this method's behavior.
- `n_Legend_columns`: Keyword option `n_Legend_columns` controlling this method's behavior.
- `fix_sigma_inf`: Keyword option `fix_sigma_inf` controlling this method's behavior.
- `flag_zero_frac`: Keyword option `flag_zero_frac` controlling this method's behavior.
- `legendpos`: Keyword option `legendpos` controlling this method's behavior.
- `legendpadding`: Keyword option `legendpadding` controlling this method's behavior.
- `legendmargin`: Keyword option `legendmargin` controlling this method's behavior.
- `legend`: Boolean toggle controlling output or execution behavior.
- `plot_mean`: Keyword option `plot_mean` controlling this method's behavior.

# Returns
- `result::Figure`: Output of `plot_beta_bins` with type annotation `Figure`.

# Throws
- `AssertionError`: Raised when explicit input preconditions fail.
- `ErrorException`: Raised for invalid option combinations or unsupported inputs."""
function plot_beta_bins(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
    bounds_μinf = nothing,
    bounds_β = (1e-3, 5.0),
    μinf_init = nothing,
    β_init = 0.5,
    N0::Union{Real,Nothing} = nothing,
    bin_plot::Union{Int,Nothing} = nothing,
    xlabel::AbstractString = "histogram bins",
    ylabel::AbstractString = "β",
    double_column::Bool = false,
    magnification::Real = 1.0,
    n_Legend_columns::Int = 1,
    fix_sigma_inf::Bool = true,
    flag_zero_frac::Real = 0.05,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    legend::Bool = false,
    plot_mean::Bool = false,
)::Figure

    nbins_orig = size(X, 2)

    fit = fit_mu_convergence(
        X;
        batchsize = batchsize,
        bin_average = bin_average,
        bounds_μinf = bounds_μinf,
        bounds_β = bounds_β,
        μinf_init = μinf_init,
        β_init = β_init,
        fix_sigma_inf = fix_sigma_inf,
    )

    μ = fit.μ
    Ns = fit.Ns
    ns = Float64.(Ns)

    if N0 === nothing
        bin_fits = fit.bin_fits
        mean_fit = fit.mean_fit
    else
        nsteps, nbins = size(μ)
        bin_fits = Vector{Union{NamedTuple,Missing}}(undef, nbins)

        for k in 1:nbins
            μn = view(μ, :, k)
            mask = isfinite.(μn) .& (ns .>= N0)
            ns_k = ns[mask]
            μn_k = μn[mask]

            if length(μn_k) < 3
                bin_fits[k] = missing
            else
                bin_fits[k] = fit_mu_infty_beta(
                    μn_k,
                    ns_k;
                    μinf_init = μinf_init,
                    bounds_μinf = bounds_μinf,
                    bounds_β = bounds_β,
                    β_init = β_init,
                    fix_sigma_inf = fix_sigma_inf,
                )
            end
        end

        μmean = vec(mean(μ; dims=2))
        mask = isfinite.(μmean) .& (ns .>= N0)
        ns_m = ns[mask]
        μm   = μmean[mask]

        mean_fit = fit_mu_infty_beta(
            μm,
            ns_m;
            μinf_init = μm[end],
            bounds_μinf = bounds_μinf,
            bounds_β = bounds_β,
            β_init = β_init,
            fix_sigma_inf = fix_sigma_inf,
        )
    end

    βs = Float64[]
    bins = Int[]
    flagged = Bool[]

    for (k, f) in enumerate(bin_fits)
        f === missing && continue
        push!(bins, k)
        push!(βs, f.β)
        μn = view(μ, :, k)
        mask = isfinite.(μn)
        if N0 !== nothing
            mask = mask .& (Ns .>= N0)
        end
        μn_k = μn[mask]
        ns_k = ns[mask]
        if length(μn_k) < 3
            push!(flagged, true)
        else
            frac_zero = mean(μn_k .== 0)
            push!(flagged, frac_zero > flag_zero_frac)
        end
    end

    @assert !isempty(βs) "No bins with valid β fits"

    figsize = apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_x = false,
        logscale_y = false,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
    )

    fig_height = bin_plot === nothing ? figsize[2] : 1.6 * figsize[2]
    fig = Figure(size = (figsize[1], fig_height))
    ax  = Axis(fig[1, 1])

    ax.xlabel = xlabel
    ax.ylabel = ylabel

    xbins = if bin_average == 1
        bins
    else
        centers = Vector{Float64}(undef, length(bins))
        for (i, k) in enumerate(bins)
            lo = (k - 1) * bin_average + 1
            hi = min(k * bin_average, nbins_orig)
            centers[i] = (lo + hi) / 2
        end
        centers
    end

    #scatter!(ax, xbins, βs; markersize = 8)
    lines!(ax, xbins, βs)
    colors_obs = Makie.theme(:palette).color
    colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
    flag_color = colors[mod1(2, length(colors))]
    i = 1
    while i <= length(flagged)
        if flagged[i]
            j = i
            while j < length(flagged) && flagged[j+1]
                j += 1
            end
            lines!(ax, xbins[i:j], βs[i:j]; color = flag_color)
            i = j + 1
        else
            i += 1
        end
    end

    if bin_plot !== nothing
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        idx = findfirst(==(bin_plot_avg), bins)
        if idx !== nothing
            scatter!(ax, [xbins[idx]], [βs[idx]]; markersize = 8, color = :black)
        end
    end
    ylo = minimum(βs)
    yhi = maximum(βs)
    pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
    ylims!(ax, ylo - pad, yhi + pad)

    hlines!(ax, [0.5];
        linestyle = :dash,
        linewidth = 2 * magnification,
        color = :black,
        label = L"\beta = \frac{1}{2}",
    )
    hlines!(ax, [0.0];
        linestyle = :solid,
        linewidth = 2 * magnification,
        color = :black,
        label = L"\beta = 0",
    )

    mean_color = colorant"#D12771"
    if plot_mean
        hlines!(ax, [mean_fit.β];
            linestyle = :dot,
            linewidth = 2 * magnification,
            color = mean_color,
            label = L"\beta_{\mathrm{mean}}",
        )
    end

    if legend
        legend_kwargs = (position = legendpos,)
        legendpadding !== nothing && (legend_kwargs = merge(legend_kwargs, (padding = legendpadding,)))
        legendmargin !== nothing && (legend_kwargs = merge(legend_kwargs, (margin = legendmargin,)))
        n_Legend_columns > 1 && (legend_kwargs = merge(legend_kwargs, (nbanks = n_Legend_columns,)))
        axislegend(ax; legend_kwargs...)
    end

    if bin_plot !== nothing
        @assert 1 <= bin_plot <= nbins_orig "bin_plot out of range"
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        @assert 1 <= bin_plot_avg <= length(bin_fits) "bin_plot out of range"
        fit = bin_fits[bin_plot_avg]
        @assert fit !== missing "bin_plot has no valid fit"

        μn = view(μ, :, bin_plot_avg)
        mask = isfinite.(μn)
        if N0 !== nothing
            mask = mask .& (Ns .>= N0)
        end
        ns = Float64.(Ns[mask])
        μn = μn[mask]

        μinf = fit.μinf
        B = fit.B
        β = fit.β
        B_rounded = round(B, sigdigits = 1)
        upper = μinf .+ B .* ns.^(-β)
        lower = μinf .- B .* ns.^(-β)

        mask = isfinite.(μn) .& isfinite.(upper) .& isfinite.(lower)
        ns = ns[mask]
        μn = μn[mask]
        upper = upper[mask]
        lower = lower[mask]

        ax2 = Axis(fig[2, 1])

        ax2.xlabel = L"sample size $N$"
        ax2.ylabel = L"\mu(N)"
        ax2.title = latexstring("\\mathrm{bin} = $(bin_plot),\\ B = $(B_rounded),\\ \\mu_\\infty = $(μinf)")
        xlims!(ax2, 0, maximum(ns))
        ylo = minimum(μn)
        yhi = maximum(μn)
        pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
        ylims!(ax2, ylo - pad, yhi + pad)

        lines!(ax2, ns, μn; color = :black)
        scatter!(ax2, ns, μn; markersize = 6, color = :black)

        lines!(ax2, ns, upper; linestyle = :dash, color = mean_color)
        lines!(ax2, ns, lower; linestyle = :dash, color = mean_color)
        hlines!(ax2, [μinf]; linestyle = :dot, color = :black)
        band!(ax2, ns, lower, upper; color = (mean_color, 0.15))
    end

    return fig
end
