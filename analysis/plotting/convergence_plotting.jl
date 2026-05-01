using CairoMakie
using Colors
using ColorSchemes
using LaTeXStrings
using Observables
using Statistics

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
    )::CairoMakie.Figure

Compute convergence of histogram standard deviations.

# Returns
- `CairoMakie.Figure` with:
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
- `BoundsError`: Raised when `bin_plot` is outside the valid bin range.
- `DomainError`: Raised when no valid bin fits are available for plotting."""
function convergence_plots_std_change( # can maybe be removed
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
)::CairoMakie.Figure

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
        Δσ_avg[j] = Statistics.mean(Δσ[j])
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

    fig = CairoMakie.Figure(size=figsize)

    ax1 = CairoMakie.Axis(
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
        CairoMakie.lines!(ax1, x[mask], y[mask];
               color = get(ColorSchemes.viridis, normN(Ns[i])))
    end

    ax1.xlabel = xlabel
    ax1.ylabel = bin_average > 1 ? "Δσ (bin-averaged)" : "Δσₖ"
    xlim !== nothing && CairoMakie.xlims!(ax1, xlim...)
    ylim1 !== nothing && CairoMakie.ylims!(ax1, ylim1...)

    apply_paper_theme!(
        double_column = false,
        magnification = magnification,
        logscale_x = true,
        logscale_y = true,
        n_Legend_columns = n_Legend_columns,
    )

    ax2 = CairoMakie.Axis(
        fig[2, 1];
        xscale = log10,
        yscale = log10,
    )
    mask = Δσ_avg .> 0
    CairoMakie.lines!(ax2, Ns[mask], Δσ_avg[mask];
           color = get.(Ref(ColorSchemes.viridis), normN.(Ns[mask])))

    ax2.xlabel = "sample size"
    ax2.ylabel = "⟨Δσ⟩"
    ylim2 !== nothing && CairoMakie.ylims!(ax2, ylim2...)

    # Add a colorbar legend for sample size N
    CairoMakie.Colorbar(fig[1:2, 2];
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
    )::CairoMakie.Figure

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
- `xlims`: Optional axis limits for the main x-axis.
- `normalize_x_axis_with_size`: If `true`, plot bin positions in units of
  `bin_index / cset_size`.
- `cset_size`: Optional causal-set size `n`, required when
  `normalize_x_axis_with_size = true`.
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
- `xlims`: Optional axis limits for the main x-axis.
- `ylims`: Optional axis limits for the main y-axis.
- `return_data`: If `true`, also return the fitted `α` values, the relative
  end-of-fit amplitudes `|A N_max^{-α}| / |σ(N_max)|`, and the normalized RMSE.
- `ylims`: Optional axis limits for the main y-axis.
- `xlims`: Optional axis limits for the main x-axis.
- `ylims`: Optional axis limits for the main y-axis.

# Returns
- `result::CairoMakie.Figure`: Output of `plot_alpha_bins` with type annotation `CairoMakie.Figure`.

# Throws
- `BoundsError`: Raised when `bin_plot` is outside the valid bin range.
- `DomainError`: Raised when no valid bin fits are available for plotting."""
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
    xlims::Union{Nothing,Tuple} = nothing,
    ylims::Union{Nothing,Tuple} = nothing,
    normalize_x_axis_with_size::Bool = false,
    cset_size::Union{Nothing,Real} = nothing,
    remove_zeros::Bool = false,
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
    return_data::Bool = false,
)

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

    if N0 === nothing && !remove_zeros
        bin_fits = fit.bin_fits
        mean_fit = fit.mean_fit
    else
        nsteps, nbins = size(σ)
        bin_fits = Vector{Union{NamedTuple,Missing}}(undef, nbins)

        for k in 1:nbins
            σn = view(σ, :, k)
            mask = isfinite.(σn)
            if remove_zeros
                mask = mask .& (σn .!= 0)
            end
            if N0 !== nothing
                mask = mask .& (ns .>= N0)
            end
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

        σmean = vec(Statistics.mean(σ; dims=2))
        mask = isfinite.(σmean)
        if remove_zeros
            mask = mask .& (σmean .!= 0)
        end
        if N0 !== nothing
            mask = mask .& (ns .>= N0)
        end
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
    rel_end_amplitudes = Float64[]
    rel_rmses = Float64[]
    bins = Int[]
    flagged = Bool[]
    bin_curves = Vector{Vector{Float64}}()

    for (k, f) in enumerate(bin_fits)
        f === missing && continue
        push!(bins, k)
        push!(αs, f.α)
        σn = view(σ, :, k)
        mask = isfinite.(σn)
        if remove_zeros
            mask = mask .& (σn .!= 0)
        end
        if N0 !== nothing
            mask = mask .& (Ns .>= N0)
        end
        σn_k = σn[mask]
        ns_k = ns[mask]
        nfit = length(σn_k)
        nmax = nfit > 0 ? ns_k[end] : NaN
        end_data = nfit > 0 ? abs(σn_k[end]) : NaN
        end_fit = nfit > 0 ? abs(f.A * nmax^(-f.α)) : NaN
        push!(rel_end_amplitudes, (nfit > 0 && end_data > 0) ? end_fit / end_data : 0.0)
        if nfit > 0
            residuals = abs.(σn_k .- f.σinf) .- f.A .* ns_k.^(-f.α)
            scale = abs(σn_k[end])
            push!(rel_rmses, scale > 0 ? sqrt(sum((residuals ./ scale).^2) / nfit) : 0.0)
        else
            push!(rel_rmses, 0.0)
        end
        if length(σn_k) < 3
            push!(flagged, true)
        else
            frac_zero = Statistics.mean(σn_k .== 0)
            push!(flagged, frac_zero > flag_zero_frac)
        end
        push!(bin_curves, collect(σn_k))
    end

    if isempty(αs)
        throw(DomainError((N0 = N0, flag_zero_frac = flag_zero_frac), "No bins with valid α fits after filtering"))
    end
    if normalize_x_axis_with_size
        if cset_size === nothing
            throw(ArgumentError("normalize_x_axis_with_size = true requires cset_size"))
        end
        if !isfinite(cset_size)
            throw(DomainError(cset_size, "cset_size must be finite when normalize_x_axis_with_size = true"))
        end
        if cset_size <= 0
            throw(DomainError(cset_size, "cset_size must be positive when normalize_x_axis_with_size = true"))
        end
    end
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
    fig = CairoMakie.Figure(size = (figsize[1], fig_height))
    ax  = CairoMakie.Axis(fig[1, 1])

    ax.xlabel = xlabel
    ax.ylabel = ylabel
    xlims !== nothing && CairoMakie.xlims!(ax, xlims...)
    ylims !== nothing && CairoMakie.ylims!(ax, ylims...)

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
    xbins = normalize_x_axis_with_size ? Float64.(xbins) ./ cset_size : xbins

    #CairoMakie.scatter!(ax, xbins, αs; markersize = 8)
    CairoMakie.lines!(ax, xbins, αs)
    colors_obs = CairoMakie.theme(:palette).color
    colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
    flag_color = colors[mod1(2, length(colors))]
    i = 1
    while i <= length(flagged)
        if flagged[i]
            j = i
            while j < length(flagged) && flagged[j+1]
                j += 1
            end
            CairoMakie.lines!(ax, xbins[i:j], αs[i:j]; color = flag_color)
            i = j + 1
        else
            i += 1
        end
    end

    if bin_plot !== nothing
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        idx = findfirst(==(bin_plot_avg), bins)
        if idx !== nothing
            CairoMakie.scatter!(ax, [xbins[idx]], [αs[idx]]; markersize = 8, color = :black)
        end
    end
    # --- reference lines ---------------------------------------------------
    CairoMakie.hlines!(ax, [0.5];
        linestyle = :dash,
        linewidth = 2 * magnification,
        color = :black,
        label = LaTeXStrings.L"\alpha = \frac{1}{2}",
    )
    CairoMakie.hlines!(ax, [0.0];
        linestyle = :solid,
        linewidth = 2 * magnification,
        color = :black,
        label = LaTeXStrings.L"\alpha = 0",
    )

    mean_color = Colors.colorant"#D12771"
    if plot_mean
        CairoMakie.hlines!(ax, [mean_fit.α];
            linestyle = :dot,
            linewidth = 2 * magnification,
            color = mean_color,
            label = LaTeXStrings.L"\alpha_{\mathrm{mean}}",
        )
    end

    if legend
        legend_kwargs = (position = legendpos,)
        legendpadding !== nothing && (legend_kwargs = merge(legend_kwargs, (padding = legendpadding,)))
        legendmargin !== nothing && (legend_kwargs = merge(legend_kwargs, (margin = legendmargin,)))
        n_Legend_columns > 1 && (legend_kwargs = merge(legend_kwargs, (nbanks = n_Legend_columns,)))
        CairoMakie.axislegend(ax; legend_kwargs...)
    end

    if ylims === nothing
        ylo = minimum(αs)
        yhi = maximum(αs)
        pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
        CairoMakie.ylims!(ax, ylo - pad, yhi + pad)
    else
        CairoMakie.ylims!(ax, ylims...)
    end

    if bin_plot !== nothing
        if !(1 <= bin_plot <= nbins_orig)
            throw(BoundsError(Base.OneTo(nbins_orig), bin_plot))
        end
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        if !(1 <= bin_plot_avg <= length(bin_fits))
            throw(BoundsError(Base.OneTo(length(bin_fits)), bin_plot_avg))
        end
        fit = bin_fits[bin_plot_avg]
        if fit === missing
            throw(DomainError(bin_plot, "bin_plot has no valid α fit"))
        end
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

        ax2 = CairoMakie.Axis(fig[2, 1])

        ax2.xlabel = "sample size N"
        ax2.ylabel = "σ(N)"
        ax2.title = LaTeXStrings.latexstring("\\mathrm{bin} = $(bin_plot),\\ A = $(A_rounded),\\ \\sigma_\\infty = $(σinf_rounded)")
        CairoMakie.xlims!(ax2, 0, maximum(ns))
        ylo = minimum(σn)
        yhi = maximum(σn)
        pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
        CairoMakie.ylims!(ax2, ylo - pad, yhi + pad)

        CairoMakie.lines!(ax2, ns, σn; color = :black)
        CairoMakie.scatter!(ax2, ns, σn; markersize = 6, color = :black)

        CairoMakie.lines!(ax2, ns, upper; linestyle = :dash, color = mean_color)
        CairoMakie.lines!(ax2, ns, lower; linestyle = :dash, color = mean_color)
        CairoMakie.hlines!(ax2, [σinf]; linestyle = :dot, color = :black)
        CairoMakie.band!(ax2, ns, lower, upper; color = (mean_color, 0.15))
    end

    As = [fit.A for fit in bin_fits if fit !== missing]
    return return_data ? (fig, αs, As, rel_end_amplitudes, rel_rmses, bin_curves) : fig
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
    )::CairoMakie.Figure

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
- `normalize_x_axis_with_size`: If `true`, plot bin positions in units of
  `bin_index / cset_size`.
- `cset_size`: Optional causal-set size `n`, required when
  `normalize_x_axis_with_size = true`.
- `xlims`: Optional axis limits for the main x-axis.
- `ylims`: Optional axis limits for the main y-axis.
- `return_data`: If `true`, also return the fitted `β` values, the relative
  end-of-fit amplitudes `|B N_max^{-β}| / |\\mu(N_max)|`, and the normalized RMSE.
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
- `result::CairoMakie.Figure`: Output of `plot_beta_bins` with type annotation `CairoMakie.Figure`.

# Throws
- `BoundsError`: Raised when `bin_plot` is outside the valid bin range.
- `DomainError`: Raised when no valid bin fits are available for plotting."""
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
    xlims::Union{Nothing,Tuple} = nothing,
    ylims::Union{Nothing,Tuple} = nothing,
    normalize_x_axis_with_size::Bool = false,
    cset_size::Union{Nothing,Real} = nothing,
    remove_zeros::Bool = false,
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
    return_data::Bool = false,
)

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
    μmean_curve = vec(Statistics.mean(μ; dims=2))

    if N0 === nothing && !remove_zeros
        bin_fits = fit.bin_fits
        mean_fit = fit.mean_fit
    else
        nsteps, nbins = size(μ)
        bin_fits = Vector{Union{NamedTuple,Missing}}(undef, nbins)

        for k in 1:nbins
            μn = view(μ, :, k)
            mask = isfinite.(μn)
            if remove_zeros
                mask = mask .& (μn .!= 0)
            end
            if N0 !== nothing
                mask = mask .& (ns .>= N0)
            end
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

        μmean = vec(Statistics.mean(μ; dims=2))
        mask = isfinite.(μmean)
        if remove_zeros
            mask = mask .& (μmean .!= 0)
        end
        if N0 !== nothing
            mask = mask .& (ns .>= N0)
        end
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
    rel_end_amplitudes = Float64[]
    rel_rmses = Float64[]
    bins = Int[]
    flagged = Bool[]
    bin_curves = Vector{Vector{Float64}}()

    for (k, f) in enumerate(bin_fits)
        f === missing && continue
        push!(bins, k)
        push!(βs, f.β)
        μn = view(μ, :, k)
        mask = isfinite.(μn)
        if remove_zeros
            mask = mask .& (μn .!= 0)
        end
        if N0 !== nothing
            mask = mask .& (Ns .>= N0)
        end
        μn_k = μn[mask]
        ns_k = ns[mask]
        nfit = length(μn_k)
        nmax = nfit > 0 ? ns_k[end] : NaN
        end_data = nfit > 0 ? abs(μn_k[end]) : NaN
        end_fit = nfit > 0 ? abs(f.B * nmax^(-f.β)) : NaN
        push!(rel_end_amplitudes, (nfit > 0 && end_data > 0) ? end_fit / end_data : 0.0)
        if nfit > 0
            residuals = abs.(μn_k .- f.μinf) .- f.B .* ns_k.^(-f.β)
            scale = abs(μn_k[end])
            push!(rel_rmses, scale > 0 ? sqrt(sum((residuals ./ scale).^2) / nfit) : 0.0)
        else
            push!(rel_rmses, 0.0)
        end
        if length(μn_k) < 3
            push!(flagged, true)
        else
            frac_zero = Statistics.mean(μn_k .== 0)
            push!(flagged, frac_zero > flag_zero_frac)
        end
        push!(bin_curves, collect(μn_k))
    end

    if isempty(βs)
        throw(DomainError((N0 = N0, flag_zero_frac = flag_zero_frac), "No bins with valid β fits after filtering"))
    end
    if normalize_x_axis_with_size
        if cset_size === nothing
            throw(ArgumentError("normalize_x_axis_with_size = true requires cset_size"))
        end
        if !isfinite(cset_size)
            throw(DomainError(cset_size, "cset_size must be finite when normalize_x_axis_with_size = true"))
        end
        if cset_size <= 0
            throw(DomainError(cset_size, "cset_size must be positive when normalize_x_axis_with_size = true"))
        end
    end
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
    fig = CairoMakie.Figure(size = (figsize[1], fig_height))
    ax  = CairoMakie.Axis(fig[1, 1])

    ax.xlabel = xlabel
    ax.ylabel = ylabel
    xlims !== nothing && CairoMakie.xlims!(ax, xlims...)
    ylims !== nothing && CairoMakie.ylims!(ax, ylims...)

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
    xbins = normalize_x_axis_with_size ? Float64.(xbins) ./ cset_size : xbins

    #CairoMakie.scatter!(ax, xbins, βs; markersize = 8)
    CairoMakie.lines!(ax, xbins, βs)
    colors_obs = CairoMakie.theme(:palette).color
    colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
    flag_color = colors[mod1(2, length(colors))]
    i = 1
    while i <= length(flagged)
        if flagged[i]
            j = i
            while j < length(flagged) && flagged[j+1]
                j += 1
            end
            CairoMakie.lines!(ax, xbins[i:j], βs[i:j]; color = flag_color)
            i = j + 1
        else
            i += 1
        end
    end

    if bin_plot !== nothing
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        idx = findfirst(==(bin_plot_avg), bins)
        if idx !== nothing
            CairoMakie.scatter!(ax, [xbins[idx]], [βs[idx]]; markersize = 8, color = :black)
        end
    end
    CairoMakie.hlines!(ax, [0.5];
        linestyle = :dash,
        linewidth = 2 * magnification,
        color = :black,
        label = LaTeXStrings.L"\beta = \frac{1}{2}",
    )
    CairoMakie.hlines!(ax, [0.0];
        linestyle = :solid,
        linewidth = 2 * magnification,
        color = :black,
        label = LaTeXStrings.L"\beta = 0",
    )

    mean_color = Colors.colorant"#D12771"
    if plot_mean
        CairoMakie.hlines!(ax, [mean_fit.β];
            linestyle = :dot,
            linewidth = 2 * magnification,
            color = mean_color,
            label = LaTeXStrings.L"\beta_{\mathrm{mean}}",
        )
    end

    if legend
        legend_kwargs = (position = legendpos,)
        legendpadding !== nothing && (legend_kwargs = merge(legend_kwargs, (padding = legendpadding,)))
        legendmargin !== nothing && (legend_kwargs = merge(legend_kwargs, (margin = legendmargin,)))
        n_Legend_columns > 1 && (legend_kwargs = merge(legend_kwargs, (nbanks = n_Legend_columns,)))
        CairoMakie.axislegend(ax; legend_kwargs...)
    end

    if ylims === nothing
        ylo = minimum(βs)
        yhi = maximum(βs)
        pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
        CairoMakie.ylims!(ax, ylo - pad, yhi + pad)
    else
        CairoMakie.ylims!(ax, ylims...)
    end

    if bin_plot !== nothing
        if !(1 <= bin_plot <= nbins_orig)
            throw(BoundsError(Base.OneTo(nbins_orig), bin_plot))
        end
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        if !(1 <= bin_plot_avg <= length(bin_fits))
            throw(BoundsError(Base.OneTo(length(bin_fits)), bin_plot_avg))
        end
        fit = bin_fits[bin_plot_avg]
        if fit === missing
            throw(DomainError(bin_plot, "bin_plot has no valid β fit"))
        end
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

        ax2 = CairoMakie.Axis(fig[2, 1])

        ax2.xlabel = LaTeXStrings.L"sample size $N$"
        ax2.ylabel = LaTeXStrings.L"\mu(N)"
        ax2.title = LaTeXStrings.latexstring("\\mathrm{bin} = $(bin_plot),\\ B = $(B_rounded),\\ \\mu_\\infty = $(μinf)")
        CairoMakie.xlims!(ax2, 0, maximum(ns))
        ylo = minimum(μn)
        yhi = maximum(μn)
        pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
        CairoMakie.ylims!(ax2, ylo - pad, yhi + pad)

        CairoMakie.lines!(ax2, ns, μn; color = :black)
        CairoMakie.scatter!(ax2, ns, μn; markersize = 6, color = :black)

        CairoMakie.lines!(ax2, ns, upper; linestyle = :dash, color = mean_color)
        CairoMakie.lines!(ax2, ns, lower; linestyle = :dash, color = mean_color)
        CairoMakie.hlines!(ax2, [μinf]; linestyle = :dot, color = :black)
        CairoMakie.band!(ax2, ns, lower, upper; color = (mean_color, 0.15))
    end

    Bs = [fit.B for fit in bin_fits if fit !== missing]
    return return_data ? (fig, βs, Bs, rel_end_amplitudes, rel_rmses, bin_curves) : fig
end

"""
    plot_alpha_beta_bins(
        X::AbstractMatrix{<:Real};
        kwargs...
    )

Plot `beta` on the left and `alpha` on the right in a single figure.

When `return_data = true`, returns
`(fig, βs, beta_rel_end_amplitudes, beta_rel_rmses, αs, alpha_rel_end_amplitudes, alpha_rel_rmses)`.
"""
function plot_alpha_beta_bins(
    X::AbstractMatrix{<:Real};
    batchsize = 1,
    bin_average = 1,
    bounds_σinf = nothing,
    bounds_α = (1e-3, 5.0),
    σinf_init = nothing,
    α_init = 0.5,
    bounds_μinf = nothing,
    bounds_β = (1e-3, 5.0),
    μinf_init = nothing,
    β_init = 0.5,
    N0 = nothing,
    xlabel = "histogram bins",
    ylabel = "β",
    ylabel_alpha = "α",
    ylabel_beta = "β",
    xlims = nothing,
    normalize_x_axis_with_size = false,
    cset_size = nothing,
    remove_zeros = false,
    ylims = nothing,
    ylims_alpha = nothing,
    ylims_beta = nothing,
    double_column = false,
    magnification = 1.0,
    n_Legend_columns = 1,
    fix_sigma_inf = true,
    flag_zero_frac = 0.05,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    legend = false,
    plot_mean = false,
    return_data::Bool = false,
)
    pairify(v) = v isa AbstractVector ? (length(v) == 2 ? (v[1], v[2]) : throw(ArgumentError("pair keyword values must have length 2"))) : (v, v)
    axispair(v) = v isa AbstractVector && length(v) == 2 ? (v[1], v[2]) : (v, v)

    batchsize_beta, batchsize_alpha = pairify(batchsize)
    bin_average_beta, bin_average_alpha = pairify(bin_average)
    bounds_σinf_beta, bounds_σinf_alpha = pairify(bounds_σinf)
    bounds_α_beta, bounds_α_alpha = pairify(bounds_α)
    σinf_init_beta, σinf_init_alpha = pairify(σinf_init)
    α_init_beta, α_init_alpha = pairify(α_init)
    bounds_μinf_beta, bounds_μinf_alpha = pairify(bounds_μinf)
    bounds_β_beta, bounds_β_alpha = pairify(bounds_β)
    μinf_init_beta, μinf_init_alpha = pairify(μinf_init)
    β_init_beta, β_init_alpha = pairify(β_init)
    N0_beta, N0_alpha = pairify(N0)
    xlabel_beta, xlabel_alpha = pairify(xlabel)
    ylabel_beta_default, ylabel_alpha_default = pairify(ylabel)
    ylabel_alpha_beta, ylabel_alpha_alpha = pairify(ylabel_alpha)
    ylabel_beta_beta, ylabel_beta_alpha = pairify(ylabel_beta)
    xlims_beta, xlims_alpha = axispair(xlims)
    normalize_x_axis_with_size_beta, normalize_x_axis_with_size_alpha = pairify(normalize_x_axis_with_size)
    cset_size_beta, cset_size_alpha = pairify(cset_size)
    remove_zeros_beta, remove_zeros_alpha = pairify(remove_zeros)
    ylims_beta_default, ylims_alpha_default = axispair(ylims)
    ylims_alpha_beta, ylims_alpha_alpha = axispair(ylims_alpha)
    ylims_beta_beta, ylims_beta_alpha = axispair(ylims_beta)
    double_column_beta, double_column_alpha = pairify(double_column)
    magnification_beta, magnification_alpha = pairify(magnification)
    n_Legend_columns_beta, n_Legend_columns_alpha = pairify(n_Legend_columns)
    fix_sigma_inf_beta, fix_sigma_inf_alpha = pairify(fix_sigma_inf)
    flag_zero_frac_beta, flag_zero_frac_alpha = pairify(flag_zero_frac)
    legendpos_beta, legendpos_alpha = pairify(legendpos)
    legendpadding_beta, legendpadding_alpha = pairify(legendpadding)
    legendmargin_beta, legendmargin_alpha = pairify(legendmargin)
    legend_beta, legend_alpha = pairify(legend)
    plot_mean_beta, plot_mean_alpha = pairify(plot_mean)

    alpha_result = plot_alpha_bins(
        X;
        batchsize = batchsize_alpha,
        bin_average = bin_average_alpha,
        bounds_σinf = bounds_σinf_alpha,
        bounds_α = bounds_α_alpha,
        σinf_init = σinf_init_alpha,
        α_init = α_init_alpha,
        N0 = N0_alpha,
        xlabel = xlabel_alpha,
        ylabel = ylabel_alpha_default,
        xlims = xlims_alpha,
        ylims = ylims_alpha_default,
        normalize_x_axis_with_size = normalize_x_axis_with_size_alpha,
        cset_size = cset_size_alpha,
        remove_zeros = remove_zeros_alpha,
        double_column = double_column_alpha,
        magnification = magnification_alpha,
        n_Legend_columns = n_Legend_columns_alpha,
        fix_sigma_inf = fix_sigma_inf_alpha,
        flag_zero_frac = flag_zero_frac_alpha,
        legendpos = legendpos_alpha,
        legendpadding = legendpadding_alpha,
        legendmargin = legendmargin_alpha,
        legend = legend_alpha,
        plot_mean = plot_mean_alpha,
        return_data = true,
    )

    beta_result = plot_beta_bins(
        X;
        batchsize = batchsize_beta,
        bin_average = bin_average_beta,
        bounds_μinf = bounds_μinf_beta,
        bounds_β = bounds_β_beta,
        μinf_init = μinf_init_beta,
        β_init = β_init_beta,
        N0 = N0_beta,
        xlabel = xlabel_beta,
        ylabel = ylabel_beta_default,
        xlims = xlims_beta,
        ylims = ylims_beta_default,
        normalize_x_axis_with_size = normalize_x_axis_with_size_beta,
        cset_size = cset_size_beta,
        remove_zeros = remove_zeros_beta,
        double_column = double_column_beta,
        magnification = magnification_beta,
        n_Legend_columns = n_Legend_columns_beta,
        fix_sigma_inf = fix_sigma_inf_beta,
        flag_zero_frac = flag_zero_frac_beta,
        legendpos = legendpos_beta,
        legendpadding = legendpadding_beta,
        legendmargin = legendmargin_beta,
        legend = legend_beta,
        plot_mean = plot_mean_beta,
        return_data = true,
    )

    _, αs, As, alpha_rel_end_amplitudes, alpha_rel_rmses, αbin_curves = alpha_result
    _, βs, Bs, beta_rel_end_amplitudes, beta_rel_rmses, βbin_curves = beta_result

    figsize = apply_paper_theme!(
        double_column = double_column_beta,
        magnification = magnification_beta,
        logscale_x = false,
        logscale_y = false,
        legendpos = legendpos_beta,
        legendpadding = legendpadding_beta,
        legendmargin = legendmargin_beta,
        n_Legend_columns = n_Legend_columns_beta,
    )

    fig = CairoMakie.Figure(size = (figsize[1] * 2, figsize[2]))
    ax_beta = CairoMakie.Axis(fig[1, 1])
    ax_alpha = CairoMakie.Axis(fig[1, 2])
    ax_alpha_left = CairoMakie.Axis(
        fig[1, 2];
        xlabelvisible = false,
        ylabelvisible = false,
        xticksvisible = false,
        xticklabelsvisible = false,
        xgridvisible = false,
        xminorgridvisible = false,
        backgroundcolor = :transparent,
    )
    x_beta = collect(eachindex(βs))
    x_alpha = collect(eachindex(αs))
    if normalize_x_axis_with_size_beta && cset_size_beta !== nothing
        x_beta = Float64.(x_beta) ./ cset_size_beta
    end
    if normalize_x_axis_with_size_alpha && cset_size_alpha !== nothing
        x_alpha = Float64.(x_alpha) ./ cset_size_alpha
    end
    CairoMakie.lines!(ax_beta, x_beta, βs)
    CairoMakie.lines!(ax_alpha, x_alpha, αs)
    ax_beta.xlabel = xlabel_beta
    ax_beta.ylabel = ylabel_beta_default
    ax_alpha.xlabel = xlabel_alpha
    ax_alpha.ylabel = ylabel_alpha_default
    ax_alpha.yaxisposition = :right
    ax_alpha.flip_ylabel = true
    ax_alpha.yticklabelalign = (:left, :center)
    ax_alpha.yticklabelrotation = 0
    ax_alpha_left.yaxisposition = :left
    ax_alpha_left.yticklabelalign = (:right, :center)
    ax_alpha_left.yticksvisible = true
    ax_alpha_left.yticklabelsvisible = false
    ax_alpha_left.rightspinevisible = false
    ax_alpha_left.topspinevisible = false
    ax_alpha_left.bottomspinevisible = false
    CairoMakie.linkxaxes!(ax_alpha, ax_alpha_left)
    CairoMakie.linkyaxes!(ax_alpha, ax_alpha_left)
    xlims_beta !== nothing && CairoMakie.xlims!(ax_beta, xlims_beta...)
    xlims_alpha !== nothing && CairoMakie.xlims!(ax_alpha, xlims_alpha...)
    xlims_alpha !== nothing && CairoMakie.xlims!(ax_alpha_left, xlims_alpha...)
    ylims_beta_default !== nothing && CairoMakie.ylims!(ax_beta, ylims_beta_default...)
    ylims_alpha_default !== nothing && CairoMakie.ylims!(ax_alpha, ylims_alpha_default...)
    ylims_alpha_default !== nothing && CairoMakie.ylims!(ax_alpha_left, ylims_alpha_default...)

    return return_data ? (fig, βs, Bs, beta_rel_end_amplitudes, beta_rel_rmses, βbin_curves, αs, As, alpha_rel_end_amplitudes, alpha_rel_rmses, αbin_curves) : fig
end

"""
    convergence_plot_matrix_data(observables; kwargs...)

Compute the convergence-plot data for link-degree, abundances, evs, and height.

Returns a 4-tuple of the corresponding `plot_alpha_beta_bins(...; return_data = true)` outputs.
"""
function convergence_plot_matrix_data(
    observables::AbstractVector{<:AbstractMatrix{<:Real}};
    deg_batchsize = 1,
    deg_bin_average = 1,
    deg_bounds_σinf = nothing,
    deg_bounds_α = (1e-3, 5.0),
    deg_σinf_init = nothing,
    deg_α_init = 0.5,
    deg_bounds_μinf = nothing,
    deg_bounds_β = (1e-3, 5.0),
    deg_μinf_init = nothing,
    deg_β_init = 0.5,
    deg_N0 = nothing,
    deg_fix_sigma_inf = true,
    deg_flag_zero_frac = 0.05,
    deg_remove_zeros = false,
    abundances_batchsize = 1,
    abundances_bin_average = 1,
    abundances_bounds_σinf = nothing,
    abundances_bounds_α = (1e-3, 5.0),
    abundances_σinf_init = nothing,
    abundances_α_init = 0.5,
    abundances_bounds_μinf = nothing,
    abundances_bounds_β = (1e-3, 5.0),
    abundances_μinf_init = nothing,
    abundances_β_init = 0.5,
    abundances_N0 = nothing,
    abundances_fix_sigma_inf = true,
    abundances_flag_zero_frac = 0.05,
    abundances_remove_zeros = false,
    evs_batchsize = 1,
    evs_bin_average = 1,
    evs_bounds_σinf = nothing,
    evs_bounds_α = (1e-3, 5.0),
    evs_σinf_init = nothing,
    evs_α_init = 0.5,
    evs_bounds_μinf = nothing,
    evs_bounds_β = (1e-3, 5.0),
    evs_μinf_init = nothing,
    evs_β_init = 0.5,
    evs_N0 = nothing,
    evs_fix_sigma_inf = true,
    evs_flag_zero_frac = 0.05,
    evs_remove_zeros = false,
    height_batchsize = 1,
    height_bin_average = 1,
    height_bounds_σinf = nothing,
    height_bounds_α = (1e-3, 5.0),
    height_σinf_init = nothing,
    height_α_init = 0.5,
    height_bounds_μinf = nothing,
    height_bounds_β = (1e-3, 5.0),
    height_μinf_init = nothing,
    height_β_init = 0.5,
    height_N0 = nothing,
    height_fix_sigma_inf = true,
    height_flag_zero_frac = 0.05,
    height_remove_zeros = false,
)
    length(observables) == 4 || throw(ArgumentError("observables must have length 4: degree, abundances, evs, height"))
    degree_data = plot_alpha_beta_bins(
        observables[1];
        batchsize = deg_batchsize,
        bin_average = deg_bin_average,
        bounds_σinf = deg_bounds_σinf,
        bounds_α = deg_bounds_α,
        σinf_init = deg_σinf_init,
        α_init = deg_α_init,
        bounds_μinf = deg_bounds_μinf,
        bounds_β = deg_bounds_β,
        μinf_init = deg_μinf_init,
        β_init = deg_β_init,
        N0 = deg_N0,
        fix_sigma_inf = deg_fix_sigma_inf,
        flag_zero_frac = deg_flag_zero_frac,
        remove_zeros = deg_remove_zeros,
        return_data = true,
    )
    abundances_data = plot_alpha_beta_bins(
        observables[2];
        batchsize = abundances_batchsize,
        bin_average = abundances_bin_average,
        bounds_σinf = abundances_bounds_σinf,
        bounds_α = abundances_bounds_α,
        σinf_init = abundances_σinf_init,
        α_init = abundances_α_init,
        bounds_μinf = abundances_bounds_μinf,
        bounds_β = abundances_bounds_β,
        μinf_init = abundances_μinf_init,
        β_init = abundances_β_init,
        N0 = abundances_N0,
        fix_sigma_inf = abundances_fix_sigma_inf,
        flag_zero_frac = abundances_flag_zero_frac,
        remove_zeros = abundances_remove_zeros,
        return_data = true,
    )
    evs_data = plot_alpha_beta_bins(
        observables[3];
        batchsize = evs_batchsize,
        bin_average = evs_bin_average,
        bounds_σinf = evs_bounds_σinf,
        bounds_α = evs_bounds_α,
        σinf_init = evs_σinf_init,
        α_init = evs_α_init,
        bounds_μinf = evs_bounds_μinf,
        bounds_β = evs_bounds_β,
        μinf_init = evs_μinf_init,
        β_init = evs_β_init,
        N0 = evs_N0,
        fix_sigma_inf = evs_fix_sigma_inf,
        flag_zero_frac = evs_flag_zero_frac,
        remove_zeros = evs_remove_zeros,
        return_data = true,
    )
    height_data = plot_alpha_beta_bins(
        observables[4];
        batchsize = height_batchsize,
        bin_average = height_bin_average,
        bounds_σinf = height_bounds_σinf,
        bounds_α = height_bounds_α,
        σinf_init = height_σinf_init,
        α_init = height_α_init,
        bounds_μinf = height_bounds_μinf,
        bounds_β = height_bounds_β,
        μinf_init = height_μinf_init,
        β_init = height_β_init,
        N0 = height_N0,
        fix_sigma_inf = height_fix_sigma_inf,
        flag_zero_frac = height_flag_zero_frac,
        remove_zeros = height_remove_zeros,
        return_data = true,
    )

    return [
        (
            name = "Link-degree distribution",
            data = degree_data,
            βs = degree_data[2],
            Bs = degree_data[3],
            beta_rel_end_amplitudes = degree_data[4],
            beta_rel_rmses = degree_data[5],
            βbin_curves = degree_data[6],
            αs = degree_data[7],
            As = degree_data[8],
            alpha_rel_end_amplitudes = degree_data[9],
            alpha_rel_rmses = degree_data[10],
            αbin_curves = degree_data[11],
        ),
        (
            name = "Interval abundances",
            data = abundances_data,
            βs = abundances_data[2],
            Bs = abundances_data[3],
            beta_rel_end_amplitudes = abundances_data[4],
            beta_rel_rmses = abundances_data[5],
            βbin_curves = abundances_data[6],
            αs = abundances_data[7],
            As = abundances_data[8],
            alpha_rel_end_amplitudes = abundances_data[9],
            alpha_rel_rmses = abundances_data[10],
            αbin_curves = abundances_data[11],
        ),
        (
            name = "Graph-Laplacian eigenvalues",
            data = evs_data,
            βs = evs_data[2],
            Bs = evs_data[3],
            beta_rel_end_amplitudes = evs_data[4],
            beta_rel_rmses = evs_data[5],
            βbin_curves = evs_data[6],
            αs = evs_data[7],
            As = evs_data[8],
            alpha_rel_end_amplitudes = evs_data[9],
            alpha_rel_rmses = evs_data[10],
            αbin_curves = evs_data[11],
        ),
        (
            name = "Height profile",
            data = height_data,
            βs = height_data[2],
            Bs = height_data[3],
            beta_rel_end_amplitudes = height_data[4],
            beta_rel_rmses = height_data[5],
            βbin_curves = height_data[6],
            αs = height_data[7],
            As = height_data[8],
            alpha_rel_end_amplitudes = height_data[9],
            alpha_rel_rmses = height_data[10],
            αbin_curves = height_data[11],
        ),
    ]
end

"""
    convergence_plot_matrix(data; kwargs...)

Create a 4-row matrix plot for link-degree, abundances, evs, and height from
precomputed convergence data.
"""
function convergence_plot_matrix(
    data::AbstractVector{<:NamedTuple};
    deg_xlabel = nothing,
    deg_ylabel = ("β", "α"),
    deg_yticks = nothing,
    deg_xlims = nothing,
    deg_ylims = nothing,
    deg_normalize_x_axis_with_size::Bool = false,
    deg_cset_size = nothing,
    deg_ylabel_beta = "β",
    deg_ylabel_alpha = "α",
    abundances_xlabel = nothing,
    abundances_ylabel = ("β", "α"),
    abundances_yticks = nothing,
    abundances_xlims = nothing,
    abundances_ylims = nothing,
    abundances_normalize_x_axis_with_size::Bool = false,
    abundances_cset_size = nothing,
    abundances_ylabel_beta = "β",
    abundances_ylabel_alpha = "α",
    evs_xlabel = nothing,
    evs_ylabel = ("β", "α"),
    evs_yticks = nothing,
    evs_xlims = nothing,
    evs_ylims = nothing,
    evs_normalize_x_axis_with_size::Bool = false,
    evs_cset_size = nothing,
    evs_ylabel_beta = "β",
    evs_ylabel_alpha = "α",
    height_xlabel = nothing,
    height_ylabel = ("β", "α"),
    height_yticks = nothing,
    height_xlims = nothing,
    height_ylims = nothing,
    height_normalize_x_axis_with_size::Bool = false,
    height_cset_size = nothing,
    height_ylabel_beta = "β",
    height_ylabel_alpha = "α",
    deg_ylims_beta = nothing,
    deg_ylims_alpha = nothing,
    abundances_ylims_beta = nothing,
    abundances_ylims_alpha = nothing,
    evs_ylims_beta = nothing,
    evs_ylims_alpha = nothing,
    height_ylims_beta = nothing,
    height_ylims_alpha = nothing,
    double_column::Bool = false,
    magnification::Real = 1.0,
    n_Legend_columns::Int = 1,
    return_data::Bool = false,
)
    figsize = apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_x = false,
        logscale_y = false,
        legendpos = :rt,
        legendpadding = nothing,
        legendmargin = nothing,
        n_Legend_columns = n_Legend_columns,
    )
    fig = CairoMakie.Figure(size = (figsize[1] * 2, 4 * figsize[2]))
    CairoMakie.rowgap!(fig.layout, 2)
    split_panel_value(v) = begin
        if v === nothing
            (nothing, nothing)
        elseif v isa Tuple || v isa AbstractVector
            if length(v) == 2 && all(x -> x isa Tuple || x isa AbstractVector, v)
                (v[1], v[2])
            else
                (v, v)
            end
        else
            (v, v)
        end
    end
    row_specs = [
        (data[1], deg_ylabel, deg_yticks, deg_ylabel_beta, deg_ylabel_alpha, deg_xlims, deg_ylims, deg_ylims_beta, deg_ylims_alpha, deg_normalize_x_axis_with_size, deg_cset_size, deg_xlabel),
        (data[2], abundances_ylabel, abundances_yticks, abundances_ylabel_beta, abundances_ylabel_alpha, abundances_xlims, abundances_ylims, abundances_ylims_beta, abundances_ylims_alpha, abundances_normalize_x_axis_with_size, abundances_cset_size, abundances_xlabel),
        (data[3], evs_ylabel, evs_yticks, evs_ylabel_beta, evs_ylabel_alpha, evs_xlims, evs_ylims, evs_ylims_beta, evs_ylims_alpha, evs_normalize_x_axis_with_size, evs_cset_size, evs_xlabel),
        (data[4], height_ylabel, height_yticks, height_ylabel_beta, height_ylabel_alpha, height_xlims, height_ylims, height_ylims_beta, height_ylims_alpha, height_normalize_x_axis_with_size, height_cset_size, height_xlabel),
    ]
    for (row, (row_data, ylabel_spec, yticks_spec, ylabel_beta, ylabel_alpha, xlim_i, ylim_spec, ylim_beta_i, ylim_alpha_i, normalize_x_axis_with_size, cset_size, xlabel_i)) in enumerate(row_specs)
        title = row_data.name
        shared_ylims = ylim_spec !== nothing && (ylim_spec isa AbstractVector || ylim_spec isa Tuple) && !(length(ylim_spec) == 2 && all(x -> x isa Tuple || x isa AbstractVector, ylim_spec))
        label_row = 2 * row - 1
        plot_row = 2 * row
        CairoMakie.Label(fig[label_row, 1:2], title; tellheight = false, halign = :center, font = :bold, fontsize = 16)
        ax_beta = CairoMakie.Axis(fig[plot_row, 1])
        ax_alpha = CairoMakie.Axis(fig[plot_row, 2])
        ax_alpha_left = CairoMakie.Axis(
            fig[plot_row, 2];
            xlabelvisible = false,
            ylabelvisible = false,
            xticksvisible = false,
            xticklabelsvisible = false,
            xgridvisible = false,
            xminorgridvisible = false,
            backgroundcolor = :transparent,
        )
        βs = row_data.βs
        αs = row_data.αs
        beta_x = collect(eachindex(βs))
        alpha_x = collect(eachindex(αs))
        if normalize_x_axis_with_size && cset_size !== nothing
            beta_x = beta_x ./ cset_size
            alpha_x = alpha_x ./ cset_size
        end
        CairoMakie.lines!(ax_beta, beta_x, βs)
        CairoMakie.lines!(ax_alpha, alpha_x, αs)
        if ylabel_spec isa AbstractString
            ax_beta.ylabel = ylabel_spec
            ax_alpha.ylabel = ylabel_spec
        elseif ylabel_spec isa AbstractVector || ylabel_spec isa Tuple
            ax_beta.ylabel = ylabel_spec[1]
            ax_alpha.ylabel = ylabel_spec[2]
        else
            ax_beta.ylabel = ylabel_beta
            ax_alpha.ylabel = ylabel_alpha
        end
        if yticks_spec !== nothing
            ax_beta.yticks = yticks_spec
            ax_alpha.yticks = shared_ylims ? yticks_spec : yticks_spec
            ax_alpha_left.yticks = shared_ylims ? yticks_spec : yticks_spec
        end
        if xlabel_i !== nothing
            ax_beta.xlabel = xlabel_i
            ax_alpha.xlabel = xlabel_i
        end
        ax_alpha.yaxisposition = :right
        ax_alpha.flip_ylabel = true
        ax_alpha.yticklabelalign = (:left, :center)
        ax_alpha.yticklabelrotation = 0
        ax_alpha.yticklabelsvisible = !shared_ylims
        ax_alpha_left.yaxisposition = :left
        ax_alpha_left.yticklabelalign = (:right, :center)
        ax_alpha_left.yticksvisible = true
        ax_alpha_left.yticklabelsvisible = false
        ax_alpha_left.rightspinevisible = false
        ax_alpha_left.topspinevisible = false
        ax_alpha_left.bottomspinevisible = false
        CairoMakie.linkxaxes!(ax_alpha, ax_alpha_left)
        CairoMakie.linkyaxes!(ax_alpha, ax_alpha_left)
        CairoMakie.rowsize!(fig.layout, label_row, CairoMakie.Fixed(14))
        xlim_i !== nothing && CairoMakie.xlims!(ax_beta, xlim_i...)
        xlim_i !== nothing && CairoMakie.xlims!(ax_alpha, xlim_i...)
        xlim_i !== nothing && CairoMakie.xlims!(ax_alpha_left, xlim_i...)
        if ylim_spec !== nothing
            ylim_beta_i, ylim_alpha_i = split_panel_value(ylim_spec)
        end
        ylim_beta_i !== nothing && CairoMakie.ylims!(ax_beta, ylim_beta_i...)
        ylim_alpha_i !== nothing && CairoMakie.ylims!(ax_alpha, ylim_alpha_i...)
        ylim_alpha_i !== nothing && CairoMakie.ylims!(ax_alpha_left, ylim_alpha_i...)
    end
    return return_data ? (fig, data) : fig
end
