@testsnippet setupHistogramPlotting begin
    using Test
    using Statistics
    using Random
    using CairoMakie
    using LaTeXStrings
    using PlotUtils
    using Observables
    using Colors

    # Provide a deterministic per-test temp output path for save wrappers.
    _hist_plot_tmpdir = mktempdir()
    @eval CausalSetZoology fig_path(name::String) = joinpath($_hist_plot_tmpdir, string(name, ".png"))

    obsval(x) = x isa Observables.Observable ? Observables.to_value(x) : x
    figure_axes(fig::Figure) = [b for b in fig.content if b isa Axis]
    figure_colorbars(fig::Figure) = [b for b in fig.content if b isa Colorbar]
    figure_legends(fig::Figure) = [b for b in fig.content if b isa Legend]
    nplots(ax::Axis) = length(ax.scene.plots)
end

# Validates helper for metadata-vector length checks.
@testitem "histogram_plotting helpers: validate series metadata lengths" setup=[setupHistogramPlotting] begin
    # Matching lengths should pass silently.
    @test CausalSetZoology.validate_series_meta_lengths(2, ["A", "B"], [:line, :scatter]) === nothing
    @test CausalSetZoology.validate_series_meta_lengths(2, nothing, nothing) === nothing
end

# Validates throw behavior for metadata-length mismatches.
@testitem "histogram_plotting helpers: validate series metadata lengths validation" setup=[setupHistogramPlotting] begin
    # Label length mismatch should throw.
    @test_throws ArgumentError CausalSetZoology.validate_series_meta_lengths(2, ["A"], nothing)

    # Plot-type length mismatch should throw.
    @test_throws ArgumentError CausalSetZoology.validate_series_meta_lengths(2, nothing, [:line])
end

# Verifies axis construction helper with labels, scales, and explicit limits.
@testitem "histogram_plotting helpers: create axis" setup=[setupHistogramPlotting] begin
    # Explicit options should propagate to axis scales and labels.
    fig, ax = CausalSetZoology.create_hist_axis(
        xlim = (1.0, 4.0),
        ylim = (0.2, 9.0),
        logscale_x = true,
        logscale_y = true,
        plotlabel = "Title",
        xlabel = "x",
        ylabel = "y",
    )

    @test fig isa Figure
    @test ax isa Axis
    @test obsval(ax.xscale) === log10
    @test obsval(ax.yscale) === log10
    @test string(obsval(ax.xlabel)) == "x"
    @test string(obsval(ax.ylabel)) == "y"
    @test string(obsval(ax.title)) == "Title"
    @test length(figure_axes(fig)) >= 1

    # Defaults should still produce an axis with default labels.
    _, ax_default = CausalSetZoology.create_hist_axis()
    @test string(obsval(ax_default.ylabel)) == "count"
end

# Checks log-epsilon helper in all branches.
@testitem "histogram_plotting helpers: compute log eps" setup=[setupHistogramPlotting] begin
    # Non-log mode returns sentinel -Inf.
    @test CausalSetZoology.compute_log_eps_from_means([[1.0, 2.0]], nothing, false) == -Inf

    # Explicit y-limits drive epsilon directly.
    @test CausalSetZoology.compute_log_eps_from_means([[1.0, 2.0]], (0.5, 10.0), true) == 5e-4

    # Automatic epsilon uses minimum positive mean value.
    eps = CausalSetZoology.compute_log_eps_from_means([[0.0, 2.0], [3.0, 5.0]], nothing, true)
    @test eps == 2e-3
end

# Validates log-epsilon throw path with no positive means.
@testitem "histogram_plotting helpers: compute log eps validation" setup=[setupHistogramPlotting] begin
    # Log plotting requires at least one positive value.
    @test_throws DomainError CausalSetZoology.compute_log_eps_from_means([[0.0, -1.0]], nothing, true)
end

# Verifies series-preparation helper for linear and log-y modes.
@testitem "histogram_plotting helpers: prepare plot series" setup=[setupHistogramPlotting] begin
    mean = [0.0, 2.0, -1.0, 4.0]
    std = [0.5, 0.2, 0.1, 0.3]

    # Non-log mode should keep all bins and raw bands.
    x_all, m_all, ylo_all, yhi_all = CausalSetZoology.prepare_plot_series(mean, std, false, -Inf)
    @test x_all == [1, 2, 3, 4]
    @test m_all == mean
    @test ylo_all == mean .- std
    @test yhi_all == mean .+ std

    # Log mode should drop non-positive means and clamp lower/upper to eps.
    x_log, m_log, ylo_log, yhi_log = CausalSetZoology.prepare_plot_series(mean, std, true, 1e-4)
    @test x_log == [2, 4]
    @test m_log == [2.0, 4.0]
    @test all(>=(1e-4), ylo_log)
    @test all(>=(1e-4), yhi_log)
end

# Validates series-preparation throw path for mismatched mean/std lengths.
@testitem "histogram_plotting helpers: prepare plot series validation" setup=[setupHistogramPlotting] begin
    # Mean/std vectors must have equal length.
    @test_throws ArgumentError CausalSetZoology.prepare_plot_series([1.0, 2.0], [0.1], false, -Inf)
end

# Verifies draw helper for both line and scatter rendering branches.
@testitem "histogram_plotting helpers: draw series variants" setup=[setupHistogramPlotting] begin
    fig, ax = CausalSetZoology.create_hist_axis()
    x = [1, 2, 3]
    mean = [1.0, 2.0, 3.0]
    ylo = [0.8, 1.8, 2.8]
    yhi = [1.2, 2.2, 3.2]

    # Line with std draws both band and line.
    n0 = nplots(ax)
    CausalSetZoology.draw_series!(ax, x, mean, ylo, yhi, :blue, :line; plot_std = true, label = "L")
    @test nplots(ax) >= n0 + 2

    # Scatter without std adds only scatter plot.
    n1 = nplots(ax)
    CausalSetZoology.draw_series!(ax, x, mean, ylo, yhi, :red, :scatter; plot_std = false, markersize = 10)
    @test nplots(ax) >= n1 + 1

    # Scatter with std should add scatter + errorbars.
    n2 = nplots(ax)
    CausalSetZoology.draw_series!(ax, x, mean, ylo, yhi, :green, :scatter; plot_std = true)
    @test nplots(ax) >= n2 + 2
end

# Validates draw helper throw path for unsupported plot type.
@testitem "histogram_plotting helpers: draw series validation" setup=[setupHistogramPlotting] begin
    fig, ax = CausalSetZoology.create_hist_axis()
    x = [1, 2]
    mean = [1.0, 2.0]
    ylo = [0.9, 1.9]
    yhi = [1.1, 2.1]

    # Unsupported plot type symbol should throw.
    @test_throws ArgumentError CausalSetZoology.draw_series!(ax, x, mean, ylo, yhi, :black, :bad)
end

# Verifies legend helper behavior with and without labels.
@testitem "histogram_plotting helpers: add legend" setup=[setupHistogramPlotting] begin
    fig, ax = CausalSetZoology.create_hist_axis()
    x = [1, 2, 3]
    mean = [1.0, 2.0, 3.0]

    # Without labels, helper should be a no-op.
    @test CausalSetZoology.maybe_add_legend!(ax, nothing, :rt, nothing, nothing, 1) === nothing
    @test isempty(figure_legends(fig))

    # With labels and labeled plots, helper should add a legend.
    CausalSetZoology.draw_series!(ax, x, mean, mean .- 0.1, mean .+ 0.1, :blue, :line; label = "A")
    CausalSetZoology.maybe_add_legend!(ax, ["A"], :rt, (10, 8, 8, 8), (5, 5, 5, 5), 1)
    @test length(figure_legends(fig)) >= 1
end

# Verifies save helper for both return modes and file creation.
@testitem "histogram_plotting helpers: save plot result" setup=[setupHistogramPlotting] begin
    # Plain-figure return path.
    fig1, _ = CausalSetZoology.create_hist_axis()
    name1 = "hist_plot_helper_plain_$(rand(1:10^9))"
    out1 = CausalSetZoology.save_plot_result(fig1, name1, false)
    @test out1 === fig1
    @test isfile(CausalSetZoology.fig_path(name1))

    # (fig, ax) return path.
    fig2, ax2 = CausalSetZoology.create_hist_axis()
    name2 = "hist_plot_helper_axis_$(rand(1:10^9))"
    out2 = CausalSetZoology.save_plot_result((fig2, ax2), name2, true)
    @test out2[1] === fig2
    @test out2[2] === ax2
    @test isfile(CausalSetZoology.fig_path(name2))
end

# Verifies plain-vector coercion helper for valid, invalid, and empty input.
@testitem "histogram_plotting helpers: coerce plain vector groups" setup=[setupHistogramPlotting] begin
    # Valid nested vectors should be coerced preserving grouping and lengths.
    inp = Any[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]
    out = CausalSetZoology.coerce_plain_vector_groups(inp)
    @test out !== nothing
    @test length(out) == 2
    @test length(out[1]) == 2
    @test out[1][1] == [1.0, 2.0]

    # Empty input should return typed empty groups.
    out_empty = CausalSetZoology.coerce_plain_vector_groups(Any[])
    @test out_empty !== nothing
    @test isempty(out_empty)

    # Invalid sample shape should return nothing.
    @test CausalSetZoology.coerce_plain_vector_groups(Any[[[1.0, 2.0], 7]]) === nothing
end

# Verifies scalar-vector coercion helper for valid, invalid, and empty input.
@testitem "histogram_plotting helpers: coerce scalar vector groups" setup=[setupHistogramPlotting] begin
    # Valid scalar-tagged vectors should be coerced with scalar tags preserved.
    inp = Any[[([1.0, 2.0], 1.0), ([3.0, 4.0], 2.0)]]
    out = CausalSetZoology.coerce_scalar_vector_groups(inp)
    @test out !== nothing
    @test length(out) == 1
    @test out[1][1][1] == [1.0, 2.0]
    @test out[1][1][2] == 1.0

    # Empty input should return typed empty groups.
    out_empty = CausalSetZoology.coerce_scalar_vector_groups(Any[])
    @test out_empty !== nothing
    @test isempty(out_empty)

    # Invalid tuple shape and invalid scalar type should return nothing.
    @test CausalSetZoology.coerce_scalar_vector_groups(Any[[(1.0, 2.0, 3.0)]]) === nothing
    @test CausalSetZoology.coerce_scalar_vector_groups(Any[[([1.0, 2.0], "bad")]]) === nothing
end

# Verifies primary plotting API (plain overload) with metadata, scales, and legends.
@testitem "histogram_plotting: plot mean/std plain overload" setup=[setupHistogramPlotting] begin
    d_plain = [
        ([1.0, 2.0, 3.0], [0.1, 0.2, 0.1]),
        ([1.5, 2.2, 2.8], [0.1, 0.1, 0.2]),
    ]

    # Mixed line/scatter rendering with labels and log scales.
    fig, ax = CausalSetZoology.plot_mean_histograms_with_std(
        d_plain;
        return_axis = true,
        plot_types = [:line, :scatter],
        hist_labels = ["A", "B"],
        xlabel = "bin",
        ylabel = "count",
        logscale_x = true,
        logscale_y = true,
        linewidth = 2,
        markersize = 8,
    )

    @test fig isa Figure
    @test ax isa Axis
    @test string(obsval(ax.xlabel)) == "bin"
    @test string(obsval(ax.ylabel)) == "count"
    @test obsval(ax.xscale) === log10
    @test obsval(ax.yscale) === log10
    @test nplots(ax) >= 4
    @test length(figure_legends(fig)) >= 1
end

# Validates plain-overload throw behavior.
@testitem "histogram_plotting: plot mean/std plain overload validation" setup=[setupHistogramPlotting] begin
    d_plain = [
        ([1.0, 2.0, 3.0], [0.1, 0.2, 0.1]),
        ([1.5, 2.2, 2.8], [0.1, 0.1, 0.2]),
    ]

    # Invalid plot type and metadata mismatches should throw.
    @test_throws ArgumentError CausalSetZoology.plot_mean_histograms_with_std(d_plain; plot_types = [:bad, :line])
    @test_throws ArgumentError CausalSetZoology.plot_mean_histograms_with_std(d_plain; hist_labels = ["only_one"])
    @test_throws ArgumentError CausalSetZoology.plot_mean_histograms_with_std(d_plain; plot_types = [:line])

    # Logscale y requires positive means.
    d_nonpositive = [([0.0, -1.0], [0.1, 0.1])]
    @test_throws DomainError CausalSetZoology.plot_mean_histograms_with_std(d_nonpositive; logscale_y = true)
end

# Verifies scalar overload behavior including colorbar, comp overlay, and options.
@testitem "histogram_plotting: plot mean/std scalar overload" setup=[setupHistogramPlotting] begin
    d_scalar = [
        (1.0, [1.0, 2.0], [0.1, 0.2]),
        (2.0, [2.0, 3.0], [0.2, 0.3]),
    ]

    # Scalar mode should add a colorbar and support comparison bands.
    fig, ax = CausalSetZoology.plot_mean_histograms_with_std(
        d_scalar;
        return_axis = true,
        colorbar_label = "s",
        colorbar_ticks = [(1.0, "low"), (2.0, "high")],
        comp = ([1.4, 2.6], [0.1, 0.1]),
    )
    @test fig isa Figure
    @test ax isa Axis
    @test length(figure_colorbars(fig)) >= 1
    @test nplots(ax) >= 5
    @test string(obsval(first(figure_colorbars(fig)).label)) == "s"

    # plot_std=false should reduce plotted primitives.
    _, ax_std = CausalSetZoology.plot_mean_histograms_with_std(d_scalar; return_axis = true, plot_std = true)
    _, ax_no_std = CausalSetZoology.plot_mean_histograms_with_std(d_scalar; return_axis = true, plot_std = false)
    @test nplots(ax_std) > nplots(ax_no_std)

    # Inverted color scaling and floating colorbar position/size should run.
    fig_inv, _ = CausalSetZoology.plot_mean_histograms_with_std(
        d_scalar;
        return_axis = true,
        invert_color_scaling = true,
        colorbar_pos = (0.1, 0.9),
        colorbar_size = (20, 120),
    )
    @test length(figure_colorbars(fig_inv)) >= 1
end

# Validates scalar-overload throw behavior.
@testitem "histogram_plotting: plot mean/std scalar overload validation" setup=[setupHistogramPlotting] begin
    d_scalar = [
        (1.0, [1.0, 2.0], [0.1, 0.2]),
        (2.0, [2.0, 3.0], [0.2, 0.3]),
    ]

    # Invalid comp tuple dimensions should throw.
    @test_throws ArgumentError CausalSetZoology.plot_mean_histograms_with_std(d_scalar; comp = ([1.0, 2.0], [0.1]))

    # Invalid scalar tuple shape should throw once it reaches the scalar overload.
    @test_throws ArgumentError CausalSetZoology.plot_mean_histograms_with_std(Tuple[(1.0, [1.0, 2.0])])

    # Logscale y requires at least one positive mean.
    d_scalar_nonpositive = [(1.0, [0.0, -1.0], [0.1, 0.1])]
    @test_throws DomainError CausalSetZoology.plot_mean_histograms_with_std(d_scalar_nonpositive; logscale_y = true)
end

# Verifies plain histogram wrapper including save and default log scales.
@testitem "histogram_plotting: plot and save hists plain wrapper" setup=[setupHistogramPlotting] begin
    hplain = [[Dict(1 => 1.0, 2 => 2.0), Dict(1 => 2.0, 2 => 1.0)]]
    name = "h_plain_$(rand(1:10^9))"

    fig, ax = CausalSetZoology.plot_and_save_hists(hplain, name; return_axis = true)
    @test fig isa Figure
    @test ax isa Axis
    @test obsval(ax.xscale) === log10
    @test obsval(ax.yscale) === log10
    @test isfile(CausalSetZoology.fig_path(name))
end

# Verifies scalar histogram wrapper including colorbar/comp options.
@testitem "histogram_plotting: plot and save hists scalar wrapper" setup=[setupHistogramPlotting] begin
    hscalar = Vector{Vector{Tuple{Dict{Int64,Float64},Real}}}([
        [(Dict{Int64,Float64}(1 => 1.0, 2 => 2.0), 1.0), (Dict{Int64,Float64}(1 => 2.0, 2 => 1.0), 2.0)],
    ])
    # Comparison input for scalar wrapper is plain histogram samples (no scalar tags).
    comp = [[Dict{Int64,Float64}(1 => 1.5, 2 => 1.5), Dict{Int64,Float64}(1 => 1.0, 2 => 2.0)]]
    name = "h_scalar_$(rand(1:10^9))"

    fig, ax = CausalSetZoology.plot_and_save_hists(
        hscalar,
        name;
        num_bins = 2,
        return_axis = true,
        colorbar_label = "scalar",
        comp = comp,
        comp_linewidth = 3,
    )

    @test fig isa Figure
    @test ax isa Axis
    @test length(figure_colorbars(fig)) >= 1
    @test isfile(CausalSetZoology.fig_path(name))
end

# Verifies vector wrapper dispatcher for plain vectors.
@testitem "histogram_plotting: plot and save vectors dispatcher plain" setup=[setupHistogramPlotting] begin
    vplain = [[[1.0, 2.0], [2.0, 3.0]]]
    name = "v_plain_dispatch_$(rand(1:10^9))"

    fig, ax = CausalSetZoology.plot_and_save_vectors(vplain, name; return_axis = true)
    @test fig isa Figure
    @test ax isa Axis
    @test obsval(ax.xscale) === log10
    @test obsval(ax.yscale) === log10
    @test isfile(CausalSetZoology.fig_path(name))
end

# Verifies vector wrapper dispatcher for scalar-tagged vectors.
@testitem "histogram_plotting: plot and save vectors dispatcher scalar" setup=[setupHistogramPlotting] begin
    vscalar = [[([1.0, 2.0], 1.0), ([2.0, 3.0], 2.0), ([1.5, 2.5], 1.0)]]
    name = "v_scalar_dispatch_$(rand(1:10^9))"

    fig, ax = CausalSetZoology.plot_and_save_vectors(
        vscalar,
        name;
        num_bins = 2,
        return_axis = true,
        colorbar_label = "scalar",
        # Comparison input for scalar-vector wrapper is plain vectors, grouped one level.
        comp = [[[1.2, 2.2], [1.8, 2.8]]],
    )

    @test fig isa Figure
    @test ax isa Axis
    @test length(figure_colorbars(fig)) >= 1
    @test isfile(CausalSetZoology.fig_path(name))
end

# Validates vector dispatcher throw path for unsupported input format.
@testitem "histogram_plotting: plot and save vectors dispatcher validation" setup=[setupHistogramPlotting] begin
    # Non-grouped numeric vector is unsupported.
    @test_throws ArgumentError CausalSetZoology.plot_and_save_vectors(Any[1, 2, 3], "bad")
end

# Verifies direct plain-vector wrapper function.
@testitem "histogram_plotting: plot and save vectors plain wrapper" setup=[setupHistogramPlotting] begin
    vplain = Vector{Vector{AbstractVector}}([[[1.0, 2.0], [2.0, 3.0]]])
    name = "v_plain_direct_$(rand(1:10^9))"

    fig, ax = CausalSetZoology.plot_and_save_vectors_plain(vplain, name; return_axis = true)
    @test fig isa Figure
    @test ax isa Axis
    @test isfile(CausalSetZoology.fig_path(name))
end

# Verifies direct scalar-vector wrapper function.
@testitem "histogram_plotting: plot and save vectors scalar wrapper" setup=[setupHistogramPlotting] begin
    vscalar = Vector{Vector{Tuple{AbstractVector,Real}}}([
        [([1.0, 2.0], 1.0), ([2.0, 3.0], 2.0), ([1.5, 2.5], 1.0)],
    ])
    name = "v_scalar_direct_$(rand(1:10^9))"

    fig, ax = CausalSetZoology.plot_and_save_vectors_scalar(
        vscalar,
        name;
        num_bins = 2,
        return_axis = true,
        colorbar_label = "scalar",
        # Comparison input for scalar-vector wrapper is plain vectors, grouped one level.
        comp = [[[1.2, 2.2], [1.8, 2.8]]],
    )

    @test fig isa Figure
    @test ax isa Axis
    @test length(figure_colorbars(fig)) >= 1
    @test isfile(CausalSetZoology.fig_path(name))
end
