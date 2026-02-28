@testsnippet setupPlotMatrix begin
    using Test
    using Random
    using Statistics
    using CairoMakie
    using LaTeXStrings
    using PlotUtils
    using Observables
    using Colors
    using Printf

    # Minimal valid inputs for hist-hist-vec-hist matrix plotting tests.
    function sample_hist_vec_matrix_data()
        h1 = Vector{Vector{Tuple{Dict{Int64,Float64},Real}}}([
            [(Dict{Int64,Float64}(1 => 1.0, 2 => 2.0), 1.0), (Dict{Int64,Float64}(1 => 2.0, 2 => 1.0), 2.0)],
        ])
        h2 = Vector{Vector{Tuple{Dict{Int64,Float64},Real}}}([
            [(Dict{Int64,Float64}(1 => 1.5, 2 => 2.5), 1.0), (Dict{Int64,Float64}(1 => 2.5, 2 => 1.5), 2.0)],
        ])
        v3 = [[([1.0, 2.0], 1.0), ([2.0, 3.0], 2.0)]]
        h4 = Vector{Vector{Tuple{Dict{Int64,Float64},Real}}}([
            [(Dict{Int64,Float64}(1 => 3.0, 2 => 1.0), 1.0), (Dict{Int64,Float64}(1 => 2.0, 2 => 2.0), 2.0)],
        ])
        return (h1, h2, v3, h4)
    end

    base_xlim() = [nothing, nothing, nothing, nothing]
    base_ylim() = [nothing, nothing, nothing, nothing]
    base_xlabel() = ["x1", "x2", "x3", "x4"]
    base_ylabel() = ["y1", "y2", "y3", "y4"]
    plot_out(name::String) = joinpath(mktempdir(), string(name, ".png"))
end

# Verifies normal helper rendering path including labels/limits/ticks and comp overlay.
@testitem "plot_matrix: plot_hist_or_vec_panel basic" setup=[setupPlotMatrix] begin
    # Ensure plotting defaults are reset for this standalone helper test.
    CausalSetZoology.apply_paper_theme!(logscale_x = false, logscale_y = false)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = identity, yscale = identity)
    data = [(1.0, [1.0, 2.0, 3.0], [0.1, 0.1, 0.1]), (2.0, [2.0, 3.0, 4.0], [0.2, 0.2, 0.2])]
    comp = ([1.5, 2.5, 3.5], [0.1, 0.1, 0.1])

    ret = CausalSetZoology.plot_hist_or_vec_panel!(
        ax,
        data,
        comp,
        (1.0, 3.0),
        (0.0, 5.0),
        "xlabel",
        "ylabel",
        [(1.0, "1"), (2.0, "2")],
        [(1.0, "1"), (4.0, "4")];
        logscale_y = false,
        invert_color_scaling = false,
        plot_std = true,
        vmin = 1.0,
        denom = 1.0,
        colormap = :viridis,
        comp_color = :black,
        comp_linewidth = 2.0,
    )

    @test ret === nothing
    @test ax.xlabel[] == "xlabel"
    @test ax.ylabel[] == "ylabel"
    @test ax.xticks[][1] == [1.0, 2.0]
    @test ax.yticks[][1] == [1.0, 4.0]
end

# Covers the log-y clipping/masking path and no-comp path.
@testitem "plot_matrix: plot_hist_or_vec_panel logscale branch" setup=[setupPlotMatrix] begin
    # Ensure plotting defaults are reset before testing explicit log-y behavior.
    CausalSetZoology.apply_paper_theme!(logscale_x = false, logscale_y = true)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = identity, yscale = log10)

    # Includes nonpositive means to exercise positive-value masking.
    data = [(1.0, [-1.0, 0.0, 0.5, 1.0], [0.1, 0.1, 0.1, 0.1])]
    ret = CausalSetZoology.plot_hist_or_vec_panel!(
        ax,
        data,
        nothing,
        nothing,
        (1e-3, 2.0),
        nothing,
        nothing,
        nothing,
        nothing;
        logscale_y = true,
        invert_color_scaling = true,
        plot_std = false,
        vmin = 1.0,
        denom = 1.0,
        colormap = :viridis,
        comp_color = :black,
        comp_linewidth = nothing,
    )

    @test ret === nothing
end

# Throws for malformed curve tuples (mean/std length mismatch).
@testitem "plot_matrix: plot_hist_or_vec_panel throws" setup=[setupPlotMatrix] begin
    # Ensure plotting defaults are reset for this standalone helper test.
    CausalSetZoology.apply_paper_theme!(logscale_x = false, logscale_y = false)
    fig = Figure()
    # Force linear axes to avoid global theme side effects from log tick setup.
    ax = Axis(fig[1, 1]; xscale = identity, yscale = identity)
    bad = [(1.0, [1.0, 2.0], [0.1])]

    @test_throws ArgumentError CausalSetZoology.plot_hist_or_vec_panel!(
        ax,
        bad,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing;
        logscale_y = false,
        invert_color_scaling = false,
        plot_std = true,
        vmin = 0.0,
        denom = 1.0,
        colormap = :viridis,
        comp_color = :black,
        comp_linewidth = nothing,
    )
end

# Verifies helper dispatch for histogram groups, vector groups, and empty groups.
@testitem "plot_matrix: avg_hist_or_vec helper" setup=[setupPlotMatrix] begin
    # Histogram/scalar group should produce one averaged curve per scalar.
    hgroup = [
        (Dict{Int64,Float64}(1 => 1.0, 2 => 3.0), 10.0),
        (Dict{Int64,Float64}(1 => 3.0, 2 => 1.0), 10.0),
        (Dict{Int64,Float64}(1 => 2.0, 2 => 2.0), 20.0),
    ]
    havg = CausalSetZoology.avg_hist_or_vec(hgroup)
    @test length(havg) == 2
    @test havg[1][1] == 10.0
    @test havg[1][2] ≈ [2.0, 2.0]
    @test havg[1][3] ≈ [1.0, 1.0]
    @test havg[2][1] == 20.0
    @test havg[2][2] ≈ [2.0, 2.0]
    @test havg[2][3] ≈ [0.0, 0.0]

    # Vector/scalar group path should route through vector averaging.
    vgroup = [
        ([1.0, 3.0], 1.0),
        ([3.0, 1.0], 1.0),
        ([2.0, 2.0], 2.0),
    ]
    vavg = CausalSetZoology.avg_hist_or_vec(vgroup)
    @test length(vavg) == 2
    @test vavg[1][1] == 1.0
    @test vavg[1][2] ≈ [2.0, 2.0]
    @test vavg[1][3] ≈ [1.0, 1.0]
    @test vavg[2][1] == 2.0
    @test vavg[2][2] ≈ [2.0, 2.0]
    @test vavg[2][3] ≈ [0.0, 0.0]

    # Empty input should return an empty standardized container.
    eavg = CausalSetZoology.avg_hist_or_vec(Any[])
    @test isempty(eavg)
end

# Verifies helper behavior for comparison groups, including no-op cases.
@testitem "plot_matrix: comp_avg_hist_or_vec helper" setup=[setupPlotMatrix] begin
    # Histogram/scalar path returns a single pooled (mean, std) comparison curve.
    hgroup = [
        (Dict{Int64,Float64}(1 => 1.0, 2 => 3.0), 10.0),
        (Dict{Int64,Float64}(1 => 3.0, 2 => 1.0), 20.0),
    ]
    hcomp = CausalSetZoology.comp_avg_hist_or_vec(hgroup)
    @test hcomp !== nothing
    @test hcomp[1] ≈ [2.0, 2.0]
    @test hcomp[2] ≈ [1.0, 1.0]

    # Vector/scalar path returns a single pooled (mean, std) comparison curve.
    vgroup = [
        ([1.0, 3.0], 10.0),
        ([3.0, 1.0], 20.0),
    ]
    vcomp = CausalSetZoology.comp_avg_hist_or_vec(vgroup)
    @test vcomp !== nothing
    @test vcomp[1] ≈ [2.0, 2.0]
    @test vcomp[2] ≈ [1.0, 1.0]

    # Empty and non-tuple groups are explicitly ignored.
    @test CausalSetZoology.comp_avg_hist_or_vec(Any[]) === nothing
    @test CausalSetZoology.comp_avg_hist_or_vec(Any[1, 2, 3]) === nothing
end

# Verifies the main matrix function returns figure + exactly four axes.
@testitem "plot_matrix: hist_hist_vec_hist basic return" setup=[setupPlotMatrix] begin
    data = sample_hist_vec_matrix_data()

    fig, axs = CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_basic");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        num_bins = 2,
        logscale_x = false,
        logscale_y = false,
        return_axis = true,
    )

    @test fig isa Figure
    @test axs isa Vector{CairoMakie.Axis}
    @test length(axs) == 4
end

# Covers axis-position options (top x-axis and right y-axis toggles).
@testitem "plot_matrix: hist_hist_vec_hist axis placement options" setup=[setupPlotMatrix] begin
    data = sample_hist_vec_matrix_data()

    _, axs_top_right = CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_axes_top_right");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        logscale_x = false,
        logscale_y = false,
        top_xaxis = true,
        right_yaxis = true,
        return_axis = true,
    )

    @test axs_top_right[1].xaxisposition[] == :top
    @test axs_top_right[2].xaxisposition[] == :top
    @test axs_top_right[2].yaxisposition[] == :right
    @test axs_top_right[4].yaxisposition[] == :right

    _, axs_bottom_left = CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_axes_bottom_left");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        logscale_x = false,
        logscale_y = false,
        top_xaxis = false,
        right_yaxis = false,
        return_axis = true,
    )

    @test axs_bottom_left[1].xaxisposition[] == :bottom
    @test axs_bottom_left[2].yaxisposition[] == :left
end

# Covers colorbar and tick customization branches.
@testitem "plot_matrix: hist_hist_vec_hist colorbar and ticks options" setup=[setupPlotMatrix] begin
    data = sample_hist_vec_matrix_data()

    fig, axs = CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_colorbar_opts");
        xlim = [(1.0, 3.0), nothing, nothing, nothing],
        ylim = [nothing, nothing, nothing, nothing],
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        xticks = [[(1.0, "1"), (2.0, "2")], nothing, nothing, nothing],
        yticks = [nothing, [(1.0, "1")], nothing, nothing],
        colorbar_side = :top,
        colorbar_label = "scalar",
        colorbar_label_pos = :top,
        colorbar_ticks = [(1.0, "1"), (2.0, "2")],
        colorbar_size = (12.0, 100.0),
        colorbar_pos = (0.5, 0.5),
        logscale_x = false,
        logscale_y = false,
        return_axis = true,
    )

    @test fig isa Figure
    @test length(axs) == 4
    @test axs[1].xticks[][1] == [1.0, 2.0]
    @test axs[2].yticks[][1] == [1.0]
end

# Covers optional comparison overlay branch.
@testitem "plot_matrix: hist_hist_vec_hist comp overlay" setup=[setupPlotMatrix] begin
    data = sample_hist_vec_matrix_data()
    comp = sample_hist_vec_matrix_data()

    fig = CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_comp_overlay");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        comp = comp,
        comp_linewidth = 1.5,
        logscale_x = false,
        logscale_y = false,
    )

    @test fig isa Figure
end

# Structural validation errors in the matrix function.
@testitem "plot_matrix: hist_hist_vec_hist structural throws" setup=[setupPlotMatrix] begin
    data = sample_hist_vec_matrix_data()

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_xlim");
        xlim = [nothing],
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        logscale_x = false,
        logscale_y = false,
    )

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_ylim");
        xlim = base_xlim(),
        ylim = [nothing],
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        logscale_x = false,
        logscale_y = false,
    )

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_xlabel");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = ["x"],
        ylabel = base_ylabel(),
        logscale_x = false,
        logscale_y = false,
    )

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_ylabel");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = ["y"],
        logscale_x = false,
        logscale_y = false,
    )

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_xticks");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        xticks = [nothing],
        logscale_x = false,
        logscale_y = false,
    )

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_yticks");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        yticks = [nothing],
        logscale_x = false,
        logscale_y = false,
    )
end

# Domain and option validation errors in the matrix function.
@testitem "plot_matrix: hist_hist_vec_hist domain and option throws" setup=[setupPlotMatrix] begin
    data = sample_hist_vec_matrix_data()

    @test_throws DomainError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_mag");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        magnification = 0.0,
        logscale_x = false,
        logscale_y = false,
    )

    @test_throws DomainError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_rowgap");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        rowgap = -1.0,
        logscale_x = false,
        logscale_y = false,
    )

    @test_throws DomainError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_colgap");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        colgap = -1.0,
        logscale_x = false,
        logscale_y = false,
    )

    @test_throws DomainError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_cb_size");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        colorbar_size = (-1.0, 1.0),
        logscale_x = false,
        logscale_y = false,
    )

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_side");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        colorbar_side = :invalid,
        logscale_x = false,
        logscale_y = false,
    )

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_hist_plot_matrix(
        data,
        plot_out("matrix_bad_label_pos");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        colorbar_label = "c",
        colorbar_label_pos = :invalid,
        logscale_x = false,
        logscale_y = false,
    )
end

# Distinguishability wrapper validation that occurs before path-loading logic.
@testitem "plot_matrix: hist_hist_vec_distinguishability structural validation" setup=[setupPlotMatrix] begin
    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
        ["a"], ["b"], :scalar, plot_out("bad_xlim");
        xlim = [nothing],
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
    )

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
        ["a"], ["b"], :scalar, plot_out("bad_ylim");
        xlim = base_xlim(),
        ylim = [nothing],
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
    )

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
        ["a"], ["b"], :scalar, plot_out("bad_xlabel");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = ["x"],
        ylabel = base_ylabel(),
    )

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
        ["a"], ["b"], :scalar, plot_out("bad_ylabel");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = ["y"],
    )

    @test_throws ArgumentError CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
        ["a"], ["b"], :scalar, plot_out("bad_symmetric");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        symmetric = true,
    )

    @test_throws DomainError CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
        ["a"], ["b"], :scalar, plot_out("bad_mag");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        magnification = 0.0,
    )

    @test_throws DomainError CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
        ["a"], ["b"], :scalar, plot_out("bad_rowgap");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        rowgap = -1.0,
    )

    @test_throws DomainError CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
        ["a"], ["b"], :scalar, plot_out("bad_colgap");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        colgap = -1.0,
    )

    @test_throws DomainError CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
        ["a"], ["b"], :scalar, plot_out("bad_cb_size");
        xlim = base_xlim(),
        ylim = base_ylim(),
        xlabel = base_xlabel(),
        ylabel = base_ylabel(),
        colorbar_size = (-1.0, 1.0),
    )
end

# Distinguishability wrapper should still fail on non-existent data after validation passes.
@testitem "plot_matrix: hist_hist_vec_distinguishability missing data paths" setup=[setupPlotMatrix] begin
    err = try
        CausalSetZoology.hist_hist_vec_distinguishability_plot_matrix(
            ["missing_path"], ["missing_path"], :scalar, plot_out("missing_data");
            xlim = base_xlim(),
            ylim = base_ylim(),
            xlabel = base_xlabel(),
            ylabel = base_ylabel(),
            symmetric = false,
        )
        nothing
    catch e
        e
    end

    # Threaded dataloading wraps file-open failures into task exceptions.
    @test err !== nothing
    @test err isa Union{CompositeException, TaskFailedException, SystemError}
end
