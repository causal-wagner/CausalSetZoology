@testsnippet setupPlotTheme begin
    using Test
    using CairoMakie
    using LaTeXStrings
    using Printf
    using Colors
    using Observables

    # Helper: unwrap Makie/Observables values when theme fields are stored as Observable.
    obsval(x) = x isa Observables.AbstractObservable ? Observables.to_value(x) : x
end

# Verifies internal major-tick construction returns sensible monotone log ticks.
@testitem "plot_theme: _logticks_internal basic" setup=[setupPlotTheme] begin
    ticks, labels, kind = CausalSetZoology._logticks_internal(0.1, 100.0)

    @test !isempty(ticks)
    @test length(ticks) == length(labels)
    @test all(t -> t > 0, ticks)
    @test issorted(ticks)
    @test kind in (:decades, :sparse, :dense)

    # Labels should be LaTeX labels as documented.
    @test all(l -> l isa LaTeXStrings.LaTeXString, labels)
end

# Verifies wrapper parity: `logticks` returns the major ticks/labels from internal helper.
@testitem "plot_theme: logticks wrapper parity" setup=[setupPlotTheme] begin
    ti, li, _ = CausalSetZoology._logticks_internal(0.1, 100.0)
    tw, lw = CausalSetZoology.logticks(0.1, 100.0)

    @test tw == ti
    @test lw == li
end

# Verifies minor-tick properties for a representative range.
@testitem "plot_theme: logminorticks basic" setup=[setupPlotTheme] begin
    lo, hi = 0.1, 100.0
    minors = CausalSetZoology.logminorticks(lo, hi)

    @test all(x -> lo <= x <= hi, minors)
    @test all(x -> x > 0, minors)
    @test issorted(minors)
end

# Verifies that minors do not overlap returned major ticks for typical ranges.
@testitem "plot_theme: logminorticks excludes major ticks" setup=[setupPlotTheme] begin
    for (lo, hi) in ((1.0, 10.0), (0.1, 10.0), (2.0, 200.0))
        majors, _ = CausalSetZoology.logticks(lo, hi)
        minors = CausalSetZoology.logminorticks(lo, hi)

        @test isempty(intersect(majors, minors))
        @test all(x -> lo <= x <= hi, minors)
    end
end

# Verifies Makie minor-tick integration forwards to `logminorticks`.
@testitem "plot_theme: makie minor tick integration" setup=[setupPlotTheme] begin
    lo, hi = 0.1, 10.0
    mt = Makie.get_minor_tickvalues(CausalSetZoology.logminorticks, log10, nothing, lo, hi)
    direct = CausalSetZoology.logminorticks(lo, hi)

    @test mt isa Vector{Float64}
    @test mt == direct

    # Also cover other logarithmic scale dispatch targets supported by the method.
    mt_log = Makie.get_minor_tickvalues(CausalSetZoology.logminorticks, log, nothing, lo, hi)
    mt_log2 = Makie.get_minor_tickvalues(CausalSetZoology.logminorticks, log2, nothing, lo, hi)
    @test mt_log == direct
    @test mt_log2 == direct
end

# Verifies theme sizing scales correctly with magnification and column mode.
@testitem "plot_theme: apply_paper_theme sizing" setup=[setupPlotTheme] begin
    s1 = CausalSetZoology.apply_paper_theme!(; double_column = false, magnification = 1.0, logscale_x = false, logscale_y = false)
    s2 = CausalSetZoology.apply_paper_theme!(; double_column = false, magnification = 2.0, logscale_x = false, logscale_y = false)
    sd = CausalSetZoology.apply_paper_theme!(; double_column = true, magnification = 1.0, logscale_x = false, logscale_y = false)

    @test length(s1) == 2
    @test all(>(0), s1)
    @test s2[1] ≈ 2s1[1]
    @test s2[2] ≈ 2s1[2]
    @test sd[1] > s1[1]
    @test sd[2] ≈ s1[2]
end

# Verifies selected theme fields are updated as requested.
@testitem "plot_theme: apply_paper_theme palette and axis options" setup=[setupPlotTheme] begin
    custom_xticks = [(1.0, "1"), (10.0, "10")]
    custom_yticks = [(0.1, "0.1"), (1.0, "1")]

    CausalSetZoology.apply_paper_theme!(;
        logscale_x = true,
        logscale_y = true,
        xticks = custom_xticks,
        yticks = custom_yticks,
        n_Legend_columns = 3,
        color_transparency = 0.4,
    )

    axis_theme = CairoMakie.theme(:Axis)
    legend_theme = CairoMakie.theme(:Legend)
    palette_obs = CairoMakie.theme(:palette).color
    palette = obsval(palette_obs)

    # Explicit xticks/yticks should override log tick functions.
    @test obsval(getproperty(axis_theme, :xticks)) == custom_xticks
    @test obsval(getproperty(axis_theme, :yticks)) == custom_yticks

    # Legend and palette settings should reflect kwargs.
    @test obsval(getproperty(legend_theme, :nbanks)) == 3
    @test all(c -> c[2] ≈ 0.4, palette)
end

# Verifies log tick wiring and legend style overrides when kwargs are provided.
@testitem "plot_theme: apply_paper_theme log wiring and legend overrides" setup=[setupPlotTheme] begin
    custom_padding = (1.0, 2.0, 3.0, 4.0)
    custom_margin = (5.0, 6.0, 7.0, 8.0)

    CausalSetZoology.apply_paper_theme!(;
        logscale_x = true,
        logscale_y = true,
        xticks = nothing,
        yticks = nothing,
        legendpos = :lb,
        legendpadding = custom_padding,
        legendmargin = custom_margin,
        n_Legend_columns = 2,
    )

    axis_theme = CairoMakie.theme(:Axis)
    legend_theme = CairoMakie.theme(:Legend)

    # With logscale enabled and no explicit override, log tick functions should be installed.
    @test obsval(getproperty(axis_theme, :xticks)) === CausalSetZoology.logticks
    @test obsval(getproperty(axis_theme, :yticks)) === CausalSetZoology.logticks
    @test obsval(getproperty(axis_theme, :xminorticks)) === CausalSetZoology.logminorticks
    @test obsval(getproperty(axis_theme, :yminorticks)) === CausalSetZoology.logminorticks

    # Legend options should reflect explicit overrides.
    @test obsval(getproperty(legend_theme, :position)) == :lb
    @test obsval(getproperty(legend_theme, :padding)) == custom_padding
    @test obsval(getproperty(legend_theme, :margin)) == custom_margin
    @test obsval(getproperty(legend_theme, :nbanks)) == 2
end

# Throws are separated into their own testitem.
@testitem "plot_theme: validation throws" setup=[setupPlotTheme] begin
    @test_throws DomainError CausalSetZoology._logticks_internal(0.0, 10.0)
    @test_throws DomainError CausalSetZoology._logticks_internal(-1.0, 10.0)
    @test_throws DomainError CausalSetZoology._logticks_internal(1.0, 1.0)
    @test_throws DomainError CausalSetZoology._logticks_internal(1.0, 10.0; base = 1.0)
    @test_throws DomainError CausalSetZoology.logticks(0.0, 10.0)
    @test_throws DomainError CausalSetZoology.logminorticks(0.0, 10.0)

    @test_throws DomainError CausalSetZoology.apply_paper_theme!(; color_transparency = -0.1)
    @test_throws DomainError CausalSetZoology.apply_paper_theme!(; color_transparency = 1.1)
    @test_throws DomainError CausalSetZoology.apply_paper_theme!(; magnification = 0.0)
    @test_throws DomainError CausalSetZoology.apply_paper_theme!(; n_Legend_columns = 0)
end
