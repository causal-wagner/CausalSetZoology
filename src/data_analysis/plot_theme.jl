"""
    _logticks_internal(lo::Real, hi::Real; base::Real = 10.0)

Internal helper for logarithmic tick construction.

See the primary public method `logticks(lo, hi; base)` for behavior. This helper
returns `(major_ticks, major_labels, kind)` where `kind` describes the selected
major-tick pattern.

# Arguments
- `lo`: Lower interval bound (must be positive).
- `hi`: Upper interval bound (must satisfy `hi > lo`).

# Keyword Arguments
- `base`: Logarithm base used for tick construction (`base > 1`).

# Returns
- `ticks, labels, kind`: Major ticks, matching labels, and selected density kind.

# Throws
- `DomainError`: If interval or base constraints are violated."""
function _logticks_internal(lo::Real, hi::Real; base::Real = 10.0)
    lo <= 0 && throw(DomainError(lo, "lo must be > 0"))
    hi <= lo && throw(DomainError((lo, hi), "hi must be > lo"))
    base <= 1 && throw(DomainError(base, "base must be > 1"))

    logb(x) = log(x) / log(base)

    pmin = floor(Int, logb(lo))
    pmax = ceil(Int,  logb(hi))

    # helper for LaTeX labels
    latex_label(v) = begin
        if v < 1e-3 || v ≥ 1e4
            k = round(Int, logb(v))
            LaTeXStrings.LaTeXString("\$10^{$k}\$")
        elseif v ≥ 1
            # integers like 1, 10, 100, 1000 without scientific notation
            LaTeXStrings.LaTeXString("\$$(Int(round(v)))\$")
        else
            # numbers in (1e-3, 1): fixed-point, no exponent
            LaTeXStrings.LaTeXString(Printf.@sprintf("\$%.3f\$", v))
        end
    end

    # candidate generators for major ticks
    candidates = [
        (:decades, p -> [base^Float64(p)]),
        (:sparse,  p -> [1*base^Float64(p), 5*base^Float64(p)]),
        (:dense,   p -> [m*base^Float64(p) for m in 1:9]),
    ]

    preferred_counts = (5, 4, 6, 3)

    best_ticks = Float64[]
    best_labels = LaTeXStrings.LaTeXString[]
    best_kind = :decades

    for target in preferred_counts
        for (kind, gen) in candidates
            ticks = Float64[]
            for p in pmin:pmax
                for v in gen(p)
                    lo ≤ v ≤ hi && push!(ticks, v)
                end
            end
            sort!(ticks)

            if length(ticks) == target
                labels = latex_label.(ticks)
                return ticks, labels, kind
            end

            if isempty(best_ticks) && 3 ≤ length(ticks) ≤ 6
                best_ticks = ticks
                best_labels = latex_label.(ticks)
                best_kind = kind
            end
        end
    end

    if !isempty(best_ticks)
        return best_ticks, best_labels, best_kind
    end

    # fallback: decades subset
    ticks = [base^Float64(p) for p in pmin:pmax if lo ≤ base^Float64(p) ≤ hi]
    labels = latex_label.(ticks)

    return ticks, labels, :decades
end

"""
    logticks(lo::Real, hi::Real; base::Real = 10.0)

Return logarithmic major ticks and labels for the interval `[lo, hi]`.

# Rules
- `lo > 0` is required.
- Major ticks are chosen automatically to target 3-6 ticks, with preference
  order `5 -> 4 -> 6 -> 3`.
- Candidate major patterns:
  - decades: `base^k`
  - sparse: `{1, 5} * base^k`
  - dense: `{1, ..., 9} * base^k`
- Labels are formatted as LaTeX strings with numeric notation or powers.

# Returns
- `(major_ticks, major_labels)` where `major_ticks::Vector{Float64}` and
  `major_labels` are corresponding LaTeX labels.

# See also
- `logminorticks` for matching minor ticks.

# Arguments
- `lo`: Lower interval bound (must be positive).
- `hi`: Upper interval bound (must satisfy `hi > lo`).

# Keyword Arguments
- `base`: Logarithm base used for tick construction (`base > 1`).

# Throws
- `DomainError`: Propagated from `_logticks_internal` for invalid interval/base inputs."""
function logticks(lo::Real, hi::Real; base::Real = 10.0)
    ticks, labels, _ = _logticks_internal(lo, hi; base = base)
    return ticks, labels
end

"""
    logminorticks(lo::Real, hi::Real; base::Real = 10.0)

Return logarithmic minor tick positions for the interval `[lo, hi]`.

See the primary method `logticks(lo, hi; base)` for the major-tick selection.

# Changes relative to `logticks`
- Returns only minor tick positions.
- Minor ticks are computed to complement the selected major pattern.
- If major ticks are already dense (`1..9` per decade), returns `Float64[]`.

# Arguments
- `lo`: Lower interval bound (must be positive).
- `hi`: Upper interval bound (must satisfy `hi > lo`).

# Keyword Arguments
- `base`: Logarithm base used for tick construction (`base > 1`).

# Returns
- `minor::Vector{Float64}`: Minor tick locations in `[lo, hi]`.

# Throws
- `DomainError`: Propagated from `_logticks_internal` for invalid interval/base inputs."""
function logminorticks(lo::Real, hi::Real; base::Real = 10.0)
    _, _, kind = _logticks_internal(lo, hi; base = base)

    pmin = floor(Int, log(lo) / log(base))
    pmax = ceil(Int,  log(hi) / log(base))

    if kind == :dense
        return Float64[]
    end

    mults = kind == :decades ? (2:9) : [2, 3, 4, 6, 7, 8, 9]
    minor = Float64[]
    for p in pmin:pmax
        for m in mults
            v = m * base^Float64(p)
            lo ≤ v ≤ hi && push!(minor, v)
        end
    end

    sort!(minor)
    return minor
end

"""
    CairoMakie.Makie.get_minor_tickvalues(::typeof(logminorticks), ::Union{typeof(log), typeof(log10), typeof(log2)}, ::Any, lo::Real, hi::Real)

Makie integration method that forwards minor-tick generation to `logminorticks`
for logarithmic axes.

# Arguments
- `lo`: Lower interval bound.
- `hi`: Upper interval bound.

# Returns
- `minor::Vector{Float64}`: Minor ticks from `logminorticks(lo, hi)`."""
function CairoMakie.Makie.get_minor_tickvalues(
    ::typeof(logminorticks),
    ::Union{typeof(log), typeof(log10), typeof(log2)},
    ::Any,
    lo::Real,
    hi::Real,
)
    return logminorticks(lo, hi)
end

"""
    apply_paper_theme!(;
        double_column::Bool = false,
        magnification::Real = 1.0,
        logscale_x::Bool = false,
        logscale_y::Bool = false,
        xticks = nothing,
        yticks = nothing,
        legendpos = :rt,
        legendpadding = nothing,
        legendmargin = nothing,
        n_Legend_columns::Int = 1,
        color_transparency::Float64 = 1.0,
    )

Configure and activate the project plotting theme for publication-style figures.

The function sets Makie global theme parameters (fonts, axis styling, legend,
palette, line widths, and optional logarithmic tick behavior) and returns the
recommended figure size in pixels.

# Returns
- `figsize::Tuple{Real,Real}` suitable for `CairoMakie.Figure(size = figsize)`.

# Notes
- `double_column` controls target paper width.
- `magnification` scales both figure size and visual element sizes.
- When `logscale_x` and/or `logscale_y` are enabled, this theme uses
  `logticks` and `logminorticks`.

# Arguments
- This method has no positional arguments.

# Keyword Arguments
- `double_column`: Use double-column paper width when `true`.
- `magnification`: Global scale factor for figure and style sizes (`> 0`).
- `logscale_x`, `logscale_y`: Enable logarithmic tick helpers for axes.
- `xticks`, `yticks`: Optional explicit tick specs that override automatic log ticks.
- `legendpos`: Legend anchor position.
- `legendpadding`, `legendmargin`: Optional legend layout spacing overrides.
- `n_Legend_columns`: Number of legend columns (`>= 1`).
- `color_transparency`: Alpha channel applied to palette colors (`0 <= value <= 1`).

# Throws
- `DomainError`: Raised when numeric option domains are invalid."""
function apply_paper_theme!(;
    double_column::Bool = false,
    magnification::Real = 1.0,
    logscale_x::Bool = false,
    logscale_y::Bool = false,
    xticks = nothing,
    yticks = nothing,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    n_Legend_columns::Int = 1,
    color_transparency::Float64 = 1.0,
)
    if !(magnification > 0)
        throw(DomainError(magnification, "magnification must be > 0"))
    end
    if !(n_Legend_columns >= 1)
        throw(DomainError(n_Legend_columns, "n_Legend_columns must be >= 1"))
    end
    if !(0.0 <= color_transparency <= 1.0)
        throw(DomainError(color_transparency, "color_transparency must satisfy 0.0 <= color_transparency <= 1.0"))
    end

    # physical sizes (in cm → inches)
    cm = 1 / 2.54
    dpi = 96
    pt = 4/3
    width_cm = double_column ? 17.8 : 8.6
    # keep height tied to single-column width to avoid doubling overall size
    height_cm = 0.75 * 8.6

    s(x) = x * magnification

    figsize = (s(width_cm * cm * dpi), s(height_cm * cm * dpi))

    # base typography (independent of figsize)
    base_fontsize = 11pt
    labelsize     = 11pt
    ticklabelsize = 11pt

    # line and tick sizes
    linewidth     = 1.5   
    ticksize      = 6
    minorticksize = 5

    axis_kwargs = (
        xlabelsize = s(labelsize),
        ylabelsize = s(labelsize),
        titlesize  = s(labelsize),

        xticklabelsize = s(ticklabelsize),
        yticklabelsize = s(ticklabelsize),

        xgridvisible = false,
        ygridvisible = false,
        gridcolor = (:black, 0.15),
        gridwidth = s(0.5),

        spinewidth = s(2.0),
        spinecolor = :black,
        leftspinevisible   = true,
        rightspinevisible  = true,
        topspinevisible    = true,
        bottomspinevisible = true,

        xtickwidth  = s(1.4),
        ytickwidth  = s(1.4),
        xticksize   = s(ticksize),
        yticksize   = s(ticksize),
        xminortickwidth = s(0.8),
        yminortickwidth = s(0.8),
        xminorticksize = s(minorticksize),
        yminorticksize = s(minorticksize),

        xticksmirrored = true,
        yticksmirrored = true,
        xtickalign  = 1.0,
        ytickalign  = 1.0,
        xminortickalign = 1.,
        yminortickalign = 1.,

        xminorticksvisible = true,
        yminorticksvisible = true,
    )

    if logscale_x
        axis_kwargs = merge(axis_kwargs, (
            xticks = logticks,
            xminorticks = logminorticks,
        ))
    end

    if logscale_y
        axis_kwargs = merge(axis_kwargs, (
            yticks = logticks,
            yminorticks = logminorticks,
        ))
    end

    xticks !== nothing && (axis_kwargs = merge(axis_kwargs, (xticks = xticks,)))
    yticks !== nothing && (axis_kwargs = merge(axis_kwargs, (yticks = yticks,)))

    base_colors = [
        Colors.colorant"#F1C21B",  # IBM Yellow
        Colors.colorant"#D12771",  # IBM Magenta
        Colors.colorant"#009D9A",  # IBM Teal
        Colors.colorant"#0F62FE",  # IBM Blue
        Colors.colorant"#6F6F6F",  # IBM Gray
        Colors.colorant"#FA4D56",  # IBM Red
        Colors.colorant"#24A148",  # IBM Green
    ]

    CairoMakie.set_theme!(
        CairoMakie.Theme(
            figure_padding = s(10),

            fonts = (
                regular      = "CMU Serif",
                italic       = "CMU Serif Italic",
                bold         = "CMU Serif Bold",
                bold_italic  = "CMU Serif Bold Italic",
            ),
            fontsize = s(base_fontsize),

            Axis = axis_kwargs,

            Legend = (
                framevisible = true,
                framewidth = s(2.0),
                framecolor = :black,
                padding = legendpadding === nothing ? (s(6), s(6), s(6), s(6)) : legendpadding,
                margin = legendmargin === nothing ? (s(11), s(11), s(11), s(11)) : legendmargin,
                labelsize = s(labelsize),
                position = legendpos,
                nbanks = n_Legend_columns,
            ),

            palette = (
                color = [(c, color_transparency) for c in base_colors],
            ),

            Lines = (
                linewidth = s(linewidth),
            ),

            Band = (
                color = (:auto, 0.3),
            ),
        )
    )
    return figsize
end
