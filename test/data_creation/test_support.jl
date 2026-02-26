function _normalize_ws(s::AbstractString)
    return replace(s, r"\s+" => " ")
end

function _extract_block(src::AbstractString, start_marker::AbstractString, end_marker::AbstractString)
    i = findfirst(start_marker, src)
    i === nothing && error("Start marker not found: $start_marker")
    j = findnext(end_marker, src, last(i) + 1)
    j === nothing && error("End marker not found: $end_marker")
    return src[first(i):first(j)-1]
end

function _eval_block_in_module(src::AbstractString; module_name::Symbol = gensym(:ScriptBlock))
    m = Module(module_name)
    Core.eval(m, :(using LinearAlgebra))
    Core.eval(m, :(using Statistics))
    Core.eval(m, Meta.parseall(src))
    return m
end

function _run_capture(cmd::Cmd)
    out = Pipe()
    proc = run(pipeline(ignorestatus(cmd), stdout = out, stderr = out); wait = false)
    close(out.in)
    text = read(out, String)
    wait(proc)
    return proc.exitcode, text
end
