module Utils
export save_test, add_test, add_metadata

using BenchmarkTools
using JSON


dic = Dict()

function save_test()
    file = "stats.json"
    io = open(file, "w+");
    JSON.print(io, dic, 2) 
end

function add_test(id::String, b::BenchmarkTools.Trial)
    dic[id] = Dict()
    dic[id]["minimum"] = time(minimum(b))
    dic[id]["median"]  = time(median(b))
    dic[id]["mean"]    = time(mean(b))
    dic[id]["maximum"] = time(maximum(b))
    dic[id]["allocs"]  = allocs(b)
    dic[id]["memory"]  = memory(b)
end

function add_metadata(id::String, meta::String)
    dic[id] = meta
end
  
end