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
    dic[id]["minimum"] = BenchmarkTools.time(BenchmarkTools.minimum(b))
    dic[id]["median"]  = BenchmarkTools.time(BenchmarkTools.median(b))
    dic[id]["mean"]    = BenchmarkTools.time(BenchmarkTools.mean(b))
    dic[id]["maximum"] = BenchmarkTools.time(BenchmarkTools.maximum(b))
    dic[id]["allocs"]  = BenchmarkTools.allocs(b)
    dic[id]["memory"]  = BenchmarkTools.memory(b)
end

function add_metadata(id::String, meta::String)
    dic[id] = meta
end
  
end