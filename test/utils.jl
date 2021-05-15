module Utils
export save_test, add_test

using BenchmarkTools
using JSON

file = "stats"
io = open(file, "w+");

dic = Dict()

function add_test(id::String, b::BenchmarkTools.Trial)
    dic[id] = Dict()
    dic[id]["minimum"] = time(minimum(b))
    dic[id]["median"]  = time(median(b))
    dic[id]["mean"]    = time(mean(b))
    dic[id]["maximum"] = time(maximum(b))
    dic[id]["allocs"]  = allocs(b)
    dic[id]["memory"]  = memory(b)
end

function save_test()
    global io 
    JSON.print(io, dic, 2) 
end
  
end