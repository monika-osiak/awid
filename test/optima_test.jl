using Optim

# optimaze(f,∇f, )
l = optimize(f, ∇f, x, method=GradientDescent(), iterations=iters
  ;inplace=false)
println(l)
println(Optim.minimizer(l))