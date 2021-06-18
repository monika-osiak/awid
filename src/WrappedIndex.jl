module Wrapped

using ExportAll
using Base:Integer
export WrappedIndex, iterate

mutable struct WrappedIndex{T}
	stride::T
	size::T
	capacity::T
end

function Base.iterate(this::WrappedIndex{T}, produced=0) where T <: Integer
	if produced == this.size
        return nothing
    else
	return (this.stride + produced - 1) % this.capacity + 1, produced + 1
    end
end

function Base.iterate(rS::Iterators.Reverse{WrappedIndex{T}}, produced=0) where T <: Integer
	this = rS.itr
	if produced == this.size
        return nothing
    else
	return mod((this.stride - produced - 2), this.capacity) + 1, produced + 1
    end
end
function Base.eltype(::Type{WrappedIndex{T}}) where T <: Integer
    return T
end
Base.length(iter::WrappedIndex) = iter.size

    @exportAll

end