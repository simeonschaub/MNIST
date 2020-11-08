using PaddedMatrices: jmult!
using ChainRulesCore
using Flux

const jmul! = jmult!

function partition(as::Tuple{Vararg{AbstractArray}}, n::Integer)
    n_batches = size(as[1])[end]
    s = div(n_batches, n, RoundUp)
    return [
        ntuple(length(as)) do j
            _view(as[j], fill(:, ndims(as[j]) - 1)..., i:min(i + s - 1, n_batches))
        end
        for i in 1:s:n_batches
    ]
end

function mul(A, B)
    C = similar(A, axes(A, 1), Base.tail(axes(B))...)
	parts = partition((C, B), 2)
	Threads.@threads for (C, B) in parts
    	jmul!(C, A, B)
	end
    return C
end

function ChainRulesCore.rrule(
    ::typeof(mul),
    A::AbstractMatrix,
    B::AbstractMatrix,
)
    function mul_pullback(Ȳ)
        return (
            NO_FIELDS,
            InplaceableThunk(
                @thunk(mul(Ȳ, B')),
                X̄ -> jmul!(X̄, Ȳ, B', true, true)
            ),
            InplaceableThunk(
                @thunk(mul(A', Ȳ)),
                X̄ -> jmul!(X̄, A', Ȳ, true, true)
            )
        )
    end
    return mul(A, B), mul_pullback
end


struct FDense{F,S<:AbstractArray,T<:AbstractArray}
  W::S
  b::T
  σ::F
end

FDense(W, b) = FDense(W, b, identity)

function FDense(in::Integer, out::Integer, σ = identity;
               initW = Flux.glorot_uniform, initb = Flux.zeros)
  return FDense(initW(out, in), initb(out), σ)
end

Flux.@functor FDense

function (a::FDense)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ.(mul(W, x) .+ b)
end
