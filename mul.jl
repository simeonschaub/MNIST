using PaddedMatrices: jmul!
using ChainRulesCore
using Flux

mul(A, B) = jmul!(similar(A, axes(A, 1), Base.tail(axes(B))...), A, B)

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
