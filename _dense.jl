using LinearAlgebra, Zygote, Flux

struct _Dense{F,S<:AbstractArray,T<:AbstractArray}
    W::S
    b::T
    σ::F
    cache::IdDict
    _Dense(W::S, b::T, σ::F) where {F,S,T} = new{F,S,T}(W, b, σ, IdDict())
end

_Dense(W, b) = _Dense(W, b, identity)

function _Dense(in::Integer, out::Integer, σ = identity;
               initW = Flux.glorot_uniform, initb = Flux.zeros)
    return _Dense(initW(out, in), initb(out), σ)
end

Flux.@functor _Dense

function (a::_Dense)(x::AbstractArray)
    return _dense(a.W, a.b, a.σ, a.cache, x)
end

function _dense(W, b, σ, cache, x)
    ax = axes(W, 1), Base.tail(axes(x))...
    tmp = get!(() -> similar(x, ax...), cache, (typeof(x), ax...))::typeof(x)
    mul!(tmp, W, x)
    tmp .= σ.(tmp .+ b)
    return tmp
end

Zygote.@adjoint function _dense(W, b, σ, cache, x)
    ax = axes(W, 1), Base.tail(axes(x))...
    tmp = get!(() -> similar(x, ax...), cache, (typeof(x), ax...))::typeof(x)

    mul!(tmp, W, x)
    #tmp .+= b
    res, dσ = Zygote._pullback(__context__, (tmp, b) -> σ.(tmp .+ b), tmp, b)

    function _dense_pullback(Δ)
        _, Δ, tmp_b = dσ(Δ)

        ax_W = axes(Δ, 1), axes(x, 1)
        tmp_W = get!(() -> similar(x, ax_W...), cache, (typeof(x), ax_W..., :W))::typeof(x)

        #ax_b = (axes(Δ, 1),)
        #tmp_b = get!(() -> similar(x, ax_b...), cache, (typeof(x), ax_b..., :b))::typeof(x).name.wrapper{eltype(x),1}

        ax_x = axes(W, 2), Base.tail(axes(Δ))...
        tmp_x = get!(() -> similar(x, axes(W, 2), axes(Δ, 2)), cache, (typeof(x), axes(W, 2), axes(Δ, 2), :x))::typeof(x)

        mul!(tmp_W, Δ, x')
        #sum!(tmp_b, Δ)
        mul!(tmp_x, W', Δ)
        return tmp_W, tmp_b, nothing, nothing, tmp_x
    end
    return res, _dense_pullback
end
