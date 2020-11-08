using Flux, Zygote
using Flux: crossentropy, Data.DataLoader
using Random: shuffle

#include("_dense.jl")
include("mul.jl")

function accuracy(ŷ, y::Flux.OneHotMatrix)
    return count(axes(ŷ, 2)) do i
        y[argmax(view(ŷ, :, i)), i]
    end / size(ŷ, 2)
end

_view(x, i...) = view(x, i...)
function _view(x::Flux.OneHotMatrix, ::Colon, i)
    data = view(x.data, i)
    return Flux.OneHotMatrix{typeof(data)}(x.height, data)
end

function partition_batch(as::Tuple{Vararg{AbstractArray}}, n::Integer)
    n_batches = size(as[1])[end]
    s = div(n_batches, n, RoundUp)
    return [
        ntuple(length(as)) do j
            _view(as[j], fill(:, ndims(as[j]) - 1)..., i:min(i + s - 1, n_batches))
        end
        for i in 1:s:n_batches
    ]
end

function step!(model, loss_function, opt; nthreads=Threads.nthreads())
    ps = Flux.params(model)

    train_loss, train_accuracy = Threads.Atomic{Float64}(0), Threads.Atomic{Float64}(0) 
    gs = Vector{Zygote.Grads}(undef, nthreads)
    for xy in ds_train
        parts = partition_batch(xy, nthreads)
        Threads.@threads for i in eachindex(parts)
            x, y = parts[i]
            local ŷ
            loss, pb = Zygote.pullback(ps) do
                ŷ = model(x)
#                Zygote.@ignore @show summary(x), summary(ŷ), summary(y)
                return loss_function(ŷ, y)
            end
            gs[i] = pb(one(loss) * size(y, 2) / size(xy[2], 2))
            train_loss[] += loss * size(y, 2) / 50000
            train_accuracy[] += accuracy(ŷ, y) * size(y, 2) / 50000
        end
        foreach(gs -> Flux.update!(opt, ps, gs), gs)
    end

    test_loss, test_accuracy = 0., 0. 
    for (x, y) in ds_test
        ŷ = model(x)
        loss = loss_function(ŷ, y)
        test_loss += loss * size(y, 2) / 10000
        test_accuracy += accuracy(ŷ, y) * size(y, 2) / 10000
    end

    @show train_loss[], train_accuracy[]
    @show test_loss, test_accuracy
    nothing
end

function prepare_imgs(_imgs)
    img_size = size(_imgs[1])
    imgs = Array{Float32}(undef, img_size..., length(_imgs))
    for i in eachindex(_imgs)
        imgs[:, :, i] .= Float32.(_imgs[i])
    end
    return imgs
end

let
imgs = Flux.Data.MNIST.images()
labels = Flux.Data.MNIST.labels()

idx = shuffle(eachindex(imgs))
idx_train, idx_test = idx[1:50000], idx[50001:end]

ds_train = (prepare_imgs(imgs[idx_train]), Flux.onehotbatch(labels[idx_train], 0:9))
#global ds_train = DataLoader(ds_train, batchsize=128, shuffle=true)
global ds_train = DataLoader(ds_train, batchsize=128, shuffle=true)

ds_test = (prepare_imgs(imgs[idx_test]), Flux.onehotbatch(labels[idx_test], 0:9))
global ds_test = DataLoader(ds_test, batchsize=128, shuffle=false)

global n1, n2 = size(first(imgs))
end

n_hidden, m = 128, 10

model = Chain(
    flatten,
    FDense(n1 * n2, n_hidden, relu),
    FDense(n_hidden, m),
    softmax,
)

opt = ADAM(0.001)

using Profile
Profile.clear_malloc_data()

@time Flux.@epochs 6 step!(model, crossentropy, opt; nthreads=2)
