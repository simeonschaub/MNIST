using Flux, Zygote
using Flux: crossentropy, Data.DataLoader
using Random: shuffle

#include("_dense.jl")

function accuracy(ŷ, y::Flux.OneHotMatrix)
    return count(axes(ŷ, 2)) do i
        y[argmax(view(ŷ, :, i)), i]
    end / size(ŷ, 2)
end

function step!(model, ds_train, ds_test, loss_function, opt)
    ps = Flux.params(model)

    train_loss, train_accuracy = 0., 0. 
    for (x, y) in ds_train
        local ŷ
        loss, pb = Zygote.pullback(ps) do
            ŷ = model(x)
            return loss_function(ŷ, y)
        end
        gs = pb(one(loss))
        Flux.update!(opt, ps, gs)
        train_loss += loss * size(y, 2) / 50000
        train_accuracy += accuracy(ŷ, y) * size(y, 2) / 50000
    end

    test_loss, test_accuracy = 0., 0. 
    for (x, y) in ds_test
        ŷ = model(x)
        loss = loss_function(ŷ, y)
        test_loss += loss * size(y, 2) / 10000
        test_accuracy += accuracy(ŷ, y) * size(y, 2) / 10000
    end

    @show train_loss, train_accuracy
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
imgs_train = Flux.Data.MNIST.images()
labels_train = Flux.Data.MNIST.labels()
ds_train = (prepare_imgs(imgs_train), Flux.onehotbatch(labels_train, 0:9))
global ds_train = DataLoader(ds_train, batchsize=128, shuffle=true)

imgs_test = Flux.Data.MNIST.images(:test)
labels_test = Flux.Data.MNIST.labels(:test)
ds_test = (prepare_imgs(imgs_test), Flux.onehotbatch(labels_test, 0:9))
global ds_test = DataLoader(ds_test, batchsize=128, shuffle=false)

global n1, n2 = size(first(imgs_train))
end

n_hidden, m = 128, 10

model = Chain(
    flatten,
    Dense(n1 * n2, n_hidden, relu),
    Dense(n_hidden, m),
    softmax,
)

opt = ADAM(0.001)

using Profile
Profile.clear_malloc_data()

GC.gc(); GC.gc()

@time Flux.@epochs 6 step!(model, ds_train, ds_test, crossentropy, opt)
