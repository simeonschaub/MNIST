using MLJ, MLJFlux, Flux
using ColorTypes


mutable struct MNISTBuilder <: MLJFlux.Builder
    n_hidden::Int
end

function MLJFlux.build(builder::MNISTBuilder, (n1, n2), m, n_channels)
    return Chain(
        flatten,
        Dense(n1 * n2, builder.n_hidden, relu),
        Dense(builder.n_hidden, m),
    )
end

function prepare_imgs(_imgs)
    img_size = size(_imgs[1])
    imgs = Array{Gray{Float32}}(undef, img_size..., length(_imgs))
    for i in eachindex(_imgs)
        imgs[:, :, i] .= Gray{Float32}.(_imgs[i])
    end
    #return reinterpret(SMatrix{img_size...,Gray{Float32},prod(img_size)}, vec(imgs))
    return [view(imgs, :, :, i) for i in axes(imgs, 3)]
end

imgs = Flux.Data.MNIST.images()
#imgs = prepare_imgs(imgs)
labels = Flux.Data.MNIST.labels()
labels = coerce(labels, Multiclass)

@load ImageClassifier
clf = ImageClassifier(;
    builder=MNISTBuilder(128),
    optimiser=ADAM(0.001),
    loss=Flux.crossentropy,
    epochs=6,
    batch_size=128,
)

mach = machine(clf, imgs, labels)

@time evaluate!(
    mach;
    resampling=Holdout(fraction_train=5/6, shuffle=true, rng=123),
    operation=predict_mode,
    measure=[accuracy, #=cross_entropy, =#misclassification_rate],
    verbosity = 3,
)
