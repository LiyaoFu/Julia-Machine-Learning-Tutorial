## A convolutional neural network model
## by Liyao Fu and Zhongyi Cao
## based on the Julia Flux package.
## This is an example to show how CNN
## should be implemented in Julia.
## This script also combines various
## packages from the Julia ecosystem with Flux.

using Flux
using Flux.Data.MNIST
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, crossentropy, throttle
using Flux: @epochs
using Statistics
using Base.Iterators: partition
using BSON
using Serialization: serialize, deserialize
using Suppressor
using ProgressMeter: @showprogress
using Plots

# General parameters
# The number of batches to partition the whole data set.
# Since the training set has 60 thousand samples, we don't
# want to pass all of them to the CNN model as it will be
# time-consuming.
global partnum = 128
# Determine whether to use GPU or CPU for training
global use_gpu = true

# utility functions
# Loss function used in training. We use cross entropy function
# from Flux library here, and there are other options such as the
# mean square error.
function loss(x, y)
    return Flux.crossentropy(model(x), y)
end

# Evaluate the accuracy of the training by calculating how many
# predictions equal to the results. Onecold, as opposite to onehot,
# finds the largest element for each label from 1 to 10(referring to
# the ten numbers the dataset represent), as it was converted into
# the onehot format when manipulating the data.
function evaluate(x, y)
    return mean(onecold(model(x)) .== onecold(y))
end

# Divides the samples into batches. The x set(data) and y set(label)
# are matched since the onehotbatch function from Flux finds the index
# of the hottest, or most frequent, value from labels and convert its
# format into a boolean vector with only that value being true.
function make_batch(xset, yset, idx)
    geti(n) = Float32.(xset[idx[n]])
    # the labels of the values are digits, as indicated by the dataset
    ylabs = 0:9
    # we want the batch be a four dimensional float array, with the first
    # two representing its two dimensions(width and height), and the last
    # two representing the channel layer and the number of batch.
    # The images from dataset are 28x28, with 1 channel, and 128(batchsize
    # can be modified in parameter) batches.
    xbatch = Array{Float32}(undef, (size(xset[1])..., 1, length(idx)))
    for i in 1:length(idx)
        xbatch[:, :, :, i] = geti(i)
    end
    ybatch = onehotbatch(yset[idx], ylabs)
    return (xbatch, ybatch)
end

# The function to load the data set and convert it to our desired format
function load_data()
    # Load data set from Flux, so we want to suppress the
    # warning about using MLDatasets instead.(this will appear
    # on the latest Julia version)
    @suppress_err begin
    global xtrain = Flux.Data.MNIST.images()
    global ytrain = Flux.Data.MNIST.labels()
    global xtest = Flux.Data.MNIST.images(:test)
    global ytest = Flux.Data.MNIST.labels(:test)
    end
    trainlen = length(xtrain)
    testlen = length(xtest)
    println("Loading MNIST data set: $(trainlen) train samples and $(testlen) test samples")
    # Make batch with the function we defined. Partition the training set
    # first before making batches for it.
    println("Making batch")
    train_set = [make_batch(xtrain, ytrain, i) for i in partition(1:trainlen, partnum)]
    test_set = make_batch(xtest, ytest, 1:testlen)
    return train_set, test_set
end

# Build a CNN model. There are many choices to use,
# here we apply four convolutional layers, two max-pooling
# layers, a dense layer, and a softmax layer as an example
# of a common CNN.
function build_model()
    # Create model
    println("Building model")
    return Chain(
        Conv((5, 5), 1=>8, pad=2, stride=2, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 8=>16, pad=1, stride=2, relu),
        Conv((3, 3), 16=>32, pad=1, stride=2, relu),
        Conv((3, 3), 32=>32, pad=1, stride=2, relu),
        GlobalMaxPool(),
        flatten,
        Dense(32, 10),
        softmax)
end

# The training function trains the model with a maximum of epochnum(200) epochs,
# and logs the information for later analysis.
function train()
    # train in each epoch
    # These parameters can be changed for testing
    global lr = 0.0003
    global min_lr = 1e-7
    global dec_lr = 10.0
    global opt = ADAM(lr)
    global epochnum = 200
    global before_better = 7
    global before_end = 15

    # These logs the best accuracy and its epoch, plus the last time
    # a best accuracy or a modification happened.
    global best_acc = 0
    global best_round = 0
    global last_round = 0
    # We want to log the epoch and accuracy for each epoch.
    x = []
    y = []
    # The following code can train with one line
    # @epochs epochnum Flux.train!(loss, ps, train_data, opt)
    # However, we need extra information to build our reporting
    # graph and analyzing results.
    for epoch in 1:epochnum
        # The easiest way to train with Julia, using the
        # Flux.train! function.
        Flux.train!(loss, params(model), train_set, opt)
        # Evaluate accuracy
        acc = evaluate(test_set...)
        # Add data to array
        push!(x, epoch)
        push!(y, acc)
        println("Accuracy in Epoch $(epoch): $(acc)")
        # If a accuracy higher than the historical record occurs,
        # log new record
        if (acc > best_acc)
            println("New Best Accuracy found")
            best_acc = acc
            last_round = epoch
            best_round = epoch
        end
        # If no better accuracy lasts for ten epochs, we take it as
        # no improvement and requires lower learning rate
        if (epoch - last_round >= before_better && lr >= min_lr)
            lr = lr / dec_lr
            opt = ADAM(lr)
            println("No improvement for $(before_better) rounds. Decrease learning rate by $(dec_lr) times.")
            last_round = epoch
        end
        # If no better accuracy lasts for twenty epochs, we take
        # the training as converged
        if (epoch - last_round >= before_end)
            println("No improvement for $(before_end) rounds. Ending training")
            break
        end
    end
    println("Best accuracy from training: $(best_acc)")
    # log data for plotting
    return x, y, best_round
end

# Main function for everything
function main()
    # load data
    train_set, test_set = load_data()
    # build model
    global model = build_model()
    # GPU for better performance
    if use_gpu
        global device = gpu
        println("Training on GPU")
    else
        global device = cpu
        println("Training on CPU")
    end
    global train_set = device.(train_set)
    global test_set = device.(test_set)
    model = device(model)

    println("Accuracy before training: $(evaluate(test_set...))")
    # Training
    plotx, ploty, mark = train()
    # Plotting
    plot(plotx, ploty, title = "Accuracy by epoch", label = "Accuracy")
    plot!([mark], seriestype="vline", label = "Best Accuracy")
end

# Call the main function to start
main()