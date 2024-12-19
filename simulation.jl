using Distributions
include("types.jl")
inlcude("utils.jl")

struct SimulationState
    test_user_id::Int
    counts::Matrix{Float65}
    #means::Vector{Float64}
    rewards::Vector{Float64}

    movies_reccommended::Vector{Int}
    actual_error::Vector{Float64}
    forecast_ratings::Vector{Float64}
    actual_ratings::Vector{Float64}
    rewards::Vector{Float64}

end


function pick_cluster(user_ind, alpha, counts)
    dirichlet_prior= Dirichlet(alpha+counts[user_ind, :])
    cluster_sample= rand(dirichlet_prior)
    cluster_dist= Categorical(cluster_sample)
    cluster= rand(cluster_dist)
    return cluster
end

