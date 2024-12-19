struct RatingsData
    raw_matrix::Matrix{Union{Float64, Missing}}
    imputed_matrix::Matrix{Float64}
    user_map::Dict{Int, Int}
    movie_map::Dict{Int, Int}
    user_means::Vector{Float64}
end


struct ClusteringConfig
    k::Int64
    num_clusters::Int64
    train_ratio::Float64
    min_ratings_per_user::Int64
    min_ratings_per_movie::Int64

    function ClusteringConfig(; k, num_clusters, train_ratio, min_ratings_per_user, min_ratings_per_movie)
        new(k, num_clusters, train_ratio, min_ratings_per_user, min_ratings_per_movie)
    end
end 


struct SVDResult
    U::Matrix{Float64}
    S::Matrix{Float64}
    V::Matrix{Float64}
end

#=
struct SimulationState
    test_user_id::Int
    movie_queues::Vector{Queue{Int}}
    counts::Vector{Float64}
    #means::Vector{Float64}
    movies_reccommended::Vector{Int}
    actual_error::Vector{Float64}
    forecast_ratings::Vector{Float64}
    actual_ratings::Vector{Float64}
    rewards::Vector{Float64}
end
=#