using ArgParse
using Random
using DataStructures

function parse_args()
    s = ArgParseSettings()
   
    @add_arg_table s begin
        "--k"
        help = "Description for arg1"
        arg_type = Int

        "--num_clusters"
        help = "Description for arg2"
        arg_type = Int

        "--train_ratio"
        help = "Description for arg3"
        arg_type = Float64

        "--min_ratings_per_user"
        help = "Description for arg4"
        arg_type = Int

        "--min_ratings_per_movie"
        help = "Description for arg5"
        arg_type = Int
    end
    
    return ArgParse.parse_args(s)
end

function split_users(active_users, train_ratio)
    num_train = Int(floor(train_ratio * length(active_users)))
    train_users = Random.shuffle(collect(active_users))[1:num_train]
    test_users = setdiff(active_users, train_users)
    return (train_users, test_users)
end


function validate_config(config::ClusteringConfig)
    @assert 0 < config.k <= 100 "k must be between 1 and 100"
    @assert 0 < config.num_clusters <= 100 "num_clusters must be between 1 and 100"
    @assert 0 < config.train_ratio < 1 "train_ratio must be between 0 and 1"
end


# Create an array of n empty Queues
function create_queues(num_clusters, num_movies, sorted_cluster_ratings)
    queues = [Queue{Int}() for _ in 1:num_clusters]
    for i in 1:num_clusters
        for j in 1:num_movies
            enqueue!(queues[i], sorted_cluster_ratings[i][j])
        end
    end
    return queues
end