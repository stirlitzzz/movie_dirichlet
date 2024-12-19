include("io.jl")
include("types.jl")
include("utils.jl")
include("preprocessing.jl")
include("svd_model.jl")
using Clustering
using LinearAlgebra
using DataFrames
using Plots



function main()
    println("Starting up...")

    path="data/ml-32m/"

    args = parse_args()
    config = ClusteringConfig(k=args["k"], num_clusters=args["num_clusters"], train_ratio=args["train_ratio"], min_ratings_per_user=args["min_ratings_per_user"], min_ratings_per_movie=args["min_ratings_per_movie"])
    validate_config(config)
    
    df = load_ratings(path*"ratings.csv")
    ratings_data=preprocess_ratings(df, config.min_ratings_per_user, config.min_ratings_per_movie)

    println("Loaded $(size(df, 1)) ratings")
    println("Loaded $(length(ratings_data.user_map)) users")
    (train_users, test_users) = split_users(Set(keys(ratings_data.user_map)), config.train_ratio)

    dataset_train = get_data_for_user_subset(ratings_data, Set(train_users))
    dataset_test= get_data_for_user_subset(ratings_data, Set(test_users))

    #U, S, V = svd(dataset_train.imputed_matrix)
    svd_result = create_svd_model(dataset_train.imputed_matrix)
    truncated_svd_result = truncate_svd(svd_result, config.k)
    println("Truncated SVD to $(config.k) dimensions")

    println("Split $(length(train_users)) train users and $(length(test_users)) test users")
    println("train_users: $(train_users)")
    
    S=[svd_result.S[i, i] for i in 1:size(svd_result.S, 1)]

    println("dimensions of matrices: U=$(size(svd_result.U)), S=$(size(svd_result.S)), V=$(size(svd_result.V))")
    p1=plot(cumsum(S) ./ sum(S), legend=false, xlabel="Number of singular values", ylabel="Fraction of total variance", title="Fraction of total variance vs. number of singular values")
    savefig(p1, "cumulative_variance_plot.png")
    p2=plot(S./sum(S), xlim=[0,10],legend=false, xlabel="Number of singular values", ylabel="Fraction of total variance", title="Fraction of total variance vs. number of singular values")
    savefig(p2, "singular_values_plot.png")


    U_k=truncated_svd_result.U

    kmeans_result = kmeans(transpose(U_k), config.num_clusters)

    # Extract the cluster assignments for each user
    cluster_assignments = kmeans_result.assignments

    # Print the cluster centers
    println("Cluster Centers:")
    println(kmeans_result.centers)
    p3=scatter(U_k[:, 1], U_k[:, 2], group=cluster_assignments, xlabel="Feature 1", ylabel="Feature 2", title="User Clusters")
    savefig(p3, "user_clusters.png")


    #cluster_movie_forecasts=transpose(kmeans_result.centers)*S_k*transpose(V_k)
    println("kmeans_result.centers: $(kmeans_result.centers)")
    cluster_movie_forecasts=predict_from_users(transpose(kmeans_result.centers), truncated_svd_result)
    println("Cluster movie forecasts: $(size(cluster_movie_forecasts))")
    #num_movies= size(cluster_movie_forecasts, 2)
    #sorted_cluster_ratings= [sortperm(cluster_movie_forecasts[i, :], rev=true) for i in 1:num_clusters]

    #queues= create_queues(num_clusters, num_movies, sorted_cluster_ratings)
    #println("Cluster 1 top 10 movies: ", [dequeue!(queues[1]) for _ in 1:10])


end

# Call the main function
main()