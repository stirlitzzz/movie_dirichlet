using Distributions
using Random

mutable struct Environment
    available_movies
    movie_reviews
end


struct DirichletPolicy #<: Policy 
    counts
    cluster_centers
    cluster_forecasts
end

struct RandomClusterPolicy
    num_clusters
    cluster_centers
    cluster_forecasts
end

function (π::DirichletPolicy)(s::Environment, user::Int, movie::Int)
    alpha=ones(length(π.counts))
    dirichlet_prior= Dirichlet(alpha+π.counts)
    cluster_sample= rand(dirichlet_prior)
    cluster_dist= Categorical(cluster_sample)
    cluster= rand(cluster_dist)

    forecasts=π.cluster_forecasts[cluster, s.available_movies]
    sorted_indices = sortperm(forecasts, rev=true)
    top_half=sorted_indices[1:length(sorted_indices)÷2]
    movie_pick= rand(top_half)
    movie_pick= argmax(forecasts)
    #movie_pick= argmax(forecasts)
    #println("movie_pick: $(movie_pick)")
    return (cluster,s.available_movies[movie_pick])
end

function updated_policy(π::DirichletPolicy, user_ind, movie_ind, actual_rating)
    #average_user_rating=mean(dataset_test.raw_matrix[user_ind, :])
    #average_user_rating=mean(skipmissing(dataset_test.raw_matrix[user_ind, :]))
    movie_forecasts=π.cluster_forecasts[:, movie_ind]
    best_forecast=argmin((movie_forecasts.-actual_rating).^2)
    π.counts[best_forecast]+=1
    #println("updated counts: $(π.counts)")
    return π
end

function forecast_error(π::DirichletPolicy, user_ind, movie_ind, actual_rating,forecast_cluster)
    #average_user_rating=mean(dataset_test.raw_matrix[user_ind, :])
    #average_user_rating=mean(skipmissing(dataset_test.raw_matrix[user_ind, :]))
    movie_forecasts=π.cluster_forecasts[:, movie_ind]
    best_forecast=argmin((movie_forecasts.-actual_rating).^2)
    #π.counts[best_forecast]+=1
    #println("updated counts: $(π.counts)")
    return (movie_forecasts[forecast_cluster]-actual_rating)^2
end

function forecast_quality(π::DirichletPolicy, user_ind, movie_ind, actual_rating,forecast_cluster)
    #average_user_rating=mean(dataset_test.raw_matrix[user_ind, :])
    #average_user_rating=mean(skipmissing(dataset_test.raw_matrix[user_ind, :]))
    movie_forecasts=π.cluster_forecasts[:, movie_ind]
    random_cluster= rand(1:length(π.counts))
    best_forecast=argmin((movie_forecasts.-actual_rating).^2)
    #π.counts[best_forecast]+=1
    #println("updated counts: $(π.counts)")
    return (movie_forecasts[forecast_cluster]-actual_rating).^2-(movie_forecasts[random_cluster]-actual_rating)^2
end


function (π::RandomClusterPolicy)(s::Environment, user::Int, movie::Int)
    cluster= rand(1:π.num_clusters)

    forecasts=π.cluster_forecasts[cluster, s.available_movies]
    #movie_pick= argmax(forecasts)
    sorted_indices = sortperm(forecasts, rev=true)
    top_half=sorted_indices[1:length(sorted_indices)÷2]
    movie_pick= rand(top_half)
    movie_pick= argmax(forecasts)
    return (cluster,s.available_movies[movie_pick])
end

function updated_policy(π::RandomClusterPolicy, user_ind, movie_ind, actual_rating)
    return π
end

function forecast_error(π::RandomClusterPolicy, user_ind, movie_ind, actual_rating,forecast_cluster)
    movie_forecasts=π.cluster_forecasts[:, movie_ind]
    best_forecast=argmin((movie_forecasts.-actual_rating).^2)
    return (movie_forecasts[forecast_cluster]-actual_rating)^2
end



reward(env,user,movie)=env.movie_reviews[user,movie]
new_env(env,movie)=Environment(setdiff(env.available_movies, Set([movie])),env.movie_reviews)

function simulate_episode(env, user_ind, policy,num_steps)
    #println("len available_movies=$(length(env.available_movies))")
    all_tracking=[]
    for i in 1:num_steps
        tracking=Dict{String,Any}()
        #print("user_ind: $(user_ind)")
        #println("result=$(policy(env,user_ind,1))")
        (cluster,a)=policy(env,user_ind,1)
        #print("policy: $(a)")
        r=reward(env,user_ind,a)
        env=new_env(env,a)
        tracking["i"]=i
        tracking["user"]=1
        tracking["movie"]=a
        tracking["reward"]=r
        tracking["action"]=a
        tracking["cluster"]=cluster
        tracking["error"]=forecast_error(policy,user_ind,a,r,cluster)
        tracking["len available_movies"]=length(env.available_movies)
        #println("len available_movies=$(length(env.available_movies))")
        #tracking["policy_counts"]=copy(policy.counts)
        if hasproperty(policy, :counts)
            tracking["policy_counts"] = copy(policy.counts)
            tracking["forecast_quality"]=forecast_quality(policy,user_ind,a,r,cluster)
        end
        policy=updated_policy(policy,user_ind,a,r)
        #println("updated policy: $(policy.counts)")
        #tracking["state"]=env
        all_tracking=push!(all_tracking,tracking)
    end
    #println("final len available_movies=$(length(env.available_movies))")
    #println("user_ind=$(user_ind)")
    return all_tracking
end


function run_all_users_episodes(π,dataset_test, kmeans_result, cluster_movie_forecasts, num_clusters, num_steps=100)
    all_user_tracking=[]
    π_0=deepcopy(π)
    for user_ind in 1:length(dataset_test.user_map)
        available_movies = findall(!ismissing, dataset_test.raw_matrix[user_ind, :])
        env=Environment(available_movies,dataset_test.imputed_matrix)
        #π=DirichletPolicy(ones(num_clusters),kmeans_result.centers,cluster_movie_forecasts)
        π=deepcopy(π_0)
        #π=policy2
        user_tracking=simulate_episode(env,user_ind,π,num_steps)
        user_tracking_processed=DataFrame(user_tracking)
        all_user_tracking=push!(all_user_tracking,user_tracking_processed)
    end
    return all_user_tracking
end
