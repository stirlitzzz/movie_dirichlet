using DataFrames
using Statistics
include("types.jl")


function filter_active_users(ratings::DataFrame, min_ratings::Int)::Set{Int}
    # User filtering logic
    user_counts = combine(groupby(ratings, :userId), nrow => :count)
    active_users=Set(user_counts[user_counts.count .>= min_ratings, :userId])

    return active_users
end

function filter_active_movies(ratings::DataFrame, min_ratings::Int)::Set{Int}
    # Movie filtering logic
    movie_counts = combine(groupby(ratings, :movieId), nrow => :count)
    active_movies=Set(movie_counts[movie_counts.count .>= min_ratings, :movieId])

    return active_movies
end


function filter_movies_users(ratings::DataFrame, active_users::Set{Int}, active_movies::Set{Int})::DataFrame
    # Filter out inactive users and movies
    return filter(row -> row.userId in active_users && row.movieId in active_movies, ratings)
end

function CreateImputedMatrix(mat,subtract_mean=false)
    #imputed_matrix= copy(mat)
    imputed_matrix=fill(NaN, size(mat))

    means= [mean(skipmissing(row)) for row in eachrow(mat)]
    for i in 1:size(mat, 1)
        row= mat[i, :]
        mean_rating= mean(skipmissing(row))
        for j in 1:size(mat, 2)
            if ismissing(mat[i, j])
                imputed_matrix[i, j]= mean_rating
            else
                imputed_matrix[i, j]= mat[i, j]
            end
            if subtract_mean
                imputed_matrix[i, j]-= mean_rating
            end
    
        end
    end
    #imputed_matrix .= Float64().(imputed_matrix)
    return (imputed_matrix, means)
end

function CreateRatingStruct(active_users, popular_movies, dense_ratings)
    user_to_idx = Dict(user => idx for (idx, user) in enumerate(active_users))
    movie_to_idx = Dict(movie => idx for (idx, movie) in enumerate(popular_movies))
    ratings_matrix = Matrix{Union{Float64, Missing}}(missing, length(active_users), length(popular_movies))
    for row in eachrow(dense_ratings)
        user_idx = user_to_idx[row.userId]
        movie_idx = movie_to_idx[row.movieId]
        ratings_matrix[user_idx, movie_idx] = row.rating
    end
    mean_rating = mean(skipmissing(ratings_matrix))
    (imputed_ratings_matrix,means) = CreateImputedMatrix(ratings_matrix,true)
    result=RatingsData(ratings_matrix, imputed_ratings_matrix, user_to_idx, movie_to_idx, means)
    #imputed_ratings_matrix = fill(mean_rating, length(active_users), length(popular_movies))
    return result
end


function preprocess_ratings(ratings::DataFrame, min_ratings_per_user::Int, min_ratings_per_movie::Int)::RatingsData
    active_users = filter_active_users(ratings, min_ratings_per_user)
    active_movies = filter_active_movies(ratings, min_ratings_per_movie)
    filtered_ratings = filter_movies_users(ratings, active_users, active_movies)
    ratings_data = CreateRatingStruct(active_users, active_movies, filtered_ratings)
    result=RatingsData(ratings_data.raw_matrix, ratings_data.imputed_matrix, ratings_data.user_map, ratings_data.movie_map, ratings_data.user_means)
    return result
 end

function get_data_for_user_subset(ratings_data::RatingsData, user_subset::Set{Int})
    user_indices = [ratings_data.user_map[user] for user in user_subset]
    raw_matrix = ratings_data.raw_matrix[user_indices, :]
    imputed_matrix = ratings_data.imputed_matrix[user_indices, :]
    user_means = ratings_data.user_means[user_indices]
    user_to_idx = Dict(user => idx for (idx, user) in enumerate(user_subset))
    new_ratings_data = RatingsData(raw_matrix, imputed_matrix, user_to_idx, ratings_data.movie_map, user_means)
    return new_ratings_data

end