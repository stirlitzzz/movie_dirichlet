using LinearAlgebra
include("types.jl")

function create_svd_model(m::Matrix)
    U, S, V = svd(m)

    println("U: $(size(U)), S: $(size(S)), V: $(size(V))")
    #println("Diagonal(U): $(Diagonal(U))")
    return SVDResult(U, Diagonal(S), V)
end

function truncate_svd(svd_result::SVDResult, k::Int)
    #U = svd_result.U[:,1:k]
    U=svd_result.U[1:k, :]'
    S = svd_result.S[1:k, 1:k]
    #V=svd_result.V[1:k,:]'
    V = svd_result.V[:, 1:k]

    return SVDResult(U, S, V)
end

function predict_from_users(m_user, svd_result::SVDResult)
    return m_user* svd_result.S * svd_result.V'
end