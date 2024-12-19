using CSV
function load_ratings(path::String)
    return CSV.read(path, DataFrame)
end