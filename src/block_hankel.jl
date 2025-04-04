function block_hankel(w::Array{Float64,1}, L::Int, d::Int)::Array{Float64,2}
    """
    Builds block Hankel matrix for column vector w of order L
    args:
        w = column vector
        d = dimension of each block in w
        L = order of hankel matrix
    """
    T = Int(length(w) / d)
    if L > T
        throw(ArgumentError("L must be smaller than T"))
    end

    H = zeros(L * d, T - L + 1)
    for i in 0:(T-L)
        H[:, i+1] = w[(d*i+1):(d*(L+i))]
    end

    return H
end
