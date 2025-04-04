using LinearAlgebra
using MathOptInterface
const MOI = MathOptInterface
using Convex
using SCS

include("test.jl");

struct DeePCcontroller
    T::Int64
    T_ini::Int64
    T_fut::Int64
    m::Int64
    p::Int64
    q::Float64
    r::Float64
    lam_g::Float64
    lam_sigma::Float64
    H_u::Matrix{Float64}
    H_y::Matrix{Float64}

    function DeePCcontroller(;
        T,
        T_ini,
        T_fut,
        m,
        p,
        q,
        r,
        H_u,
        H_y,
        lam_g,
        lam_sigma
    )
        new(T, T_ini, T_fut, m, p, q, r, lam_g, lam_sigma, H_u, H_y)
    end

end

function step(controller::DeePCcontroller, u_past, y_past, y_ref)

    u = Variable(controller.m * controller.T_fut)
    y = Variable(controller.p * controller.T_fut)
    g = Variable(controller.T - controller.T_ini - controller.T_fut + 1)
    sig_y = Variable(controller.T_ini * controller.p)

    R_sqrt = sqrt(controller.r) * I(controller.T_fut-1)
    Q_sqrt = sqrt(controller.q) * I(controller.T_fut)

    U_p = controller.H_u[1:controller.m*controller.T_ini, :]
    U_f = controller.H_u[(end-controller.m*controller.T_fut+1):end, :]
    Y_p = controller.H_y[1:controller.p*controller.T_ini, :]
    Y_f = controller.H_y[(end-controller.p*controller.T_fut+1):end, :]

    # Cost function components
    cost = controller.lam_g * norm(g, 1) + controller.lam_sigma * norm(sig_y, 2)
    cost += controller.r * norm(u, 2)
    cost += sumsquares(Q_sqrt * (y - (y_ref * ones(controller.T_fut))))

    # Constraints
    constraints = [
        U_p * g == u_past,
        U_f * g == u,
        Y_p * g == y_past,
        Y_f * g == y,
        u >= 0,
        u <= 1500
    ]

    problem = minimize(cost, constraints)
    solve!(problem, SCS.Optimizer, silent=true)

    if problem.status != MOI.OPTIMAL
        error("Optimization failed with status: $(problem.status)")
    end

    return evaluate(u), reshape(evaluate(y), (:, controller.p))
end
