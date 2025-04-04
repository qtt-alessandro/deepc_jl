using LinearAlgebra
using JuMP
using MosekTools

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
    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, "LOG", 0)
    # Variables
    @variable(model, u[1:controller.m * controller.T_fut])
    @variable(model, y[1:controller.p * controller.T_fut])
    @variable(model, g[1:(controller.T - controller.T_ini - controller.T_fut + 1)])
    @variable(model, sig_y[1:(controller.T_ini * controller.p)])
    
    # Auxiliary variables for L1 and L2 norms
    @variable(model, g_abs[1:length(g)] >= 0)
    @variable(model, t_sig_y >= 0)  # For L2 norm of sig_y
    
    # For L2 norms in objective
    @variable(model, t_u >= 0)     # For L2 norm of u
    @variable(model, t_y_ref >= 0) # For L2 norm of (y - y_ref)
    
    # Matrices
    U_p = controller.H_u[1:controller.m*controller.T_ini, :]
    U_f = controller.H_u[(end-controller.m*controller.T_fut+1):end, :]
    Y_p = controller.H_y[1:controller.p*controller.T_ini, :]
    Y_f = controller.H_y[(end-controller.p*controller.T_fut+1):end, :]
    
    R_sqrt = sqrt(controller.r) * I(controller.T_fut-1)
    Q_sqrt = sqrt(controller.q) * I(controller.T_fut)
    
    # L1 norm constraints for g
    @constraint(model, [i=1:length(g)], g_abs[i] >= g[i])
    @constraint(model, [i=1:length(g)], g_abs[i] >= -g[i])
    
    # L2 norm constraint for sig_y
    @constraint(model, [t_sig_y; sig_y] in SecondOrderCone())
    
    # L2 norm constraint for u
    @constraint(model, [t_u; u] in SecondOrderCone())
    
    # L2 norm constraint for tracking error
    y_ref_vector = y_ref * ones(controller.T_fut)
    @variable(model, y_error[1:controller.p * controller.T_fut])
    @constraint(model, y_error .== y .- y_ref_vector)
    @constraint(model, [t_y_ref; Q_sqrt * y_error] in SecondOrderCone())
    
    # System constraints
    @constraint(model, U_p * g .== u_past)
    @constraint(model, U_f * g .== u)
    @constraint(model, Y_p * g .== y_past)
    @constraint(model, Y_f * g .== y)
    
    # Input constraints
    @constraint(model, u .>= 0)
    @constraint(model, u .<= 1)
    
    # Objective function
    @objective(model, Min, 
               controller.lam_g * sum(g_abs) + 
               controller.lam_sigma * t_sig_y + 
               controller.r * t_u + 
               controller.q * t_y_ref)
    
    # Solve the optimization problem
    optimize!(model)
    
    if termination_status(model) != MOI.OPTIMAL
        error("Optimization failed with status: $(termination_status(model))")
    end
    
    u_sol = value.(u)
    y_sol = value.(y)
    
    return u_sol, reshape(y_sol, (:, controller.p))
end