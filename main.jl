using CSV
using Random
using DataFrames
using Statistics
include("src/block_hankel.jl")
include("src/deepc_controller_mosek.jl")

# Load and process data
data_raw = CSV.read("data/ic7_log_step_multi_rpm.csv", DataFrame)
max_power = maximum(data_raw[!, "Motor Electrical Power"])
max_speed = maximum(data_raw[!, "Motor Speed"])
data = data_raw[1:7:end, :]
u_data = vec(Array(data[!, "Motor Speed"]))/max_speed
y_data = vec(Array(data[!, "Motor Electrical Power"]))/max_power
data = dropmissing(data)

# Controller parameters
T_ini = 4
T_fut = 10
L = T_ini + T_fut
m = 1
p = 1

# Create Hankel matrices
H_u = block_hankel(u_data, L, m)
H_y = block_hankel(y_data, L, p)

# Initialize controller
deepc_controller = DeePCcontroller(
    T=size(data, 1),
    T_ini=T_ini,
    T_fut=T_fut,
    m=m,
    p=p,
    q=1e5,
    r=1e4,
    H_u=H_u,
    H_y=H_y,
    lam_g=1e3,
    lam_sigma=0
)

# Initialize simulation
rng = Random.MersenneTwister(42)
u_hist = zeros(T_ini)  # Start with T_ini zeros
y_hist = zeros(T_ini)  # Start with T_ini zeros
y_ref_hist = Float64[] # Store reference history
solve_times = Float64[]

# Initial conditions for controller
u_past = zeros(T_ini)
y_past = zeros(T_ini)
y_ref = 1.5/max_power

for i in 1:1500
    if i == 1 || i % 30 == 0
        # List of reference values
        ref_options = [0.7, 1.1, 1.5, 1.8, 2.0, 2.5, 3.0]
        # Select a random reference from the list
        y_ref = rand(rng, ref_options) / max_power
        println("Reference changed to: $(y_ref * max_power) W")
    end
    
    push!(y_ref_hist, y_ref * max_power)
    
    time_start = time()
    u_optimal, y_predicted = step(deepc_controller, u_past, y_past, y_ref)
    solve_time = time() - time_start
    
    push!(solve_times, solve_time)
    push!(u_hist, u_optimal[1])
    push!(y_hist, y_predicted[1])
    
    # Update past values for next iteration
    u_past = vcat(u_past[2:end], u_optimal[1:1])
    y_past = vcat(y_past[2:end], y_predicted[1:1])
end

# Scale results back to original units
u_hist = u_hist * max_speed
y_hist = y_hist * max_power

println("Average solution time: $(round(mean(solve_times) * 1000, digits=2)) ms")

# Simple plot at the end
using Plots

# Create plots
plt = plot(
    layout = (2,1), 
    size = (800, 600),
    legend = :topright
)

# Plot 1: Output and reference
plot!(
    plt[1], 
    1:length(y_hist), 
    y_hist, 
    label = "Power Output",
    linewidth = 2, 
    color = :blue
)
plot!(
    plt[1], 
    1:length(y_ref_hist), 
    y_ref_hist, 
    label = "Power Reference",
    linewidth = 2, 
    linestyle = :dash, 
    color = :red
)
ylabel!(plt[1], "Power (W)")
title!(plt[1], "System Output and Reference")

# Plot 2: Control input
plot!(
    plt[2], 
    1:length(u_hist), 
    u_hist, 
    label = "Speed Input",
    linewidth = 2, 
    color = :green
)
xlabel!(plt[2], "Simulation Step")
ylabel!(plt[2], "Speed (RPM)")
title!(plt[2], "Control Input")

# Save and display
savefig(plt, "deepc_simulation_results.png")
display(plt)
