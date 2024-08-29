
# 2 pool DON model including DON autotrophic uptake
# For the article "Oligotrophic Ocean New Production Supported by Lateral Transport of Dissolved Organic Nutrients" submited to GBC

using CSV
using DataFrames
using MAT

using F1Method
using AIBECS
import AIBECS: @units, units
import AIBECS: @limits, limits
import AIBECS: @initial_value, initial_value
import AIBECS: @flattenable, flattenable
import AIBECS: @prior, prior
using Unitful: m, d, s, yr, Myr, mol, mmol, μmol, μM, NoUnits
using Distributions
using WorldOceanAtlasTools


# AIBECS model
grd, T_Circ = OCIM2.load()
T_DON(p) = T_Circ


file = matopen("NPP.mat") # replace NPP.mat with your own NPP data field

NPP = read(file, "NPP") # mmol C m-3 yr-1 

# load WOA NO3 data and vectorize it
file2 = matopen("WOANO3.mat") # World Ocean Atlas NO3 concentration field
WOANO3 = read(file2)
WOANO3_v = vectorize(WOANO3["WOANO3"],grd) 

NPP_v = vectorize(NPP["NPP"],grd)     # mmol C m-3 yr-1
NPP_v = NPP_v.*(0.125.+ 0.03.*WOANO3_v./(0.32.+WOANO3_v)) # use equation in Galbraith pnas 2014
NPP_v = ustrip.(upreferred.(NPP_v * u"mmol/m^3/yr")) 
WOANO3_v = ustrip.(upreferred.(WOANO3_v * u"mmol/m^3"))


z_top = topdepthvec(grd) # uptake only in top two layer


# SLDON remineralization
function R_SLDON(x,p)
    @unpack τ_SLDON = p
     @. x / τ_SLDON
end

# RDON remineralization
function R_RDON(x,p)
    @unpack τ_RDON = p
      @. x / τ_RDON
end

# RDON production via SLDON transformation
function U_RDON(x,p)
    @unpack σ1,τ_SLDON = p
    @.  σ1* x / τ_SLDON
end

# SLDON production from NPP
function S_SLDON(p)
    @unpack α, β,f1 = p
    @.  0.5*f1*α*NPP_v^β*(z_top≤73)
end

# RDON production from NPP
function S_RDON(p)
    @unpack α, β,f2 = p
    @. 0.5*f2*α*NPP_v^β*(z_top≤73) 
end

# SLDON autotrophic uptake
function U_SLDON(x,p)
     @unpack Vm, Km = p
     @. ((Km+WOANO3_v)/ WOANO3_v)/Vm*x*(z_top≤73) 
end


# combine right hand equations for SLDON
function G_SLDON(SLDON,RDON,p)
    - R_SLDON(SLDON,p) + S_SLDON(p)-U_SLDON(SLDON,p) +0*RDON
end

# combine right hand equations for RDON
function G_RDON(SLDON,RDON,p)
    U_RDON(SLDON,p)- R_RDON(RDON,p) + S_RDON(p)+0*SLDON 
end


import AIBECS: @units, units
import AIBECS: @initial_value, initial_value
import AIBECS: @limits, limits
import AIBECS: @flattenable, flattenable
using Unitful: m, d, s, yr, Myr, mol, mmol, μmol, μM

# define initial values
∞ = Inf
@initial_value @units @flattenable @limits struct DONmodelParameters{U} <: AbstractParameters{U}
    τ_SLDON::U  | 2.0        | yr             | true  | (0,∞)
    τ_RDON::U   | 3500.0     | yr             | true  | (0,∞)
    f1::U       | 0.2        | NoUnits        | true  | (0,1)
    f2::U       | 0.01       | NoUnits        | true  | (0,1)
    σ1::U       | 0.01       | NoUnits        | true  | (0,1)
    α::U        | 0.017      | NoUnits        | true  | (0,∞)
    β::U        | 0.57       | NoUnits        | true  | (0,∞)
    Km::U       | 0.1        | mmol/m^3       | true  | (0,∞)
    Vm::U       | 1.25       | yr             | false | (0,∞)

end

function prior(::Type{T}, s::Symbol) where {T<:AbstractParameters}
    if flattenable(T, s)
        lb, ub = limits(T, s)
        if (lb, ub) == (0,∞)
            μ = log(initial_value(T, s))
            LogNormal(μ, 1.0)
        elseif (lb, ub) == (-∞,∞)
            μ = initial_value(T, s)
            σ = 10.0 # Assumes that a sensible unit is chosen (i.e., that within 10.0 * U)
            Distributions.Normal(μ, σ)
        else # LogitNormal with median as initial value and bounds
            m = initial_value(T, s)
            f = (m - lb) / (ub - lb)
            LocationScale(lb, ub - lb, LogitNormal(log(f/(1-f)), 1.0))
        end
    else
        nothing
    end
end

prior(::T, s::Symbol) where {T<:AbstractParameters} = prior(T,s)
prior(::Type{T}) where {T<:AbstractParameters} = Tuple(prior(T,s) for s in AIBECS.symbols(T))
prior(::T) where {T<:AbstractParameters} = prior(T)

p = DONmodelParameters()

nb = sum(iswet(grd))
F = AIBECSFunction((T_DON,  T_DON), (G_SLDON,G_RDON), nb, DONmodelParameters)

x = ustrip(upreferred(2.0mmol/m^3)) * ones(2nb) # initial guess
prob = SteadyStateProblem(F, x, p)



# load observations

const obs = let
    obs = DataFrame(CSV.File("example_DON_dataset.csv")) # please replace "example_DON_dataset" with your DON concentration dataset
    obs = filter(:DON => <(6.5), obs)
    obs = filter(:DON => >(1), obs)

    obs.value = ustrip.(upreferred.(obs.DON))
    (obs, )
end



modify(SLDON,RDON) = (SLDON+RDON,)

ωs = (1.0,)     # the weight for the mismatch 
ωp = 1e-4       # the weight for the parameters prior estimates
f, ∇ₓf = f_and_∇ₓf(ωs, ωp, grd, modify, obs, DONmodelParameters)


using F1Method
using Distributions


λ = p2λ(p)

τ = ustrip(u"s", 1e3u"Myr")
mem = F1Method.initialize_mem(F, ∇ₓf, x, λ, CTKAlg(), τstop=τ)

function objective(λ)
    p = λ2p(DONmodelParameters, λ) ; @show p
    F1Method.objective(f, F, mem, λ, CTKAlg(), τstop=τ)
end

# calculate gradient and hessian 
gradient(λ) = F1Method.gradient(f, F, ∇ₓf, mem, λ, CTKAlg(), τstop=τ)
hessian(λ) = F1Method.hessian(f, F, ∇ₓf, mem, λ, CTKAlg(), τstop=τ)

# optimize 
using Optim

opt = Optim.Options(store_trace=false, show_trace=true, extended_trace=false, g_tol=1e-5)


res = optimize(objective, gradient, hessian, λ, NewtonTrustRegion(),opt; inplace=false)


# solve the steady state equation with new p
p_optimized = λ2p(DONmodelParameters, res.minimizer)
prob_optimized = SteadyStateProblem(F, x, p_optimized)
s_optimized = solve(prob_optimized, CTKAlg(), τstop=ustrip(s, 1e3Myr)).u

SLDON, RDON = state_to_tracers(s_optimized, grd)
DON=SLDON+RDON

hess = hessian(res.minimizer)


# save outputs 
using JLD2
tp_opt = AIBECS.table(p_optimized)

jldsave("DON2pooloutput_2024.jld2";  SLDON=SLDON, RDON=RDON, DON=DON,tp_opt = tp_opt, hess = hess)


matwrite("DON2pooloutput_2024.mat", Dict(
    "lamda" => res.minimizer.+0, # optimized parameters
	"DON" => DON,
    "RDON" =>RDON.+0,
    "SLDON" =>SLDON.+0,
    "hess" =>hess
))


println("Done!")
