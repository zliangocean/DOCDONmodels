# DOC 2 pool model
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
T_DOC(p) = T_Circ


file = matopen("NPP.mat") # replace NPP.mat with your own NPP data field

NPP = read(file, "NPP") # mmol C m-3 yr-1 


NPP_v = vectorize(NPP["NPP"],grd)     # mmol C m-3 yr-1
NPP_v = ustrip.(upreferred.(NPP_v * u"mmol/m^3/yr")) 


z_top = topdepthvec(grd) # define top layer


# SLDOC remineralization
function R_SLDOC(x,p)
    @unpack τ_SLDOC = p
     @. x / τ_SLDOC
end

# RDOC remineralization
function R_RDOC(x,p)
    @unpack τ_RDOC = p
      @. x / τ_RDOC
end


# RDOC production via SLDOC transformation
function U_RDOC(x,p)
    @unpack σ1,τ_SLDOC = p
    @.  σ1* x / τ_SLDOC
end

# SLDOC production from NPP
function S_SLDOC(p)
    @unpack α, β,f1 = p
    @.  0.5*f1*α*NPP_v^β*(z_top≤73)
end

# RDOC production from NPP
function S_RDOC(p)
    @unpack α, β,f2 = p
    @. 0.5*f2*α*NPP_v^β*(z_top≤73) 
end


# combine right hand equations
function G_SLDOC(SLDOC,RDOC,p)
    - R_SLDOC(SLDOC,p) + S_SLDOC(p) +0*RDOC
end

# combine right hand equations

function G_RDOC(SLDOC,RDOC,p)
    U_RDOC(SLDOC,p)  - R_RDOC(RDOC,p) + S_RDOC(p)
end


import AIBECS: @units, units
import AIBECS: @initial_value, initial_value
import AIBECS: @limits, limits
import AIBECS: @flattenable, flattenable
using Unitful: m, d, s, yr, Myr, mol, mmol, μmol, μM

∞ = Inf
@initial_value @units @flattenable @limits struct DOCmodelParameters{U} <: AbstractParameters{U}
    τ_SLDOC::U  | 5.0           | yr             | true  | (0,∞)
    τ_RDOC::U   | 5000.0        | yr             | true  | (0,∞)
    f1::U       | 0.2           | NoUnits        | true  | (0,1)
    f2::U       | 0.05          | NoUnits        | true  | (0,1)
    σ1::U       | 0.1           | NoUnits        | true  | (0,1)
    α::U        | 0.017         | NoUnits        | true  | (0,∞)
    β::U        | 0.57          | NoUnits        | true  | (0,∞)

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

p = DOCmodelParameters()

nb = sum(iswet(grd))
F = AIBECSFunction(( T_DOC, T_DOC), (G_SLDOC,G_RDOC), nb, DOCmodelParameters)

x = ustrip(upreferred(35.0mmol/m^3)) * ones(2nb) # initial guess
prob = SteadyStateProblem(F, x, p)



# load observations
const obs = let
    obs = DataFrame(CSV.File("example_DOC_dataset.csv")) # please replace "example_DOC_dataset" with you DOC concentration dataset
    obs = filter(:DOC => <(100), obs)

    obs.value = ustrip.(upreferred.(obs.DOC))
    (obs, )
end



modify(SLDOC,RDOC) = (SLDOC+RDOC,)

ωs = (1.0,) # the weight for the mismatch 
ωp = 1e-4       # the weight for the parameters prior estimates
f, ∇ₓf = f_and_∇ₓf(ωs, ωp, grd, modify, obs, DOCmodelParameters)


using F1Method
using Distributions




λ = p2λ(p)

τ = ustrip(u"s", 1e3u"Myr")
mem = F1Method.initialize_mem(F, ∇ₓf, x, λ, CTKAlg(), τstop=τ)

function objective(λ)
    p = λ2p(DOCmodelParameters, λ) ; @show p
    F1Method.objective(f, F, mem, λ, CTKAlg(), τstop=τ)
end

gradient(λ) = F1Method.gradient(f, F, ∇ₓf, mem, λ, CTKAlg(), τstop=τ)
hessian(λ) = F1Method.hessian(f, F, ∇ₓf, mem, λ, CTKAlg(), τstop=τ)


using Optim

opt = Optim.Options(store_trace=false, show_trace=true, extended_trace=false, g_tol=1e-5)


res = optimize(objective, gradient, hessian, λ, NewtonTrustRegion(),opt; inplace=false)


# solve the steady state equation with new p
p_optimized = λ2p(DOCmodelParameters, res.minimizer)
prob_optimized = SteadyStateProblem(F, x, p_optimized)
s_optimized = solve(prob_optimized, CTKAlg(), τstop=ustrip(s, 1e3Myr)).u

SLDOC, RDOC = state_to_tracers(s_optimized, grd)
DOC=SLDOC+RDOC

hess = hessian(res.minimizer)

# save output
using JLD2
tp_opt = AIBECS.table(p_optimized)

jldsave("DOC2pooloutput_2024.jld2"; SLDOC=SLDOC, RDOC=RDOC, DOC=DOC,tp_opt = tp_opt, hess = hess)


matwrite("DOC2pooloutput_2024.mat", Dict(
    "lamda" => res.minimizer.+0, # optimized parameters
	"DOC" => DOC,
    "RDOC" =>RDOC.+0,
    "SLDOC" =>SLDOC.+0,
    "hess" =>hess
))


println("Done!")

