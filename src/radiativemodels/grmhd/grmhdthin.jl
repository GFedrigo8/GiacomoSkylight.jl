@with_kw struct GRMHDThin{T} <: AbstractRadiativeModel
	filename::String
	frho::Float64 #Normalization factor for density (Eddington fraction)
	cut_radius::Float64
	mu_e::Float64 = 1.14 #Mean molecular weight per electron
	mu_i::Float64 = 1.3 #Mean molecular weight per ion
	mu_p::Float64 = 0.69 #Mean molecular weight per particle
	gamma::Float64 = 5.0/3.0
	Knbg::Int = 12
	interp::T = build_interpolator(filename,frho,cut_radius)
	sync::Synchrotron = Synchrotron()
end

struct GRMHDParticlesData
	pos::Vector{Float64,3}
	D::Vector{Float64}
	rho::Vector{Float64}
	u::Vector{Float64}
	mass::Vector{Float64}
	vel::Vector{Float64,3}
	B::Vector{Float64,3}
	hmin::Float64
end


struct GRMHDPointData
	rho::Float64
	u::Float64
	velx::Float64
	vely::Float64
	velz::Float64
	Bx::Float64
	By::Float64
	Bz::Float64
end

function number_densities(p::GRMHDPointData, model::GRMHDThin)
	mp = Constants.mp
	mu_e = model.mu_e
	mu_i = model.mu_i
	return GRMHDPointData.rho/mu_e/mp, GRMHDPointData.rho/mu_i/mp
end

function electron_temperature(p::GRMHDPointData,model::GRMHDThin)
	mp = Constants.mp
	kB = Constants.kB
	mu_p = model.mu_p
	gamma = model.gamma
	P = GRMHDPointData.rho*GRMHDPointData.u*(gamma-1)
	T = P*mu_p*mp/kB/GRMHDPointData.rho
	return T
end

#Need to convert v and B to physical values

function read_data(filename,frho,cut_radius)
	f=h5open(filename,"r")
	pos = f["PartType0/Coordinates"][:,:]
	u = f["PartType0/InternalEnergy"][:]
	vel = f["PartType0/Velocities"][:,:]
	B = f["PartType0/MagneticField"][:,:]
	mass = f["PartType0/Masses"][:]
	rho = f["PartType0/RestMassDensity"][:]
	h = f["PartType0/SmoothingLength"][:]
	D = f["PartType0/Density"][:]
	posBH = f["PartType5/Coordinates"][:,:]
	mass .*= frho
	rho .*= frho
	D .*= frho
	B .*= sqrt(factor)
	mask = findall(((pos[1,:].^2 .+ pos[2,:].^2) .< cut_radius^2) .& (abs.(pos[3,:]) .< cut_radius))
	pos = pos[:,mask]
	u=u[mask]
	vel=vel[:,mask]
	B=B[:,mask]
	mass=mass[mask]
	h=h[mask]
	D=D[mask]
	rho=rho[mask]
	return GRMHDParticleData(pos,D,rho,u,mass,vel,B,minimum(h)/2)
end


function rest_frame_emissivity!(jε,
	position,
	ε,
	g,
	spacetime,
	model::GRMHDThin,
	coords_top)
	sy=model.sync
	pointdata=model.interp(position)
	ne,ni = number_densities(pointdata,model)
	Te = electron_temperature(pointdata,model)
#CONVERT B and Vel
	α = sy.α(Te)
	β = sy.β(Te)
	γ = sy.γ(Te)
	@inbounds begin
	        for (i, εk) in enumerate(ε)
        	    jε[i] = synchrotron_emissivity(εk, ne, Te, B, α, β, γ) + bremmstrahlung_emissivity(εk, ne, ni, Te)
	        end
	    end
	return nothing
end

function rest_frame_absorptivity!(aε,
        position,
        ε,
        g,
        spacetime,
        model::GRMHDThin,
        coords_top)
        sy=model.sync
        pointdata=model.interp(position)
        ne,ni = number_densities(pointdata,model)
        Te = electron_temperature(pointdata,model)
#CONVERT B and Vel  
        α = sy.α(Te)
        β = sy.β(Te)
        γ = sy.γ(Te)   
        @inbounds begin
                for (i, εk) in enumerate(ε)
                    jε = synchrotron_emissivity(εk, ne, Te, B, α, β, γ) + bremmstrahlung_emissivity(εk, ne, ni, Te)
		    aε[i] = jε/planck_function(εk, Te)
                end
            end
	return nothing
end 

function build_interpolator(filename,frho,cut_radius)
	data = read_data(filename,frho,cut_radius)
	octree = Octree(data.pos,data.hmin,Knbg)
	interp = position -> find_mean_quantities(data,octree,position)
	return interp
end

function kernel(q::Float64,h::Float64) #q: coordinate; h: kernel size; return value of the kernel at distance q fro$
    if q<=0.5
        return (1 + (6*q^2)*(q-1)) * 8.0 / π / h^3
    elseif q<= 1.0
        return (2*((1-q)^3)) * 8.0 / π / h^3
    else
        return 0.0
    end
end


function find_mean_quantities(data,octree,position)
    width,_=find_cell(octree,position)
    h_here = sqrt(3)*width[1]/2
    nn = knn(octree, position, h_here) |> collect
    rho_mean = 0.0
    vx_mean = 0.0
    vy_mean = 0.0
    vz_mean = 0.0
    Bx_mean = 0.0
    By_mean = 0.0
    Bz_mean = 0.0
    u_mean = 0.0
    norm = 0.0
    @. q = sqrt((position[2] - pos[1,:])^2 + (position[3] - pos[2,:])^2 + (position[4] - pos[3,:])^2) /h_here
    for i in nn
        m_i = mass[i]
        D_i = D[i]
        ker = kernel(q,h_here)
        w = m_i*ker/D_i
        norm += w
        rho_mean += m_i*ker
        u_mean += w*u[i]
        vx_mean += w*vel[1,i]
        vy_mean += w*vel[2,i]
        vz_mean += w*vel[3,i]
        Bx_mean += w*B[1,i]
        By_mean += w*B[2,i]
        Bz_mean += w*B[3,i]
    end
    u_mean /= norm
    vx_mean /= norm
    vy_mean /= norm
    vz_mean /= norm
    Bx_mean /= norm
    By_mean /= norm
    Bz_mean /= norm
    return GRMHDPointData(rho_mean,u_mean,vx_mean,vy_mean,vz_mean,Bx_mean,By_mean,Bz_mean)
end
