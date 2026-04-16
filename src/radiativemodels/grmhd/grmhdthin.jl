@with_kw struct GRMHDThin{T} <: AbstractRadiativeModel
	filename::String
	fρ::Float64 #Normalization factor for density (Eddington fraction)
	cut_radius::Float64 #Radius and height of the cylinder used to select data
	BHBMass = 1e6 #Binary total mass in solar masses
	μe::Float64 = 1.14 #Mean molecular weight per electron
	μi::Float64 = 1.3 #Mean molecular weight per ion
	μp::Float64 = 0.59 #Mean molecular weight per particle
	gamma::Float64 = 5.0/3.0 #EoS adiabatic index
	Knbg::Int = 12 #Target number of particles in a octree's leaf
	interp::T = build_interpolator(filename,fρ,cut_radius)
	sync::Synchrotron = Synchrotron()
end

struct GRMHDParticlesData
	pos::Vector{Float64,3}
	D::Vector{Float64}
	ρ::Vector{Float64}
	u::Vector{Float64}
	mass::Vector{Float64}
	vel::Vector{Float64,3}
	B::Vector{Float64,3}
	hmin::Float64
end


struct GRMHDPointData
	ρ::Float64
	u::Float64
	velx::Float64
	vely::Float64
	velz::Float64
	Bx::Float64
	By::Float64
	Bz::Float64
end

function build_interpolator(filename,fρ,cut_radius)
	data = read_data(filename,fρ,cut_radius)
	octree = Octree(data.pos,data.hmin,Knbg)
	interp = position -> find_mean_quantities(data,octree,position)
	return interp
end

function v_and_B_phys!(pd::GRMHDPointData,g,spacetimecache)
	gdet = sqrt(-det4x4sym(g))
	gcon = spacetimecache.ginv
	inv4x4sym!(gcon,g)
	α = -1/gcon[1,1] #Lapse squared
	β = [gcon[1,2]*α,gcon[1,3]*α,gcon[1,4]*α] #Shift vector
	α = sqrt(α) #Lapse function
	γdet = gdet/α #sqrt of the determinant of the spatial metric
	pd.velx += β[1]
	pd.vely += β[2]
	pd.velz += β[3]
	pd.velx /= α
	pd.vely /= α
	pd.velz /= α
	pd.Bx /= γdet
	pd.By /= γdet
	pd.Bz /= γdet
	return nothing
end

function convert_quantities_cgs!(pd::GRMHDPointData,model::GRMHDThin)
	Munit = model.BHBMass*PhysicalConstants.M_sun
	Lunit = Munit*PhysicalConstants.G
	pd.ρ *= Munit/(Lunit^3)
	pd.u *= PhysicalConstants.c2
	pd.velx *= PhysicalConstants.c
	pd.vely *= PhysicalConstants.c
	pd.velz *= PhysicalConstants.c
	pd.Bx *= PhysicalConstants.c*sqrt(Munit/(Lunit^3))
	pd.By *= PhysicalConstants.c*sqrt(Munit/(Lunit^3))
	pd.Bz *= PhysicalConstants.c*sqrt(Munit/(Lunit^3))
	return nothing
end

function number_densities(pd::GRMHDPointData, model::GRMHDThin)
	mp = PhysicalConstants.mp
	μe = model.μe
	μi = model.μi
	return pd.ρ/μe/mp, pd.ρ/μi/mp
end

function electron_temperature(pd::GRMHDPointData,model::GRMHDThin)
	mp = PhysicalConstants.mp
	kB = PhysicalConstants.k_B
	μp = model.μp
	gamma = model.gamma
	P = pd.ρ*pd.u*(gamma-1)
	T = P*μp*mp/kB/pd.ρ
	return T
end

function read_data(filename,fρ,cut_radius)
	f=h5open(filename,"r")
	pos = f["PartType0/Coordinates"][:,:]
	u = f["PartType0/InternalEnergy"][:]
	vel = f["PartType0/Velocities"][:,:]
	B = f["PartType0/MagneticField"][:,:]
	mass = f["PartType0/Masses"][:]
	ρ = f["PartType0/RestMassDensity"][:]
	h = f["PartType0/SmoothingLength"][:]
	D = f["PartType0/Density"][:]
	posBH = f["PartType5/Coordinates"][:,:]
	mass .*= fρ
	ρ .*= fρ
	D .*= fρ
	B .*= sqrt(factor)
	mask = findall(((pos[1,:].^2 .+ pos[2,:].^2) .< cut_radius^2) .& (abs.(pos[3,:]) .< cut_radius))
	pos = pos[:,mask]
	u=u[mask]
	vel=vel[:,mask]
	B=B[:,mask]
	mass=mass[mask]
	h=h[mask]
	D=D[mask]
	ρ=ρ[mask]
	return GRMHDParticleData(pos,D,ρ,u,mass,vel,B,minimum(h)/2)
end


function rest_frame_emissivity!(jε,
	position,
	ε,
	g,
	spacetime,
	spacetimecache,
	model::GRMHDThin,
	coords_top)
	sy=model.sync
	pointdata=model.interp(position)
	v_and_B_phys!(pointdata,g,spacetimecache)
	convert_quantities_cgs!(pointdata,model)
	ne,ni = number_densities(pointdata,model)
	Te = electron_temperature(pointdata,model)
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
	spacetimecache,
        model::GRMHDThin,
        coords_top)
        sy=model.sync
        pointdata=model.interp(position)
	v_and_B_phys!(pointdata,g,spacetimecache)
	convert_quantities_cgs!(pointdata,model)
        ne,ni = number_densities(pointdata,model)
        Te = electron_temperature(pointdata,model)
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
    ρ_mean = 0.0
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
        ker = kernel(q,h_here)
        w = mass[i]*ker/D[i]
        norm += w
        ρ_mean += mass[i]*ker
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
    return GRMHDPointData(ρ_mean,u_mean,vx_mean,vy_mean,vz_mean,Bx_mean,By_mean,Bz_mean)
end
