abstract type AbstractSKSSpacetime <: AbstractSpacetime end

coordinates_topology(::AbstractSKSSpacetime) = CartesianTopology()
radius(position, ::AbstractSKSSpacetime) = sqrt(position[2]^2 + position[3]^2 + position[4]^2)

total_mass(spacetime::AbstractSKSSpacetime) = spacetime.m[1] + spacetime.m[2]
mass_ratio(spacetime::AbstractSKSSpacetime) = spacetime.m[2]/spacetime.m[1]
reduced_mass(spacetime::AbstractSKSSpacetime) = spacetime.m[1] * spacetime.m[2] / total_mass(spacetime)
symmetric_mass_ratio(spacetime::AbstractSKSSpacetime) = reduced_mass(spacetime) / total_mass(spacetime)
mass_difference(spacetime::AbstractSKSSpacetime) = spacetime.m[1] - spacetime.m[2]
mass_relative_difference(spacetime::AbstractSKSSpacetime) = mass_difference(spacetime) / total_mass(spacetime)


@with_kw struct SKSSpacetime <: AbstractSKSSpacetime
    m::Vector{Float64} #Masses of the two BHs
    pos1::Vector{Float64} #Position of the first BH
    vel1::Vector{Float64} #Velocity of the first BH
    chi1::Vector{Float64} #Spin of the first BH
    pos2::Vector{Float64} #Position of the second BH
    vel2::Vector{Float64} #Velocity of the second BH
    chi2::Vector{Float64} #Spin of the second BH
    @assert all(m.>=0.0)
    @assert sqrt(chi1[1]^2 + chi1[2]^2 + chi1[3]^2) <= 1.0
    @assert sqrt(chi2[1]^2 + chi2[2]^2 + chi2[3]^2) <= 1.0
end

function metric!(g::AbstractMatrix, position::AbstractVector, spacetime::SKSSpacetime)

    # Extra parameters for regularization
    AST_a1_buffer = 0.01
    AST_a2_buffer = 0.01
    AST_cutoff_floor = 0.1
    AST_adjust_mass1 = 1
    AST_adjust_mass2 = 1

    # Load the evaluation point position
    _,x,y,z = position

    # Load the BHs properties
    # BHs positions
    xi1x = spacetime.pos1[1]
    xi1y = spacetime.pos1[2]
    xi1z = spacetime.pos1[3]
    xi2x = spacetime.pos2[1]
    xi2y = spacetime.pos2[2]
    xi2z = spacetime.pos2[3]

    # BHs velocities
    v1x  = spacetime.vel1[1] + 1e-40
    v1y  = spacetime.vel1[2] + 1e-40
    v1z  = spacetime.vel1[3] + 1e-40
    v2x =  spacetime.vel2[1] + 1e-40
    v2y =  spacetime.vel2[2] + 1e-40
    v2z =  spacetime.vel2[3] + 1e-40

    v2  =  sqrt( v2x * v2x + v2y * v2y + v2z * v2z )
    v1  =  sqrt( v1x * v1x + v1y * v1y + v1z * v1z )

    # BHs masses
    m1_t = spacetime.m[1]
    m2_t = spacetime.m[2]

    # BHs spins
    a1x  = spacetime.chi1[1]*m1_t
    a1y  = spacetime.chi1[2]*m1_t
    a1z  = spacetime.chi1[3]*m1_t

    a2x =  spacetime.chi2[1]*m2_t
    a2y =  spacetime.chi2[2]*m2_t
    a2z =  spacetime.chi2[3]*m2_t

    a1_t = sqrt( a1x*a1x + a1y*a1y + a1z*a1z + 1e-40)
    a2_t = sqrt( a2x*a2x + a2y*a2y + a2z*a2z + 1e-40)

    # Load coordinates
    oo1= v1 * v1
    oo2 = oo1 * -1
    oo3 = 1.0 + oo2
    oo4 = sqrt(oo3)
    oo5 = 1 / oo4
    oo6 = x * -1
    oo7 = oo6 + xi1x
    oo8 = v1x * oo7
    oo9 = y * -1
    oo10 = z * -1
    oo11 = v2 * v2
    oo12 = oo11 * -1
    oo13 = 1 + oo12
    oo14 = sqrt(oo13)
    oo15 = 1 / oo14
    oo16 = oo6 + xi2x
    oo17 = v2x * oo16
    oo18 = xi1x * -1
    oo19 = 1 / oo1
    oo20 = -1 + oo4
    oo21 = xi1y * -1
    oo22 = xi1z * -1
    oo23 = xi2x * -1
    oo24 = 1 / oo11
    oo25 = -1 + oo14
    oo26 = xi2y * -1
    oo27 = xi2z * -1
    oo28 = xi1y * v1y
    oo29 = xi1z * v1z
    oo30 = v1y * (y * -1)
    oo31 = v1z * (z * -1)
    oo32 = oo28 + (oo29 + (oo30 + (oo31 + oo8)))
    oo33 = xi2y * v2y
    oo34 = xi2z * v2z
    oo35 = v2y * (y * -1)
    oo36 = v2z * (z * -1)
    oo37 = oo17 + (oo33 + (oo34 + (oo35 + oo36)))
    x0BH1 = (oo8 + ((oo9 + xi1y) * v1y + (oo10 + xi1z) * v1z)) * oo5
    x0BH2 = (oo17 + ((oo9 + xi2y) * v2y + (oo10 + xi2z) * v2z)) * oo15
    x1BH1 = (oo18 + x) - oo20 * (oo5 * (v1x * (((oo18 + x) * v1x + ((oo21 + y) * v1y + (oo22 + z) * v1z)) * oo19)))
    x1BH2 = (oo23 + x) - oo24 * (oo25 * (v2x * (((oo23 + x) * v2x + ((oo26 + y) * v2y + (oo27 + z) * v2z)) * oo15)))
    x2BH1 = oo21 + (oo20 * (oo32 * (oo5 * (v1y * oo19))) + y)
    x2BH2 = oo26 + (oo24 * (oo25 * (oo37 * (v2y * oo15))) + y)
    x3BH1 = oo22 + (oo20 * (oo32 * (oo5 * (v1z * oo19))) + z)
    x3BH2 = oo27 + (oo24 * (oo25 * (oo37 * (v2z * oo15))) + z)

    # Adjust mass
    # This is useful for reducing the effective mass of each BH
    # Adjust by hand to get the correct irreducible mass of the BH
    a1 = a1_t * AST_adjust_mass1
    m1 = m1_t * AST_adjust_mass1
    a2 = a2_t * AST_adjust_mass2
    m2 = m2_t * AST_adjust_mass2

    #//============================================//
    #// Regularize horizon and apply excision mask //
    #//============================================//

    # Define radius with respect to BH frame
    rBH1 = sqrt( x1BH1*x1BH1 + x2BH1*x2BH1 + x3BH1*x3BH1)
    rBH2 = sqrt( x1BH2*x1BH2 + x2BH2*x2BH2 + x3BH2*x3BH2)

    # Define radius cutoff
    rBH1_Cutoff = abs(a1) * ( 1.0 + AST_a1_buffer) + AST_cutoff_floor
    rBH2_Cutoff = abs(a2) * ( 1.0 + AST_a2_buffer) + AST_cutoff_floor

    # Apply excision
    if rBH1 < rBH1_Cutoff if x3BH1>0 x3BH1 = rBH1_Cutoff else x3BH1 = -1.0*rBH1_Cutoff end end
    if rBH2 < rBH2_Cutoff if x3BH2>0 x3BH1 = rBH1_Cutoff else x3BH2 = -1.0*rBH2_Cutoff end end

    #//=================//
    #//     Metric      //
    #//=================//
    o1 = 1.4142135623730951
    o2 = 1 / o1
    o3 = a1x * a1x
    o4 = o3 * -1
    o5 = a1z * a1z
    o6 = o5 * -1
    o7 = a2x * a2x
    o8 = o7 * -1
    o9 = x1BH1 * x1BH1
    o10 = x2BH1 * x2BH1
    o11 = x3BH1 * x3BH1
    o12 = x1BH1 * a1x
    o13 = x2BH1 * a2x
    o14 = x3BH1 * a1z
    o15 = o12 + (o13 + o14)
    o16 = o15 * o15
    o17 = o16 * 4
    o18 = o10 + (o11 + (o4 + (o6 + (o8 + o9))))
    o19 = o18 * o18
    o20 = o17 + o19
    o21 = sqrt(o20)
    o22 = o10 + (o11 + (o21 + (o4 + (o6 + (o8 + o9)))))
    o23 = o22^(1.5)
    o24 = o22 * o22
    o25 = o24 * 0.25
    o26 = o16 + o25
    o27 = 1 / o26
    o28 = x2BH1 * a1z
    o29 = a2x * (x3BH1 * -1)
    o30 = sqrt(o22)
    o31 = 1 / o30
    o32 = o1 * (o15 * (o31 * a1x))
    o33 = o30 * (x1BH1 * o2)
    o34 = o28 + (o29 + (o32 + o33))
    o35 = o22 * 0.5
    o36 = o3 + (o35 + (o5 + o7))
    o37 = 1 / o36
    o38 = o2 * (o23 * (o27 * (o34 * (o37 * m1))))
    o39 = a1z * (x1BH1 * -1)
    o40 = x3BH1 * a1x
    o41 = o1 * (o15 * (o31 * a2x))
    o42 = o30 * (x2BH1 * o2)
    o43 = o39 + (o40 + (o41 + o42))
    o44 = o2 * (o23 * (o27 * (o37 * (o43 * m1))))
    o45 = x1BH1 * a2x
    o46 = a1x * (x2BH1 * -1)
    o47 = o1 * (o15 * (o31 * a1z))
    o48 = o30 * (x3BH1 * o2)
    o49 = o45 + (o46 + (o47 + o48))
    o50 = o2 * (o23 * (o27 * (o37 * (o49 * m1))))
    o51 = o36 * o36
    o52 = 1 / o51
    o53 = o2 * (o23 * (o27 * (o34 * (o43 * (o52 * m1)))))
    o54 = o2 * (o23 * (o27 * (o34 * (o49 * (o52 * m1)))))
    o55 = o2 * (o23 * (o27 * (o43 * (o49 * (o52 * m1)))))
    o56 = a2y * a2y
    o57 = o56 * -1
    o58 = a2z * a2z
    o59 = o58 * -1
    o60 = x1BH2 * x1BH2
    o61 = x2BH2 * x2BH2
    o62 = x3BH2 * x3BH2
    o63 = x1BH2 * a2x
    o64 = x2BH2 * a2y
    o65 = x3BH2 * a2z
    o66 = o63 + (o64 + o65)
    o67 = o66 * o66
    o68 = o67 * 4
    o69 = o57 + (o59 + (o60 + (o61 + (o62 + o8))))
    o70 = o69 * o69
    o71 = o68 + o70
    o72 = sqrt(o71)
    o73 = o57 + (o59 + (o60 + (o61 + (o62 + (o72 + o8)))))
    o74 = o73^(1.5)
    o75 = o73 * o73
    o76 = o75 * 0.25
    o77 = o67 + o76
    o78 = 1 / o77
    o79 = x2BH2 * a2z;
    o80 = a2y * (x3BH2 * -1)
    o81 = sqrt(o73)
    o82 = 1 / o81
    o83 = o1 * (o66 * (o82 * a2x))
    o84 = o81 * (x1BH2 * o2)
    o85 = o79 + (o80 + (o83 + o84))
    o86 = o73 * 0.5
    o87 = o56 + (o58 + (o7 + o86))
    o88 = 1 / o87
    o89 = o2 * (o74 * (o78 * (o85 * (o88 * m2))))
    o90 = a2z * (x1BH2 * -1)
    o91 = x3BH2 * a2x
    o92 = o1 * (o66 * (o82 * a2y))
    o93 = o81 * (x2BH2 * o2)
    o94 = o90 + (o91 + (o92 + o93))
    o95 = o2 * (o74 * (o78 * (o88 * (o94 * m2))))
    o96 = x1BH2 * a2y
    o97 = a2x * (x2BH2 * -1)
    o98 = o1 * (o66 * (o82 * a2z))
    o99 = o81 * (x3BH2 * o2)
    o100 = o96 + (o97 + (o98 + o99))
    o101 = o100 * (o2 * (o74 * (o78 * (o88 * m2))))
    o102 = o87 * o87
    o103 = 1 / o102
    o104 = o103 * (o2 * (o74 * (o78 * (o85 * (o94 * m2)))))
    o105 = o100 * (o103 * (o2 * (o74 * (o78 * (o85 * m2)))))
    o106 = o100 * (o103 * (o2 * (o74 * (o78 * (o94 * m2)))))
    o107 = v1 * v1
    o108 = o107 * -1
    o109 = 1 + o108
    o110 = sqrt(o109)
    o111 = 1 / o110
    o112 = o111 * (v1x * -1)
    o113 = o111 * (v1y * -1)
    o114 = o111 * (v1z * -1)
    o115 = 1 / o107
    o116 = -1 + o111
    o117 = o116 * (v1x * (v1y * o115))
    o118 = o116 * (v1x * (v1z * o115))
    o119 = o116 * (v1y * (v1z * o115))
    o120 = v2 * v2
    o121 = o120 * -1
    o122 = 1 + o121
    o123 = sqrt(o122)
    o124 = 1 / o123
    o125 = o124 * (v2x * -1)
    o126 = o124 * (v2y * -1)
    o127 = o124 * (v2z * -1)
    o128 = 1 / o120
    o129 = -1 + o124
    o130 = o129 * (v2x * (v2y * o128))
    o131 = o129 * (v2x * (v2z * o128))
    o132 = o129 * (v2y * (v2z * o128))
    KS1_0_0 = o2 * (o23 * (o27 * m1))
    KS1_0_1 = o38
    KS1_0_2 = o44
    KS1_0_3 = o50
    KS1_1_0 = o38
    KS1_1_1 = o2 * (o23 * (o27 * ((o34 * o34) * (o52 * m1))))
    KS1_1_2 = o53
    KS1_1_3 = o54
    KS1_2_0 = o44
    KS1_2_1 = o53
    KS1_2_2 = o2 * (o23 * (o27 * ((o43 * o43) * (o52 * m1))))
    KS1_2_3 = o55
    KS1_3_0 = o50
    KS1_3_1 = o54
    KS1_3_2 = o55
    KS1_3_3 = o2 * (o23 * (o27 * ((o49 * o49) * (o52 * m1))))
    KS2_0_0 = o2 * (o74 * (o78 * m2))
    KS2_0_1 = o89
    KS2_0_2 = o95
    KS2_0_3 = o101
    KS2_1_0 = o89
    KS2_1_1 = o103 * (o2 * (o74 * (o78 * ((o85 * o85) * m2))))
    KS2_1_2 = o104
    KS2_1_3 = o105
    KS2_2_0 = o95
    KS2_2_1 = o104
    KS2_2_2 = o103 * (o2 * (o74 * (o78 * ((o94 * o94) * m2))))
    KS2_2_3 = o106
    KS2_3_0 = o101
    KS2_3_1 = o105
    KS2_3_2 = o106
    KS2_3_3 = (o100 * o100) * (o103 * (o2 * (o74 * (o78 * m2))))
    J1_0_0 = o111
    J1_0_1 = o112
    J1_0_2 = o113
    J1_0_3 = o114
    J1_1_0 = o112
    J1_1_1 = 1 + o116 * ((v1x * v1x) * o115)
    J1_1_2 = o117
    J1_1_3 = o118
    J1_2_0 = o113
    J1_2_1 = o117
    J1_2_2 = 1 + o116 * ((v1y * v1y) * o115)
    J1_2_3 = o119
    J1_3_0 = o114
    J1_3_1 = o118
    J1_3_2 = o119
    J1_3_3 = 1 + o116 * ((v1z * v1z) * o115)
    J2_0_0 = o124
    J2_0_1 = o125
    J2_0_2 = o126
    J2_0_3 = o127
    J2_1_0 = o125
    J2_1_1 = 1 + o129 * ((v2x * v2x) * o128)
    J2_1_2 = o130
    J2_1_3 = o131
    J2_2_0 = o126
    J2_2_1 = o130
    J2_2_2 = 1 + o129 * ((v2y * v2y) * o128)
    J2_2_3 = o132
    J2_3_0 = o127
    J2_3_1 = o131
    J2_3_2 = o132
    J2_3_3 = 1 + o129 * ((v2z * v2z) * o128)

    # Initialize Minkowski metric
    g[1,1] = -1.0
    g[2,2] = 1.0
    g[3,3] = 1.0
    g[4,4] = 1.0

    # Evaluate non-flat metric components
    result0 = J2_0_0*J2_0_0*KS2_0_0 + J1_0_0*J1_0_0*KS1_0_0 + J2_0_0*J2_1_0*KS2_0_1 + J1_0_0*J1_1_0*KS1_0_1 + J2_0_0*J2_2_0*KS2_0_2 + J1_0_0*J1_2_0*KS1_0_2 + J2_0_0*J2_3_0*KS2_0_3 + J1_0_0*J1_3_0*KS1_0_3 + J2_1_0*J2_0_0*KS2_1_0 + J1_1_0*J1_0_0*KS1_1_0 + J2_1_0*J2_1_0*KS2_1_1 + J1_1_0*J1_1_0*KS1_1_1 + J2_1_0*J2_2_0*KS2_1_2 + J1_1_0*J1_2_0*KS1_1_2 + J2_1_0*J2_3_0*KS2_1_3 + J1_1_0*J1_3_0*KS1_1_3 + J2_2_0*J2_0_0*KS2_2_0 + J1_2_0*J1_0_0*KS1_2_0 + J2_2_0*J2_1_0*KS2_2_1 + J1_2_0*J1_1_0*KS1_2_1 + J2_2_0*J2_2_0*KS2_2_2 + J1_2_0*J1_2_0*KS1_2_2 + J2_2_0*J2_3_0*KS2_2_3 + J1_2_0*J1_3_0*KS1_2_3 + J2_3_0*J2_0_0*KS2_3_0 + J1_3_0*J1_0_0*KS1_3_0 + J2_3_0*J2_1_0*KS2_3_1 + J1_3_0*J1_1_0*KS1_3_1 + J2_3_0*J2_2_0*KS2_3_2 + J1_3_0*J1_2_0*KS1_3_2 + J2_3_0*J2_3_0*KS2_3_3 + J1_3_0*J1_3_0*KS1_3_3

    result1 = J2_0_0*J2_0_1*KS2_0_0 + J1_0_0*J1_0_1*KS1_0_0 + J2_0_0*J2_1_1*KS2_0_1 + J1_0_0*J1_1_1*KS1_0_1 + J2_0_0*J2_2_1*KS2_0_2 + J1_0_0*J1_2_1*KS1_0_2 + J2_0_0*J2_3_1*KS2_0_3 + J1_0_0*J1_3_1*KS1_0_3 + J2_1_0*J2_0_1*KS2_1_0 + J1_1_0*J1_0_1*KS1_1_0 + J2_1_0*J2_1_1*KS2_1_1 + J1_1_0*J1_1_1*KS1_1_1 + J2_1_0*J2_2_1*KS2_1_2 + J1_1_0*J1_2_1*KS1_1_2 + J2_1_0*J2_3_1*KS2_1_3 + J1_1_0*J1_3_1*KS1_1_3 + J2_2_0*J2_0_1*KS2_2_0 + J1_2_0*J1_0_1*KS1_2_0 + J2_2_0*J2_1_1*KS2_2_1 + J1_2_0*J1_1_1*KS1_2_1 + J2_2_0*J2_2_1*KS2_2_2 + J1_2_0*J1_2_1*KS1_2_2 + J2_2_0*J2_3_1*KS2_2_3 + J1_2_0*J1_3_1*KS1_2_3 + J2_3_0*J2_0_1*KS2_3_0 + J1_3_0*J1_0_1*KS1_3_0 + J2_3_0*J2_1_1*KS2_3_1 + J1_3_0*J1_1_1*KS1_3_1 + J2_3_0*J2_2_1*KS2_3_2 + J1_3_0*J1_2_1*KS1_3_2 + J2_3_0*J2_3_1*KS2_3_3 + J1_3_0*J1_3_1*KS1_3_3

    result2 = J2_0_0*J2_0_2*KS2_0_0 + J1_0_0*J1_0_2*KS1_0_0 + J2_0_0*J2_1_2*KS2_0_1 + J1_0_0*J1_1_2*KS1_0_1 + J2_0_0*J2_2_2*KS2_0_2 + J1_0_0*J1_2_2*KS1_0_2 + J2_0_0*J2_3_2*KS2_0_3 + J1_0_0*J1_3_2*KS1_0_3 + J2_1_0*J2_0_2*KS2_1_0 + J1_1_0*J1_0_2*KS1_1_0 + J2_1_0*J2_1_2*KS2_1_1 + J1_1_0*J1_1_2*KS1_1_1 + J2_1_0*J2_2_2*KS2_1_2 + J1_1_0*J1_2_2*KS1_1_2 + J2_1_0*J2_3_2*KS2_1_3 + J1_1_0*J1_3_2*KS1_1_3 + J2_2_0*J2_0_2*KS2_2_0 + J1_2_0*J1_0_2*KS1_2_0 + J2_2_0*J2_1_2*KS2_2_1 + J1_2_0*J1_1_2*KS1_2_1 + J2_2_0*J2_2_2*KS2_2_2 + J1_2_0*J1_2_2*KS1_2_2 + J2_2_0*J2_3_2*KS2_2_3 + J1_2_0*J1_3_2*KS1_2_3 + J2_3_0*J2_0_2*KS2_3_0 + J1_3_0*J1_0_2*KS1_3_0 + J2_3_0*J2_1_2*KS2_3_1 + J1_3_0*J1_1_2*KS1_3_1 + J2_3_0*J2_2_2*KS2_3_2 + J1_3_0*J1_2_2*KS1_3_2 + J2_3_0*J2_3_2*KS2_3_3 + J1_3_0*J1_3_2*KS1_3_3

    result3 = J2_0_0*J2_0_3*KS2_0_0 + J1_0_0*J1_0_3*KS1_0_0 + J2_0_0*J2_1_3*KS2_0_1 + J1_0_0*J1_1_3*KS1_0_1 + J2_0_0*J2_2_3*KS2_0_2 + J1_0_0*J1_2_3*KS1_0_2 + J2_0_0*J2_3_3*KS2_0_3 + J1_0_0*J1_3_3*KS1_0_3 + J2_1_0*J2_0_3*KS2_1_0 + J1_1_0*J1_0_3*KS1_1_0 + J2_1_0*J2_1_3*KS2_1_1 + J1_1_0*J1_1_3*KS1_1_1 + J2_1_0*J2_2_3*KS2_1_2 + J1_1_0*J1_2_3*KS1_1_2 + J2_1_0*J2_3_3*KS2_1_3 + J1_1_0*J1_3_3*KS1_1_3 + J2_2_0*J2_0_3*KS2_2_0 + J1_2_0*J1_0_3*KS1_2_0 + J2_2_0*J2_1_3*KS2_2_1 + J1_2_0*J1_1_3*KS1_2_1 + J2_2_0*J2_2_3*KS2_2_2 + J1_2_0*J1_2_3*KS1_2_2 + J2_2_0*J2_3_3*KS2_2_3 + J1_2_0*J1_3_3*KS1_2_3 + J2_3_0*J2_0_3*KS2_3_0 + J1_3_0*J1_0_3*KS1_3_0 + J2_3_0*J2_1_3*KS2_3_1 + J1_3_0*J1_1_3*KS1_3_1 + J2_3_0*J2_2_3*KS2_3_2 + J1_3_0*J1_2_3*KS1_3_2 + J2_3_0*J2_3_3*KS2_3_3 + J1_3_0*J1_3_3*KS1_3_3

    result4 = J2_0_1*J2_0_1*KS2_0_0 + J1_0_1*J1_0_1*KS1_0_0 + J2_0_1*J2_1_1*KS2_0_1 + J1_0_1*J1_1_1*KS1_0_1 + J2_0_1*J2_2_1*KS2_0_2 + J1_0_1*J1_2_1*KS1_0_2 + J2_0_1*J2_3_1*KS2_0_3 + J1_0_1*J1_3_1*KS1_0_3 + J2_1_1*J2_0_1*KS2_1_0 + J1_1_1*J1_0_1*KS1_1_0 + J2_1_1*J2_1_1*KS2_1_1 + J1_1_1*J1_1_1*KS1_1_1 + J2_1_1*J2_2_1*KS2_1_2 + J1_1_1*J1_2_1*KS1_1_2 + J2_1_1*J2_3_1*KS2_1_3 + J1_1_1*J1_3_1*KS1_1_3 + J2_2_1*J2_0_1*KS2_2_0 + J1_2_1*J1_0_1*KS1_2_0 + J2_2_1*J2_1_1*KS2_2_1 + J1_2_1*J1_1_1*KS1_2_1 + J2_2_1*J2_2_1*KS2_2_2 + J1_2_1*J1_2_1*KS1_2_2 + J2_2_1*J2_3_1*KS2_2_3 + J1_2_1*J1_3_1*KS1_2_3 + J2_3_1*J2_0_1*KS2_3_0 + J1_3_1*J1_0_1*KS1_3_0 + J2_3_1*J2_1_1*KS2_3_1 + J1_3_1*J1_1_1*KS1_3_1 + J2_3_1*J2_2_1*KS2_3_2 + J1_3_1*J1_2_1*KS1_3_2 + J2_3_1*J2_3_1*KS2_3_3 + J1_3_1*J1_3_1*KS1_3_3

    result5 = J2_0_1*J2_0_2*KS2_0_0 + J1_0_1*J1_0_2*KS1_0_0 + J2_0_1*J2_1_2*KS2_0_1 + J1_0_1*J1_1_2*KS1_0_1 + J2_0_1*J2_2_2*KS2_0_2 + J1_0_1*J1_2_2*KS1_0_2 + J2_0_1*J2_3_2*KS2_0_3 + J1_0_1*J1_3_2*KS1_0_3 + J2_1_1*J2_0_2*KS2_1_0 + J1_1_1*J1_0_2*KS1_1_0 + J2_1_1*J2_1_2*KS2_1_1 + J1_1_1*J1_1_2*KS1_1_1 + J2_1_1*J2_2_2*KS2_1_2 + J1_1_1*J1_2_2*KS1_1_2 + J2_1_1*J2_3_2*KS2_1_3 + J1_1_1*J1_3_2*KS1_1_3 + J2_2_1*J2_0_2*KS2_2_0 + J1_2_1*J1_0_2*KS1_2_0 + J2_2_1*J2_1_2*KS2_2_1 + J1_2_1*J1_1_2*KS1_2_1 + J2_2_1*J2_2_2*KS2_2_2 + J1_2_1*J1_2_2*KS1_2_2 + J2_2_1*J2_3_2*KS2_2_3 + J1_2_1*J1_3_2*KS1_2_3 + J2_3_1*J2_0_2*KS2_3_0 + J1_3_1*J1_0_2*KS1_3_0 + J2_3_1*J2_1_2*KS2_3_1 + J1_3_1*J1_1_2*KS1_3_1 + J2_3_1*J2_2_2*KS2_3_2 + J1_3_1*J1_2_2*KS1_3_2 + J2_3_1*J2_3_2*KS2_3_3 + J1_3_1*J1_3_2*KS1_3_3

    result6 = J2_0_1*J2_0_3*KS2_0_0 + J1_0_1*J1_0_3*KS1_0_0 + J2_0_1*J2_1_3*KS2_0_1 + J1_0_1*J1_1_3*KS1_0_1 + J2_0_1*J2_2_3*KS2_0_2 + J1_0_1*J1_2_3*KS1_0_2 + J2_0_1*J2_3_3*KS2_0_3 + J1_0_1*J1_3_3*KS1_0_3 + J2_1_1*J2_0_3*KS2_1_0 + J1_1_1*J1_0_3*KS1_1_0 + J2_1_1*J2_1_3*KS2_1_1 + J1_1_1*J1_1_3*KS1_1_1 + J2_1_1*J2_2_3*KS2_1_2 + J1_1_1*J1_2_3*KS1_1_2 + J2_1_1*J2_3_3*KS2_1_3 + J1_1_1*J1_3_3*KS1_1_3 + J2_2_1*J2_0_3*KS2_2_0 + J1_2_1*J1_0_3*KS1_2_0 + J2_2_1*J2_1_3*KS2_2_1 + J1_2_1*J1_1_3*KS1_2_1 + J2_2_1*J2_2_3*KS2_2_2 + J1_2_1*J1_2_3*KS1_2_2 + J2_2_1*J2_3_3*KS2_2_3 + J1_2_1*J1_3_3*KS1_2_3 + J2_3_1*J2_0_3*KS2_3_0 + J1_3_1*J1_0_3*KS1_3_0 + J2_3_1*J2_1_3*KS2_3_1 + J1_3_1*J1_1_3*KS1_3_1 + J2_3_1*J2_2_3*KS2_3_2 + J1_3_1*J1_2_3*KS1_3_2 + J2_3_1*J2_3_3*KS2_3_3 + J1_3_1*J1_3_3*KS1_3_3

    result7 = J2_0_2*J2_0_2*KS2_0_0 + J1_0_2*J1_0_2*KS1_0_0 + J2_0_2*J2_1_2*KS2_0_1 + J1_0_2*J1_1_2*KS1_0_1 + J2_0_2*J2_2_2*KS2_0_2 + J1_0_2*J1_2_2*KS1_0_2 + J2_0_2*J2_3_2*KS2_0_3 + J1_0_2*J1_3_2*KS1_0_3 + J2_1_2*J2_0_2*KS2_1_0 + J1_1_2*J1_0_2*KS1_1_0 + J2_1_2*J2_1_2*KS2_1_1 + J1_1_2*J1_1_2*KS1_1_1 + J2_1_2*J2_2_2*KS2_1_2 + J1_1_2*J1_2_2*KS1_1_2 + J2_1_2*J2_3_2*KS2_1_3 + J1_1_2*J1_3_2*KS1_1_3 + J2_2_2*J2_0_2*KS2_2_0 + J1_2_2*J1_0_2*KS1_2_0 + J2_2_2*J2_1_2*KS2_2_1 + J1_2_2*J1_1_2*KS1_2_1 + J2_2_2*J2_2_2*KS2_2_2 + J1_2_2*J1_2_2*KS1_2_2 + J2_2_2*J2_3_2*KS2_2_3 + J1_2_2*J1_3_2*KS1_2_3 + J2_3_2*J2_0_2*KS2_3_0 + J1_3_2*J1_0_2*KS1_3_0 + J2_3_2*J2_1_2*KS2_3_1 + J1_3_2*J1_1_2*KS1_3_1 + J2_3_2*J2_2_2*KS2_3_2 + J1_3_2*J1_2_2*KS1_3_2 + J2_3_2*J2_3_2*KS2_3_3 + J1_3_2*J1_3_2*KS1_3_3

    result8 = J2_0_2*J2_0_3*KS2_0_0 + J1_0_2*J1_0_3*KS1_0_0 + J2_0_2*J2_1_3*KS2_0_1 + J1_0_2*J1_1_3*KS1_0_1 + J2_0_2*J2_2_3*KS2_0_2 + J1_0_2*J1_2_3*KS1_0_2 + J2_0_2*J2_3_3*KS2_0_3 + J1_0_2*J1_3_3*KS1_0_3 + J2_1_2*J2_0_3*KS2_1_0 + J1_1_2*J1_0_3*KS1_1_0 + J2_1_2*J2_1_3*KS2_1_1 + J1_1_2*J1_1_3*KS1_1_1 + J2_1_2*J2_2_3*KS2_1_2 + J1_1_2*J1_2_3*KS1_1_2 + J2_1_2*J2_3_3*KS2_1_3 + J1_1_2*J1_3_3*KS1_1_3 + J2_2_2*J2_0_3*KS2_2_0 + J1_2_2*J1_0_3*KS1_2_0 + J2_2_2*J2_1_3*KS2_2_1 + J1_2_2*J1_1_3*KS1_2_1 + J2_2_2*J2_2_3*KS2_2_2 + J1_2_2*J1_2_3*KS1_2_2 + J2_2_2*J2_3_3*KS2_2_3 + J1_2_2*J1_3_3*KS1_2_3 + J2_3_2*J2_0_3*KS2_3_0 + J1_3_2*J1_0_3*KS1_3_0 + J2_3_2*J2_1_3*KS2_3_1 + J1_3_2*J1_1_3*KS1_3_1 + J2_3_2*J2_2_3*KS2_3_2 + J1_3_2*J1_2_3*KS1_3_2 + J2_3_2*J2_3_3*KS2_3_3 + J1_3_2*J1_3_3*KS1_3_3

    result9 = J2_0_3*J2_0_3*KS2_0_0 + J1_0_3*J1_0_3*KS1_0_0 + J2_0_3*J2_1_3*KS2_0_1 + J1_0_3*J1_1_3*KS1_0_1 + J2_0_3*J2_2_3*KS2_0_2 + J1_0_3*J1_2_3*KS1_0_2 + J2_0_3*J2_3_3*KS2_0_3 + J1_0_3*J1_3_3*KS1_0_3 + J2_1_3*J2_0_3*KS2_1_0 + J1_1_3*J1_0_3*KS1_1_0 + J2_1_3*J2_1_3*KS2_1_1 + J1_1_3*J1_1_3*KS1_1_1 + J2_1_3*J2_2_3*KS2_1_2 + J1_1_3*J1_2_3*KS1_1_2 + J2_1_3*J2_3_3*KS2_1_3 + J1_1_3*J1_3_3*KS1_1_3 + J2_2_3*J2_0_3*KS2_2_0 + J1_2_3*J1_0_3*KS1_2_0 + J2_2_3*J2_1_3*KS2_2_1 + J1_2_3*J1_1_3*KS1_2_1 + J2_2_3*J2_2_3*KS2_2_2 + J1_2_3*J1_2_3*KS1_2_2 + J2_2_3*J2_3_3*KS2_2_3 + J1_2_3*J1_3_3*KS1_2_3 + J2_3_3*J2_0_3*KS2_3_0 + J1_3_3*J1_0_3*KS1_3_0 + J2_3_3*J2_1_3*KS2_3_1 + J1_3_3*J1_1_3*KS1_3_1 + J2_3_3*J2_2_3*KS2_3_2 + J1_3_3*J1_2_3*KS1_3_2 + J2_3_3*J2_3_3*KS2_3_3 + J1_3_3*J1_3_3*KS1_3_3

    # Finally fill the metric tensor
    g[1,1] += result0
    g[2,2] += result4
    g[3,3] += result7
    g[4,4] += result9
    g[1,2] = g[2,1] = result1
    g[1,3] = g[3,1] = result2
    g[1,4] = g[4,1] = result3
    g[2,3] = g[3,2] = result5
    g[2,4] = g[4,2] = result6
    g[3,4] = g[4,3] = result8


    return nothing
end
