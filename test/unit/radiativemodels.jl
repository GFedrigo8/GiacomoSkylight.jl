using Skylight, Test

@testset "Circular hot spot" begin
    model = CircularHotSpot(
        star_radius_in_km = 1e-5*geometrized_to_CGS(5.0, Dimensions.length, M1 = 1.4),
        spin_frequency_in_Hz = geometrized_to_CGS(0.05/(2π), Dimensions.frequency, M1 = 1.4),
        center_colatitude_in_degrees = 90.0,
        angular_radius_in_radians = deg2rad(60.0),
        M1 = 1.4,
        temperature_in_keV = 0.35)

    spacetime = MinkowskiSpacetimeCartesianCoordinates()
    coords_top = CartesianTopology()
    points = Skylight.space_positions(10, spacetime, model, coords_top, nothing)

    for i in 1:10
        point = points[:, i]

        @test sum(point .* point) ≈ 25
        @test point[1] >= 5 * cos(π / 3)
    end

    vector = zeros(4)
    position = [rand(), 3.0, 0.0, 4.0]
    gμν = [-1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]

    model_cache = allocate_cache(model)
    rest_frame_four_velocity!(vector,
        position,
        gμν,
        spacetime,
        model,
        coords_top,
        model_cache)

    @test vector ≈ [1.0 / sqrt(0.9775), 0.0, 0.15 / sqrt(0.9775), 0.0]

    df = zeros(4)
    surface_differential!(df, position, model, coords_top)
    @test df == [0.0, 2 * position[2], 2 * position[3], 2 * position[4]]

    spacetime = KerrSpacetimeKerrSchildCoordinates(M = 1.0, a = 0.5)

    metric = zeros(4, 4)
    metric_inverse = zeros(4, 4)

    metric!(metric, position, spacetime)
    metric_inverse!(metric_inverse, position, spacetime, metric, nothing)

    normal = zeros(4)
    Skylight.unit_surface_normal!(normal,
        position,
        metric,
        metric_inverse,
        model,
        coords_top)

    tangent_vector = [0.0, -position[3], position[2], 0.0]

    @test Skylight.norm_squared(normal, metric) ≈ 1.0
    @test Skylight.scalar_product(normal, tangent_vector, metric)≈0.0 atol=1e-16

    vector = Skylight.lower_index(normal, metric)
    @test vector[2] / df[2] ≈ vector[4] / df[4]
end

@testset "Composite radiative model" begin
    spacetime = SchwarzschildSpacetimeSphericalCoordinates(M = 1.0)
    coords_top = coordinates_topology(spacetime)

    volume_model = DummyExtendedRegion()
    surface_model = DummyDisk(inner_radius = 6.0, outer_radius = 20.0)
    model = CompositeRadiativeModel(volume_model = volume_model, surface_model = surface_model)

    @test isa(Skylight.isvacuum(model), Skylight.NonVacuum)
    @test Skylight.has_emitting_surface(model)
    @test is_final_position_at_source([0.0, 10.0, π / 2, 0.0], spacetime, model)
    @test !is_final_position_at_source([0.0, 30.0, π / 2, 0.0], spacetime, model)

    gμν = metric([0.0, 10.0, π / 2, 0.0], spacetime)
    volume_velocity = zeros(4)
    rest_frame_four_velocity!(volume_velocity,
        [0.0, 10.0, π / 2, 0.0],
        gμν,
        spacetime,
        model,
        coords_top)
    @test volume_velocity == [1.0, 0.0, 0.0, 0.0]

    cache = allocate_cache(model)
    @test cache isa Skylight.CompositeRadiativeModelCache

    camera = ImagePlane(distance = 100.0,
        observer_inclination_in_degrees = 45.0,
        horizontal_side = 1.0,
        vertical_side = 1.0,
        horizontal_number_of_pixels = 1,
        vertical_number_of_pixels = 1)
    configurations = NonVacuumOTEConfigurations(spacetime = spacetime,
        radiative_model = model,
        camera = camera,
        observation_energies = [1e-8, 2e-8],
        unit_mass_in_solar_masses = 1.0)
    cbp = callback_parameters(spacetime, model, configurations; rhorizon_bound = 0.1)
    @test cbp isa Skylight.BlackHoleAccretionDiskCallbackParameters

    initial_data = zeros(12, 1)
    output_data = zeros(12, 1)

    pi = [0.0, 100.0, π / 2, 0.0]
    pf = [0.0, 10.0, π / 2, 0.0]
    energy_scale = sqrt(1.0 - 2.0 / pi[2])
    ki = [-1.0 / energy_scale, -energy_scale, 0.0, 0.0]
    kf = [-energy_scale / (1.0 - 2.0 / pf[2]), -energy_scale, 0.0, 0.0]
    τ = [0.4, 0.8]
    J = [1.2, 0.6]

    transfer_cache = Skylight.transfer_cache(configurations, cbp, 10.0)
    dτ, dI = Skylight.transfer_equations(vcat(pf, kf, zeros(length(configurations.observation_energies))),
        transfer_cache,
        0.0)
    @test length(dτ) == length(configurations.observation_energies)
    @test length(dI) == length(configurations.observation_energies)
    @test all(isfinite, dτ)
    @test all(isfinite, dI)

    initial_data[1:4, 1] .= pi
    initial_data[5:8, 1] .= ki
    output_data[1:4, 1] .= pf
    output_data[5:8, 1] .= kf
    output_data[9:10, 1] .= τ
    output_data[11:12, 1] .= J

    Iobs = observed_specific_intensities(initial_data, output_data, configurations)

    observer_metric = metric(pi, spacetime)
    emitter_metric = metric(pf, spacetime)
    observer_four_velocity = static_four_velocity(observer_metric)
    surface_four_velocity = zeros(4)
    rest_frame_four_velocity!(surface_four_velocity,
        pf,
        emitter_metric,
        spacetime,
        surface_model,
        coords_top)
    q = scalar_product(ki, observer_four_velocity, observer_metric) /
        scalar_product(kf, surface_four_velocity, emitter_metric)
    expected = similar(configurations.observation_energies)
    for (i, energy) in enumerate(configurations.observation_energies)
        expected[i] = energy^3 * J[i] +
                      exp(-τ[i]) * q^3 *
                      rest_frame_specific_intensity(pf,
            -kf,
            energy / q,
            surface_four_velocity,
            emitter_metric,
            spacetime,
            surface_model,
            coords_top)
    end
    @test Iobs[:, 1] ≈ expected

    pinhole = PinholeCamera(position = pi,
        horizontal_aperture_in_degrees = 1.0,
        vertical_aperture_in_degrees = 1.0,
        horizontal_number_of_pixels = 1,
        vertical_number_of_pixels = 1)
    pinhole_configurations = NonVacuumOTEConfigurations(spacetime = spacetime,
        radiative_model = model,
        camera = pinhole,
        observation_energies = [1e-8, 2e-8],
        unit_mass_in_solar_masses = 1.0)
    Iobs_pinhole = observed_specific_intensities(initial_data,
        output_data,
        pinhole_configurations;
        observer_four_velocity = observer_four_velocity)
    @test Iobs_pinhole[:, 1] ≈ expected
end
