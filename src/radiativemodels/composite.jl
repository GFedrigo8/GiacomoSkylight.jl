"""
    CompositeRadiativeModel(volume_model, surface_model)

Combine a non-vacuum volume radiative model with an emitting surface model.

The transfer equations use `volume_model` to evaluate the emissivity, absorptivity,
and bulk rest frame inside the medium. When a ray intersects the wrapped
`surface_model`, the observer-side intensity can be completed with an attenuated
boundary term built from the surface emission.
"""
@with_kw struct CompositeRadiativeModel{
    V <: AbstractRadiativeModel,
    S <: AbstractRadiativeModel,
} <: AbstractRadiativeModel
    volume_model::V
    surface_model::S

    @assert isa(isvacuum(volume_model), NonVacuum) "volume_model must be non-vacuum"
    @assert has_emitting_surface(surface_model) "surface_model must define an emitting surface"
end

struct CompositeRadiativeModelCache{V, S} <: AbstractModelCache
    volume_cache::V
    surface_cache::S
end

isvacuum(::CompositeRadiativeModel) = NonVacuum()
has_emitting_surface(::CompositeRadiativeModel) = true

volume_radiative_model(model::CompositeRadiativeModel) = model.volume_model
surface_radiative_model(model::CompositeRadiativeModel) = model.surface_model

volume_model_cache(::CompositeRadiativeModel, cache::CompositeRadiativeModelCache) = cache.volume_cache
surface_model_cache(::CompositeRadiativeModel, cache::CompositeRadiativeModelCache) = cache.surface_cache

function allocate_cache(model::CompositeRadiativeModel)
    return CompositeRadiativeModelCache(allocate_cache(model.volume_model),
        allocate_cache(model.surface_model))
end

function stationarity(model::CompositeRadiativeModel)
    return is_stationary(model.volume_model) && is_stationary(model.surface_model) ?
           IsStationary() : IsNotStationary()
end

function spherical_symmetry(model::CompositeRadiativeModel)
    return is_spherically_symmetric(model.volume_model) &&
           is_spherically_symmetric(model.surface_model) ?
           IsSphericallySymmetric() : IsNotSphericallySymmetric()
end

function axisymmetry(model::CompositeRadiativeModel)
    return is_axisymmetric(model.volume_model) && is_axisymmetric(model.surface_model) ?
           IsAxisymmetric() : IsNotAxisymmetric()
end

function helical_symmetry(model::CompositeRadiativeModel)
    return is_helically_symmetric(model.volume_model) &&
           is_helically_symmetric(model.surface_model) ?
           IsHelicallySymmetric() : IsNotHelicallySymmetric()
end

function opaque_interior_surface_trait(model::CompositeRadiativeModel)
    return opaque_interior_surface_trait(model.surface_model)
end

function rest_frame_four_velocity!(vector,
    position,
    metric,
    spacetime,
    model::CompositeRadiativeModel,
    coords_top)
    rest_frame_four_velocity!(vector,
        position,
        metric,
        spacetime,
        model.volume_model,
        coords_top)
end

function rest_frame_four_velocity!(vector,
    position,
    metric,
    spacetime,
    model::CompositeRadiativeModel,
    coords_top,
    ::Nothing,
    model_cache::CompositeRadiativeModelCache)
    rest_frame_four_velocity!(vector,
        position,
        metric,
        spacetime,
        model.volume_model,
        coords_top,
        nothing,
        model_cache.volume_cache)
end

function rest_frame_four_velocity!(vector,
    position,
    metric,
    spacetime,
    model::CompositeRadiativeModel,
    coords_top,
    spacetime_cache::AbstractSpacetimeCache,
    model_cache::CompositeRadiativeModelCache)
    rest_frame_four_velocity!(vector,
        position,
        metric,
        spacetime,
        model.volume_model,
        coords_top,
        spacetime_cache,
        model_cache.volume_cache)
end

function rest_frame_absorptivity!(αε,
    position,
    ε,
    metric,
    spacetime,
    model::CompositeRadiativeModel,
    coords_top,
    spacetime_cache,
    model_cache::CompositeRadiativeModelCache)
    rest_frame_absorptivity!(αε,
        position,
        ε,
        metric,
        spacetime,
        model.volume_model,
        coords_top,
        spacetime_cache,
        model_cache.volume_cache)
end

function rest_frame_emissivity!(jε,
    position,
    ε,
    metric,
    spacetime,
    model::CompositeRadiativeModel,
    coords_top,
    spacetime_cache,
    model_cache::CompositeRadiativeModelCache)
    rest_frame_emissivity!(jε,
        position,
        ε,
        metric,
        spacetime,
        model.volume_model,
        coords_top,
        spacetime_cache,
        model_cache.volume_cache)
end

function is_final_position_at_source(position, spacetime, model::CompositeRadiativeModel)
    return is_final_position_at_source(position, spacetime, model.surface_model)
end

function surface_differential!(differential,
    position,
    model::CompositeRadiativeModel,
    coords_top)
    surface_differential!(differential, position, model.surface_model, coords_top)
end
