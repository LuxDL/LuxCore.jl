module LuxCore

using Compat: @compat
using DispatchDoctor: @stable
using Functors: Functors, fmap, fleaves
using Random: Random, AbstractRNG, Xoshiro
using Setfield: Setfield

# PRNG Handling
"""
    replicate(rng::AbstractRNG)

Creates a copy of the `rng` state depending on its type.
"""
replicate(rng::AbstractRNG) = deepcopy(rng)
function replicate(rng::Random.TaskLocalRNG)
    @warn "`replicate` doesn't work for `TaskLocalRNG`. Returning the same `TaskLocalRNG`." maxlog=1
    return deepcopy(rng)
end

_default_rng() = Xoshiro(1234)

"""
    abstract type AbstractLuxLayer

Abstract Type for all Lux Layers

Users implementing their custom layer, **must** implement

  - `initial_parameters(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This
    returns a `NamedTuple` containing the trainable parameters for the layer.
  - `initial_states(rng::AbstractRNG, layer::CustomAbstractExplicitLayer)` -- This returns a
    NamedTuple containing the current state for the layer. For most layers this is typically
    empty. Layers that would potentially contain this include `BatchNorm`, `LSTM`, `GRU`,
    etc.

Optionally:

  - `parameter_length(layer::CustomAbstractExplicitLayer)` -- These can be automatically
    calculated, but it is recommended that the user defines these.
  - `state_length(layer::CustomAbstractExplicitLayer)` -- These can be automatically
    calculated, but it is recommended that the user defines these.

See also [`AbstractLuxContainerLayer`](@ref)
"""
abstract type AbstractLuxLayer end

"""
    initial_parameters(rng::AbstractRNG, layer)

Generate the initial parameters of the layer `l`.
"""
function initial_parameters end

"""
    initial_states(rng::AbstractRNG, layer)

Generate the initial states of the layer `l`.
"""
function initial_states end

for op in (:initial_parameters, :initial_states)
    @eval begin
        $(op)(::AbstractRNG, ::Union{AbstractLuxLayer, Nothing}) = NamedTuple()
        $(op)(rng::AbstractRNG, l::NamedTuple) = map(Base.Fix1($op, rng), l)
        function $(op)(rng::AbstractRNG, l)
            contains_lux_layer(l) && return fmap(Base.Fix1($op, rng), l; exclude=_fmap_leaf)
            throw(MethodError($op, (rng, l)))
        end
    end
end

_fmap_leaf(::AbstractLuxLayer) = true
_fmap_leaf(x) = Functors.isleaf(x)

_getemptystate(::AbstractLuxLayer) = NamedTuple()
_getemptystate(l::NamedTuple) = map(_getemptystate, l)

"""
    parameter_length(layer)

Return the total number of parameters of the layer `l`.
"""
function parameter_length(l::AbstractLuxLayer)
    return parameter_length(initial_parameters(_default_rng(), l))
end
function parameter_length(nt::Union{NamedTuple, Tuple})
    return length(nt) == 0 ? 0 : sum(parameter_length, nt)
end
parameter_length(a::AbstractArray) = length(a)

"""
    state_length(layer)

Return the total number of states of the layer `l`.
"""
state_length(l::AbstractLuxLayer) = state_length(initial_states(_default_rng(), l))
state_length(nt::Union{NamedTuple, Tuple}) = length(nt) == 0 ? 0 : sum(state_length, nt)
state_length(a::AbstractArray) = length(a)
state_length(::Any) = 1

"""
    input_size(layer)

Return the input size of the layer.
"""
function input_size end

_size(x::AbstractVector) = size(x)
_size(x::AbstractArray) = size(x)[1:(ndims(x) - 1)]
__size(x) = fmap(_size, x)

"""
    output_size(layer, x, rng)

Return the output size of the layer. If `output_size(layer)` is defined, that method
takes precedence, else we compute the layer output to determine the final size.

The fallback implementation of this function assumes the inputs were batched, i.e.,
if any of the outputs are Arrays, with `ndims(A) > 1`, it will return
`size(A)[1:(end - 1)]`. If this behavior is undesirable, provide a custom
`output_size(layer, x, rng)` implementation).
"""
function output_size(layer, x, rng)
    hasmethod(output_size, Tuple{typeof(layer)}) && return output_size(layer)
    ps, st = setup(rng, layer)
    y = first(apply(layer, x, ps, st))
    return __size(y)
end

"""
    setup(rng::AbstractRNG, layer)

Shorthand for getting the parameters and states of the layer `l`. Is equivalent to
`(initial_parameters(rng, l), initial_states(rng, l))`.

!!! warning

    This function is not pure, it mutates `rng`.
"""
setup(rng::AbstractRNG, l) = (initial_parameters(rng, l), initial_states(rng, l))

"""
    apply(model, x, ps, st)

In most cases this function simply calls `model(x, ps, st)`. However, it is still
recommended to call `apply` instead of `model(x, ps, st)` directly. Some of the reasons for
this include:

 1. For certain types of inputs `x`, we might want to perform preprocessing before calling
    `model`. For eg, if `x` is an Array of `ReverseDiff.TrackedReal`s this can cause
    significant regressions in `model(x, ps, st)` (since it won't hit any of the BLAS
    dispatches). In those cases, we would automatically convert `x` to a
    `ReverseDiff.TrackedArray`.
 2. Certain user defined inputs need to be applied to specific layers but we want the
    datatype of propagate through all the layers (even unsupported ones). In these cases,
    we can unpack the input in `apply` and pass it to the appropriate layer and then
    repack it before returning. See the Lux manual on Custom Input Types for a motivating
    example.

!!! tip

    `apply` is integrated with `DispatchDoctor.jl` that allows automatic verification of
    type stability. By default this is "disable"d. For more information, see the
    [documentation](https://github.com/MilesCranmer/DispatchDoctor.jl).
"""
@stable default_mode="disable" function apply(model::AbstractLuxLayer, x, ps, st)
    return model(x, ps, st)
end

"""
    stateless_apply(model, x, ps)

Calls `apply` and only returns the first argument. This function requires that `model` has
an empty state of `NamedTuple()`. Behavior of other kinds of models are undefined and it is
the responsibility of the user to ensure that the model has an empty state.
"""
function stateless_apply(model::AbstractLuxLayer, x, ps)
    return first(apply(model, x, ps, _getemptystate(model)))
end

"""
    display_name(layer::AbstractLuxLayer)

Printed Name of the `layer`. If the `layer` has a field `name` that is used, else the type
name is used.
"""
@generated function display_name(l::L) where {L <: AbstractLuxLayer}
    hasfield(L, :name) &&
        return :(ifelse(l.name === nothing, $(string(nameof(L))), string(l.name)))
    return :($(string(nameof(L))))
end
display_name(::T) where {T} = string(nameof(T))

# Abstract Container Layers
"""
    abstract type AbstractLuxContainerLayer{layers} <: AbstractLuxLayer

Abstract Container Type for certain Lux Layers. `layers` is a tuple containing fieldnames
for the layer, and constructs the parameters and states using those.

Users implementing their custom layer can extend the same functions as in
[`AbstractLuxLayer`](@ref).

!!! tip

    Advanced structure manipulation of these layers post construction is possible via
    `Functors.fmap`. For a more flexible interface, we recommend using
    `Lux.Experimental.@layer_map`.
"""
abstract type AbstractLuxContainerLayer{layers} <: AbstractLuxLayer end

function initial_parameters(rng::AbstractRNG,
        l::AbstractLuxContainerLayer{layers}) where {layers}
    length(layers) == 1 && return initial_parameters(rng, getfield(l, layers[1]))
    return NamedTuple{layers}(initial_parameters.(rng, getfield.((l,), layers)))
end

function initial_states(rng::AbstractRNG,
        l::AbstractLuxContainerLayer{layers}) where {layers}
    length(layers) == 1 && return initial_states(rng, getfield(l, layers[1]))
    return NamedTuple{layers}(initial_states.(rng, getfield.((l,), layers)))
end

function parameter_length(l::AbstractLuxContainerLayer{layers}) where {layers}
    return sum(parameter_length, getfield.((l,), layers))
end

function state_length(l::AbstractLuxContainerLayer{layers}) where {layers}
    return sum(state_length, getfield.((l,), layers))
end

_fmap_leaf(::AbstractLuxContainerLayer) = true

function _getemptystate(l::AbstractLuxContainerLayer{layers}) where {layers}
    length(layers) == 1 && return _getemptystate(getfield(l, first(layers)))
    return NamedTuple{layers}(_getemptystate.(getfield.((l,), layers)))
end

# Make AbstractExplicit Layers Functor Compatible
function Functors.functor(::Type{<:AbstractLuxContainerLayer{layers}},
        x) where {layers}
    _children = NamedTuple{layers}(getproperty.((x,), layers))
    recon_fn = (l, (c, n)) -> Setfield.set(l, Setfield.PropertyLens{n}(), c)
    layer_reconstructor = let x = x, recon_fn = recon_fn, layers = layers
        z -> reduce(recon_fn, zip(z, layers); init=x)
    end
    return _children, layer_reconstructor
end

# Test Mode
"""
    testmode(st::NamedTuple)

Make all occurrences of `training` in state `st` -- `Val(false)`.
"""
testmode(st::NamedTuple) = update_state(st, :training, Val(false))

"""
    trainmode(st::NamedTuple)

Make all occurrences of `training` in state `st` -- `Val(true)`.
"""
trainmode(st::NamedTuple) = update_state(st, :training, Val(true))

"""
    update_state(st::NamedTuple, key::Symbol, value;
        layer_check=_default_layer_check(key))

Recursively update all occurrences of the `key` in the state `st` with the `value`.
"""
function update_state(st::NamedTuple, key::Symbol, value;
        layer_check::LC=_default_layer_check(key)) where {LC}
    fmap_fn = let key = key, value = value
        _st -> Setfield.set(_st, Setfield.PropertyLens{key}(), value)
    end
    return fmap(fmap_fn, st; exclude=layer_check)
end

function _default_layer_check(key)
    return let key = key
        x -> hasmethod(keys, (typeof(x),)) ? (key ∈ keys(x)) : false
    end
end

"""
    contains_lux_layer(l) -> Bool

Check if the structure `l` is a Lux AbstractLuxLayer or a container of such a layer.
"""
function contains_lux_layer(l)
    return check_fmap_condition(Base.Fix2(isa, AbstractLuxLayer),
        AbstractLuxLayer, l)
end

"""
    check_fmap_condition(cond, tmatch::Union{Type, Nothing}, x) -> Bool

`fmap`s into the structure `x` and see if `cond` is satisfied for any of the leaf elements.

## Arguments

  - `cond` - A function that takes a single argument and returns a `Bool`.
  - `tmatch` - A shortcut to check if `x` is of type `tmatch`. Can be disabled by passing
    `nothing`.
  - `x` - The structure to check.

## Returns

A Boolean Value
"""
check_fmap_condition(cond::C, ::Nothing, x) where {C} = any(cond, fleaves(x))
check_fmap_condition(cond::C, ::Nothing, ::NamedTuple{()}) where {C} = any(cond, ())
function check_fmap_condition(cond::C, ::Type{T}, x) where {C, T}
    x isa T && return true
    return check_fmap_condition(cond, nothing, x)
end

@compat(public,
    (replicate, trainmode, testmode, update_state, contains_lux_layer,
        check_fmap_condition, AbstractLuxLayer, AbstractLuxContainerLayer,
        initial_parameters, initial_states, parameter_length, state_length,
        input_size, output_size, setup, apply, stateless_apply, display_name))

end
