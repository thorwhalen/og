## __init__.py

```python
"""
`meshed` contains a set of tools that allow the developer to provide a collection
of python objects (think functions) and some policy of how these should be connected
and get an aggregate object that will use the underlying objects in some way.

If you want something concrete, think of the python objects to be functions,
and the aggregation policies to be things like "function composition" (pipelines)
or DAGs.
But the intent is to be able to get more general aggregations than those.

Extras
------

`itools.py` contain tools that enable operations on graphs where graphs are represented
by an adjacency Mapping.

"""

from meshed.dag import DAG, ch_funcs, ch_names
from meshed.base import FuncNode, compare_signatures
from meshed.makers import code_to_dag, code_to_fnodes
from meshed.itools import random_graph, topological_sort
from meshed.slabs import Slabs
from meshed.util import (
    iterize,
    ConditionalIterize,
    instance_checker,
    replace_item_in_iterable,
    parameter_merger,
    provides,
    Pipe,
)
from meshed.components import Itemgetter, AttrGetter
from meshed.caching import LazyProps
```

## base.py

```python
"""
Base functionality of meshed
"""

from collections import Counter
from dataclasses import dataclass, field, fields
from functools import partial, cached_property
from typing import Union, Literal
from collections.abc import Callable, MutableMapping, Iterable, Sized, Sequence

from i2 import Sig, call_somewhat_forgivingly
from i2.signatures import (
    ch_variadics_to_non_variadic_kind,
    CallableComparator,
    compare_signatures,
)
from meshed.util import ValidationError, NameValidationError, mk_func_name
from meshed.itools import add_edge

BindInfo = Literal["var_nodes", "params", "hybrid"]


def underscore_func_node_names_maker(func: Callable, name=None, out=None):
    """This name maker will resolve names in the following fashion:

     #. look at the (func) name and out given as arguments, if None...
     #. use mk_func_name(func) to make names.

    It will use the mk_func_name(func)  itself for out, but suffix the same with
    an underscore to provide a mk_func_name.

    This is so because here we want to allow easy construction of function networks
    where a function's output will be used as another's input argument when
    that argument has the the function's (output) name.
    """
    if out is None and hasattr(func, "_provides"):
        if len(func._provides) > 0:
            out = func._provides[0]
    if name is not None and out is not None:
        if name == out:
            name = name + "_"
        return name, out

    try:
        name_of_func = mk_func_name(func)
    except NameValidationError as err:
        err_msg = err.args[0]
        err_msg += (
            f"\nSuggestion: You might want to specify a name explicitly in "
            f"FuncNode(func, name=name) instead of just giving me the func as is."
        )
        raise NameValidationError(err_msg)
    if name is None and out is None:
        return name_of_func + "_", name_of_func
    elif out is None:
        return name, "_" + name
    elif name is None:
        if name_of_func == out:
            name_of_func += "_"
        return name_of_func, out


def basic_node_validator(func_node):
    """Validates a func node. Raises ValidationError if something wrong. Returns None.

    Validates:

    * that the ``func_node`` params are valid, that is, if not ``None``
        * ``func`` should be a callable
        * ``name`` and ``out`` should be ``str``
        * ``bind`` should be a ``Dict[str, str]``
    * that the names (``.name``, ``.out`` and all ``.bind.values()``)
        * are valid python identifiers (alphanumeric or underscore not starting with
          digit)
        * are not repeated (no duplicates)
    * that ``.bind.keys()`` are indeed present as params of ``.func``

    """
    _func_node_args_validation(
        func=func_node.func, name=func_node.name, bind=func_node.bind, out=func_node.out
    )
    names = [func_node.name, func_node.out, *func_node.bind.values()]

    names_that_are_not_strings = [name for name in names if not isinstance(name, str)]
    if names_that_are_not_strings:
        names_that_are_not_strings = ", ".join(map(str, names_that_are_not_strings))
        raise ValidationError(f"Should be strings: {names_that_are_not_strings}")

    # Make sure there's no name duplicates
    _duplicates = duplicates(names)
    if _duplicates:
        raise ValidationError(f"{func_node} has duplicate names: {_duplicates}")

    # Make sure all names are identifiers
    _non_identifiers = list(filter(lambda name: not name.isidentifier(), names))
    # print(_non_identifiers, names)
    if _non_identifiers:
        raise ValidationError(f"{func_node} non-identifier names: {_non_identifiers}")

    # Making sure all src_name keys are in the function's signature
    bind_names_not_in_sig_names = func_node.bind.keys() - func_node.sig.names
    assert not bind_names_not_in_sig_names, (
        f"some bind keys weren't found as function argnames: "
        f"{', '.join(bind_names_not_in_sig_names)}"
    )


def handle_variadics(func):
    func = ch_variadics_to_non_variadic_kind(func)
    # sig = Sig(func)
    # var_kw = sig.var_keyword_name

    #   # may be always return the wrapped function
    # # func.var_kw_name = var_kw # TODO add it when needed

    return func


# TODO: When 3.10, look into and possibly use match_args in to_dict and from_dict
# TODO: Make FuncNode immutable (is there a way to use frozen=True with post_init?)
# TODO: How to get a safe hash? Needs to be immutable only?
# TODO: FuncNode(func_node) gives us FuncNode(scope -> ...). Should we have it be
#  FuncNode.from_dict(func_node.to_dict()) instead?
# @dataclass(eq=True, order=True, unsafe_hash=True)
@dataclass(order=True)
class FuncNode:
    """A function wrapper that makes the function amenable to operating in a network.

    :param func: Function to wrap
    :param name: The name to associate to the function
    :param bind: The {func_argname: external_name,...} mapping that defines where
        the node will source the data to call the function.
        This only has to be used if the external names are different from the names
        of the arguments of the function.
    :param out: The variable name the function should write it's result to

    Like we stated: `FuncNode` is meant to operate in computational networks.
    But knowing what it does will help you make the networks you want, so we commend
    your curiousity, and will oblige with an explanation.

    Say you have a function to multiply numbers.

    >>> def multiply(x, y):
    ...     return x * y

    And you use it in some code like this:

    >>> item_price = 3.5
    >>> num_of_items = 2
    >>> total_price = multiply(item_price, num_of_items)

    What the execution of `total_price = multiply(item_price, num_of_items)` does is
    - grab the values (in the locals scope -- a dict), of ``item_price`` and ``num_of_items``,
    - call the multiply function on these, and then
    - write the result to a variable (in locals) named ``total_price``

    `FuncNode` is a function wrapper that specification of such a
    `output = function(...inputs...)` assignment statement
    in such a way that it can carry it out on a `scope`.
    A `scope` is a `dict` where the function can find it's input values and write its
    output values.

    For example, the `FuncNode` form of the above statement would be:

    >>> func_node = FuncNode(
    ...     func=multiply,
    ...     bind={'x': 'item_price', 'y': 'num_of_items'})
    >>> func_node
    FuncNode(x=item_price,y=num_of_items -> multiply_ -> multiply)

    Note the `bind` is a mapping **from** the variable names of the wrapped function
    **to** the names of the scope.

    That is, when it's time to execute, it tells the `FuncNode` where to find the values
    of its inputs.

    If an input is not specified in this `bind` mapping, the scope
    (external) name is supposed to be the same as the function's (internal) name.

    The purpose of a `FuncNode` is to source some inputs somewhere, compute something
    with these, and write the result somewhere. That somewhere is what we call a
    scope. A scope is a dictionary (or any mutuable mapping to be precise) and it works
    like this:

    >>> scope = {'item_price': 3.5, 'num_of_items': 2}
    >>> func_node.call_on_scope(scope)  # see that it returns 7.0
    7.0
    >>> scope  # but also wrote this in the scope
    {'item_price': 3.5, 'num_of_items': 2, 'multiply': 7.0}

    Consider ``item_price,num_of_items -> multiply_ -> multiply``.
    See that the name of the function is used for the name of its output,
    and an underscore-suffixed name for its function name.
    That's the default behavior if you don't specify either a name (of the function)
    for the `FuncNode`, or a `out`.
    The underscore is to distinguish from the name of the function itself.
    The function gets the underscore because this favors particular naming style.

    You can give it a custom name as well.

    >>> FuncNode(multiply, name='total_price', out='daily_expense')
    FuncNode(x,y -> total_price -> daily_expense)

    If you give an `out`, but not a `name` (for the function), the function's
    name will be taken:

    >>> FuncNode(multiply, out='daily_expense')
    FuncNode(x,y -> multiply -> daily_expense)

    If you give a `name`, but not a `out`, an underscore-prefixed version of
    the `name` will be taken:

    >>> FuncNode(multiply, name='total_price')
    FuncNode(x,y -> total_price -> _total_price)

    Note: In the context of networks if you want to reuse a same function
    (say, `multiply`) in multiple places
    you'll **need** to give it a custom name because the functions are identified by
    this name in the network.


    """

    # TODO: Make everything but func keyword-only (check for non-keyword usage before)
    # Using __init__ for now, but when 3.10, use field func with kw_only=True
    func: Callable
    name: str = field(default=None)
    bind: dict = field(default_factory=dict)
    out: str = field(default=None)
    func_label: str = field(default=None)  # TODO: Integrate more
    # write_output_into_scope: bool = True  # TODO: Do we really want to allow False?
    names_maker: Callable = underscore_func_node_names_maker
    node_validator: Callable = basic_node_validator

    # def __init__(
    #     self,
    #     func: Callable,
    #     *,
    #     name: str = None,
    #     bind: dict = None,
    #     out: str = None,
    #     func_label: str = None,  # TODO: Integrate more
    #     # write_output_into_scope: bool = True  # TODO: Do we really want to allow False?
    #     names_maker: Callable = underscore_func_node_names_maker,
    #     node_validator: Callable = basic_node_validator,
    # ):
    #     self.func = func
    #     self.name = name
    #     self.bind = bind
    #     self.out = out
    #     self.func_label = func_label
    #     # self.write_output_into_scope = write_output_into_scope
    #     self.names_maker = names_maker
    #     self.node_validator = node_validator
    #     self.__post_init__()

    def __post_init__(self):
        if self.bind is None:
            self.bind = dict()
        self.func = handle_variadics(self.func)
        _func_node_args_validation(func=self.func, name=self.name, out=self.out)
        self.name, self.out = self.names_maker(self.func, self.name, self.out)
        self.__name__ = self.name
        # self.__name__ = self.name
        # The wrapped function's signature will be useful
        # when interfacing with it and the scope.
        self.sig = Sig(self.func)

        # replace integer bind keys with their corresponding name
        self.bind = _bind_where_int_keys_repl_with_argname(self.bind, self.sig.names)
        # complete bind with the argnames of the signature
        _complete_dict_with_iterable_of_required_keys(self.bind, self.sig.names)
        _func_node_args_validation(bind=self.bind)

        self.extractor = partial(_mapped_extraction, to_extract=self.bind)

        if self.func_label is None:
            self.func_label = self.name

        self.node_validator(self)

    # TODO: BindInfo lists only three unique behaviors, but there are seven actual
    #  possible values for bind_info. All the rest are convenience aliases. Is this
    #  a good idea? The hesitation here comes from the fact that the values/keys
    #  language describes the bind data structure (dict), but the var_nodes/params
    #  language describes their contextual use. If had to choose, I'd chose the latter.
    def synopsis_string(self, bind_info: BindInfo = "values"):
        """

        :param bind_info: How to represent the bind in the synopsis string. Could be:
            - 'values', `var_nodes` or `varnodes`: the values of the bind (default).
            - 'keys' or 'params': the keys of the bind
            - 'hybrid': the keys of the bind, but with the values that are the same as
                the keys omitted.
        :return:

        >>> fn = FuncNode(
        ...     func=lambda y, c: None , name='h', bind={'y': 'b', 'c': 'c'}, out='d'
        ... )
        >>> fn.synopsis_string()
        'b,c -> h -> d'
        >>> fn.synopsis_string(bind_info='keys')
        'y,c -> h -> d'
        >>> fn.synopsis_string(bind_info='hybrid')
        'y=b,c -> h -> d'
        """
        if bind_info in {"values", "varnodes", "var_nodes"}:
            return f"{','.join(self.bind.values())} -> {self.name} " f"-> {self.out}"
        elif bind_info == "hybrid":

            def gen():
                for k, v in self.bind.items():
                    if k == v:
                        yield k
                    else:
                        yield f"{k}={v}"

            return f"{','.join(gen())} -> {self.name} " f"-> {self.out}"
        elif bind_info in {"keys", "params"}:
            return f"{','.join(self.bind.keys())} -> {self.name} " f"-> {self.out}"
        else:
            raise ValueError(f"Unknown bind_info: {bind_info}")

    def __repr__(self):
        return f'FuncNode({self.synopsis_string(bind_info="hybrid")})'

    def call_on_scope(self, scope: MutableMapping, write_output_into_scope=True):
        """Call the function using the given scope both to source arguments and write
        results.

        Note: This method is only meant to be used as a backend to __call__, not as
        an actual interface method. Additional control/constraints on read and writes
        can be implemented by providing a custom scope for that."""
        relevant_kwargs = dict(self.extractor(scope))
        args, kwargs = self.sig.mk_args_and_kwargs(relevant_kwargs)
        output = call_somewhat_forgivingly(
            self.func, args, kwargs, enforce_sig=self.sig
        )
        if write_output_into_scope:
            scope[self.out] = output
        return output

    def _hash_str(self):
        """Design idea.
        Attempt to construct a hash that reflects the actual identity we want.
        Need to transform to int. Only identifier chars alphanumerics and underscore
        and space are used, so could possibly encode as int (for __hash__ method)
        in a way that is reverse-decodable and with reasonable int size.
        """
        return self.synopsis_string(bind_info="hybrid")
        # return ';'.join(self.bind) + '::' + self.out

    # TODO: Find a better one. Need to have guidance on hash and eq methods dos-&-donts
    def __hash__(self):
        return hash(self._hash_str())

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __call__(self, scope):
        """Deprecated: Don't use. Might be a normal function with a signature"""
        from warnings import warn

        raise DeprecationWarning(f"Deprecated. Use .call_on_scope(scope) instead.")
        # warn(f'Deprecated. Use .call_on_scope(scope) instead.', DeprecationWarning)
        # return self.call_on_scope(scope)

    def to_dict(self):
        """The inverse of from_dict: FuncNode.from_dict(fn.to_dict()) == fn"""
        return {x.name: getattr(self, x.name) for x in fields(self)}

    @classmethod
    def from_dict(cls, dictionary: dict):
        """The inverse of to_dict: Make a ``FuncNode`` from a dictionary of init args"""
        return cls(**dictionary)

    def ch_attrs(self, **new_attrs_values):
        """Returns a copy of the func node with some of its attributes changed

        >>> def plus(a, b):
        ...     return a + b
        ...
        >>> def minus(a, b):
        ...     return a - b
        ...
        >>> fn = FuncNode(func=plus, out='sum')
        >>> fn.func == plus
        True
        >>> fn.name == 'plus'
        True
        >>> new_fn = fn.ch_attrs(func=minus)
        >>> new_fn.func == minus
        True
        >>> new_fn.synopsis_string() == 'a,b -> plus -> sum'
        True
        >>>
        >>>
        >>> newer_fn = fn.ch_attrs(func=minus, name='sub', out='difference')
        >>> newer_fn.synopsis_string() == 'a,b -> sub -> difference'
        True
        """
        return ch_func_node_attrs(self, **new_attrs_values)

    @classmethod
    def has_as_instance(cls, obj):
        """Verify if ``obj`` is an instance of a FuncNode (or specific sub-class).

        The usefulness of this method is to not have to make a lambda with isinstance
        when filtering.

        >>> FuncNode.has_as_instance(FuncNode(lambda x: x))
        True
        >>> FuncNode.has_as_instance("I am not a FuncNode: I'm a string")
        False
        """
        return isinstance(obj, cls)

    def dot_lines(self, **kwargs):
        """Returns a list of lines that can be used to make a dot graph"""

        out = self.out

        func_id = self.name
        func_label = getattr(self, "func_label", func_id)
        if out == func_id:  # though forbidden in default FuncNode validation
            func_id = "_" + func_id

        # Get the Parameter objects for sig, with names changed to bind ones
        params = self.sig.ch_names(**self.bind).params

        yield from dot_lines_of_func_parameters(
            params, out=out, func_id=func_id, func_label=func_label, **kwargs
        )


# -------------------------------------------------------------------------------------
# viz stuff

from i2.signatures import Parameter, empty, Sig

# These are the defaults used in lined.
# TODO: Merge some of the functionalities around graph displays in lined and meshed
# TODO: Allow this to be overridden/edited by user, config2py style?
dflt_configs = dict(
    fnode_shape="box",
    vnode_shape="none",
    display_all_arguments=True,
    edge_kind="to_args_on_edge",
    input_node=True,
    output_node="output",
    func_display=True,
)


def dot_lines_of_func_parameters(
    parameters: Iterable[Parameter],
    out: str,
    func_id: str,
    *,
    func_label: str = None,
    vnode_shape: str = dflt_configs["vnode_shape"],
    fnode_shape: str = dflt_configs["fnode_shape"],
    func_display: bool = dflt_configs["func_display"],
) -> Iterable[str]:
    assert func_id != out, (
        f"Your func and output name shouldn't be the " f"same: {out=} {func_id=}"
    )
    yield f'{out} [label="{out}" shape="{vnode_shape}"]'
    for p in parameters:
        yield from param_to_dot_definition(p, shape=vnode_shape)

    if func_display:
        func_label = func_label or func_id
        yield f'{func_id} [label="{func_label}" shape="{fnode_shape}"]'
        yield f"{func_id} -> {out}"
        for p in parameters:
            yield f"{p.name} -> {func_id}"
    else:
        for p in parameters:
            yield f"{p.name} -> {out}"


def param_to_dot_definition(p: Parameter, shape=dflt_configs["vnode_shape"]):
    if p.default is not empty:
        name = p.name + "="
    elif p.kind == p.VAR_POSITIONAL:
        name = "*" + p.name
    elif p.kind == p.VAR_KEYWORD:
        name = "**" + p.name
    else:
        name = p.name
    yield f'{p.name} [label="{name}" shape="{shape}"]'


# -------------------------------------------------------------------------------------


@dataclass
class Mesh:
    func_nodes: Iterable[FuncNode]

    def synopsis_string(self, bind_info: BindInfo = "values"):
        return "\n".join(
            func_node.synopsis_string(bind_info) for func_node in self.func_nodes
        )


def validate_that_func_node_names_are_sane(func_nodes: Iterable[FuncNode]):
    """Assert that the names of func_nodes are sane.
    That is:

    * are valid dot (graphviz) names (we'll use str.isidentifier because lazy)
    * All the ``func.name`` and ``func.out`` are unique
    * more to come (TODO)...
    """
    func_nodes = list(func_nodes)
    node_names = [x.name for x in func_nodes]
    outs = [x.out for x in func_nodes]
    assert all(
        map(str.isidentifier, node_names)
    ), f"some node names weren't valid identifiers: {node_names}"
    assert all(
        map(str.isidentifier, outs)
    ), f"some return names weren't valid identifiers: {outs}"
    if len(set(node_names) | set(outs)) != 2 * len(func_nodes):
        c = Counter(node_names + outs)
        offending_names = [name for name, count in c.items() if count > 1]
        raise ValueError(
            f"Some of your node names and/or outs where used more than once. "
            f"They shouldn't. These are the names I find offensive: {offending_names}"
        )


def ensure_func_nodes(func_nodes):
    """Converts a list of objects to a list of FuncNodes."""
    # TODO: Take care of names (or track and take care if collision)
    if callable(func_nodes) and not isinstance(func_nodes, Iterable):
        # if input is a single function, make it a list containing that function
        single_func = func_nodes
        func_nodes = [single_func]
    for func_node in func_nodes:
        if is_func_node(func_node):
            yield func_node
        elif isinstance(func_node, Callable):
            yield FuncNode(func_node)
        else:
            raise TypeError(f"Can't convert this to a FuncNode: {func_node}")


_mk_func_nodes = ensure_func_nodes  # backwards compatibility


def _func_nodes_to_graph_dict(func_nodes):
    g = dict()

    for f in func_nodes:
        for src_name in f.bind.values():
            add_edge(g, src_name, f)
        add_edge(g, f, f.out)
    return g


def is_func_node(obj) -> bool:
    """
    >>> is_func_node(FuncNode(lambda x: x))
    True
    >>> is_func_node("I am not a FuncNode: I'm a string")
    False
    """
    # A weaker check than an isinstance(obj, FuncNode), which fails when we're
    # developing (therefore changing) FuncNode definition (without relaunching python
    # kernel). This is to be used instead, at least during development times
    # TODO: Replace with isinstance(obj, FuncNode) is this when development stabalizes
    #  See: https://github.com/i2mint/meshed/discussions/57
    # return isinstance(obj, FuncNode)
    cls = type(obj)
    if cls is not type:
        try:
            return any(getattr(x, "__name__", "") == "FuncNode" for x in cls.mro())
        except Exception:
            return isinstance(obj, FuncNode)
    else:
        return False


def is_not_func_node(obj) -> bool:
    """
    >>> is_not_func_node(FuncNode(lambda x: x))
    False
    >>> is_not_func_node("I am not a FuncNode: I'm a string")
    True
    """
    return not FuncNode.has_as_instance(obj)


def get_init_params_of_instance(obj):
    """Get names of instance object ``obj`` that are also parameters of the
    ``__init__`` of its class"""
    return {k: v for k, v in vars(obj).items() if k in Sig(type(obj)).names}


def ch_func_node_attrs(fn: FuncNode, **new_attrs_values):
    """Returns a copy of the func node with some of its attributes changed

    >>> def plus(a, b):
    ...     return a + b
    ...
    >>> def minus(a, b):
    ...     return a - b
    ...
    >>> fn = FuncNode(func=plus, out='sum')
    >>> fn.func == plus
    True
    >>> fn.name == 'plus'
    True
    >>> new_fn = ch_func_node_attrs(fn, func=minus)
    >>> new_fn.func == minus
    True
    >>> new_fn.synopsis_string() == 'a,b -> plus -> sum'
    True
    >>>
    >>>
    >>> newer_fn = ch_func_node_attrs(fn, func=minus, name='sub', out='difference')
    >>> newer_fn.synopsis_string() == 'a,b -> sub -> difference'
    True
    """
    init_params = get_init_params_of_instance(fn)
    if params_that_are_not_init_params := (new_attrs_values.keys() - init_params):
        raise ValueError(
            f"These are not params of {type(fn).__name__}: "
            f"{params_that_are_not_init_params}"
        )
    fn_kwargs = dict(init_params, **new_attrs_values)
    return FuncNode(**fn_kwargs)


def raise_signature_mismatch_error(fn, func):
    raise ValueError(
        "You can only change the func of a FuncNode with a another func if the "
        "signatures match.\n"
        f"\t{fn=}\n"
        f"\t{Sig(fn.func)=}\n"
        f"\t{Sig(func)=}\n"
    )


# from i2.signatures import keyed_comparator, SignatureComparator
# if compare_func is None:
#     compare_func = keyed_comparator(signature_comparator, key=Sig)


def _ch_func_node_func(fn: FuncNode, func: Callable):
    return ch_func_node_attrs(fn, func=func)


def ch_func_node_func(
    fn: FuncNode,
    func: Callable,
    *,
    func_comparator: CallableComparator = compare_signatures,
    ch_func_node=_ch_func_node_func,
    alternative=raise_signature_mismatch_error,
):
    if func_comparator(fn.func, func):
        return ch_func_node(fn, func=func)
    else:
        return alternative(fn, func)


def _new_bind(fnode, new_func):
    old_sig = Sig(fnode.func)
    new_sig = Sig(new_func)
    old_bind: dict = fnode.bind
    old_to_new_names_map = dict(zip(old_sig.names, new_sig.names))
    # TODO: assert some health stats on old_to_new_names_map
    new_bind = {old_to_new_names_map[k]: v for k, v in old_bind.items()}
    return new_bind


# TODO: Add more control (signature comparison, rebinding rules, renaming rules...)
# TODO: For example, can rebind to a function with different defaults, which are ignored.
#  Should we allow this? Should we allow to specify how to handle this?
# TODO: Should we include this in FuncNode as .ch_func(func)?
#  Possibly with an argument that specifies how to handle details, aligned with the
#  DAG.ch_funcs method. See ch_func_node_func.
def rebind_to_func(fnode: FuncNode, new_func: Callable):
    """Replaces ``fnode.func`` with ``new_func``, changing the ``.bind`` accordingly.

    >>> fn = FuncNode(lambda x, y: x + y, bind={'x': 'X', 'y': 'Y'})
    >>> fn.call_on_scope(dict(X=2, Y=3))
    5
    >>> new_fn = rebind_to_func(fn, lambda a, b, c=0: a * (b + c))
    >>> new_fn.call_on_scope(dict(X=2, Y=3))
    6
    >>> new_fn.call_on_scope(dict(X=2, Y=3, c=1))
    8
    """
    new_bind = _new_bind(fnode, new_func)
    return fnode.ch_attrs(func=new_func, bind=new_bind)


def insert_func_if_compatible(func_comparator: CallableComparator = compare_signatures):
    return partial(ch_func_node_func, func_comparator=func_comparator)


def _keys_and_values_are_strings_validation(d: dict):
    for k, v in d.items():
        if not isinstance(k, str):
            raise ValidationError(f"Should be a str: {k}")
        if not isinstance(v, str):
            raise ValidationError(f"Should be a str: {v}")


def _func_node_args_validation(
    *, func: Callable = None, name: str = None, bind: dict = None, out: str = None
):
    """Validates the four first arguments that are used to make a ``FuncNode``.
    Namely, if not ``None``,

    * ``func`` should be a callable

    * ``name`` and ``out`` should be ``str``

    * ``bind`` should be a ``Dict[str, str]``, ``Dict[int, str]`` or ``List[str]``

    * ``out`` should be a str

    """
    if func is not None and not isinstance(func, Callable):
        raise ValidationError(f"Should be callable: {func}")
    if name is not None and not isinstance(name, str):
        raise ValidationError(f"Should be a str: {name}")
    if bind is not None:
        if not isinstance(bind, dict):
            raise ValidationError(f"Should be a dict: {bind}")
        _keys_and_values_are_strings_validation(bind)
    if out is not None and not isinstance(out, str):
        raise ValidationError(f"Should be a str: {out}")


def _old_mapped_extraction(extract_from: dict, key_map: dict):
    """Deprecated: Old version of _mapped_extraction.

    for every (k, v) of key_map whose v is a key of extract_from, yields
    (v, extract_from[v])

    Meant to be curried into an extractor, and wrapped in dict.

    >>> extracted = _old_mapped_extraction(
    ...     {'a': 1, 'b': 2, 'c': 3}, # extract_from
    ...     {'A': 'a', 'C': 'c', 'D': 'd'}  # note that there's no 'd' in extract_from
    ... )
    >>> dict(extracted)
    {'a': 1, 'c': 3}

    """
    for k, v in key_map.items():
        if v in extract_from:
            yield v, extract_from[v]


def _mapped_extraction(src: dict, to_extract: dict):
    """for every (desired_name, src_name) of to_extract whose v is a key of source,
    yields (desired_name, source[src_name])

    It's purpose is to extract inputs from a src.
    The names used in the src may be different from those desired by the function,
    those to_extract specifies what to extract by a {desired_name: src_name, ...}
    map.

    _mapped_extraction_ is mant to be curried into an extractor.

    >>> extracted = _mapped_extraction(
    ...     src={'A': 1, 'B': 2, 'C': 3},
    ...     to_extract={'a': 'A', 'c': 'C', 'd': 'D'}  # note that there's no 'd' here
    ... )
    >>> dict(extracted)
    {'a': 1, 'c': 3}

    """
    for desired_name, src_name in to_extract.items():
        if src_name in src:
            yield desired_name, src[src_name]


def duplicates(elements: Iterable | Sized):
    c = Counter(elements)
    if len(c) != len(elements):
        return [name for name, count in c.items() if count > 1]
    else:
        return []


def _bind_where_int_keys_repl_with_argname(bind: dict, names: Sequence[str]) -> dict:
    """

    :param bind: A bind dict, as used in FuncNode
    :param names: A sequence of strings
    :return: A bind dict where integer keys were replaced with the corresponding
        name from names.

    >>> bind = {0: 'a', 1: 'b', 'c': 'x', 'd': 'y'}
    >>> names = 'e f g h'.split()
    >>> _bind_where_int_keys_repl_with_argname(bind, names)
    {'e': 'a', 'f': 'b', 'c': 'x', 'd': 'y'}
    """

    def transformed_items():
        for k, v in bind.items():
            if isinstance(k, int):
                argname = names[k]
                yield argname, v
            else:
                yield k, v

    return dict(transformed_items())


def _complete_dict_with_iterable_of_required_keys(
    to_complete: dict, complete_with: Iterable
):
    """Complete `to_complete` (in place) with `complete_with`
    `complete_with` contains values that must be covered by `to_complete`
    Those values that are not covered will be inserted in to_complete,
    with key=val

    >>> d = {'a': 'A', 'c': 'C'}
    >>> _complete_dict_with_iterable_of_required_keys(d, 'abc')
    >>> d
    {'a': 'A', 'c': 'C', 'b': 'b'}

    """
    keys_already_covered = set(to_complete)
    for required_key in complete_with:
        if required_key not in keys_already_covered:
            to_complete[required_key] = required_key


from typing import NewType, Dict, Tuple
from collections.abc import Mapping

# TODO: Make a type where ``isinstance(s, Identifier) == s.isidentifier()``
Identifier = NewType("Identifier", str)  # + should satisfy str.isidentifier
Bind = Union[
    str,  # Identifier or ' '.join(Iterable[Identifier])
    dict[Identifier, Identifier],
    Sequence[Union[Identifier, tuple[Identifier, Identifier]]],
]

IdentifierMapping = dict[Identifier, Identifier]


def identifier_mapping(x: Bind) -> IdentifierMapping:
    """Get an ``IdentifierMapping`` dict from a more loosely defined ``Bind``.

    You can get an identifier mapping (that is, an explicit for for a ``bind`` argument)
    from...

    ... a single space-separated string

    >>> identifier_mapping('x a_b yz')  #
    {'x': 'x', 'a_b': 'a_b', 'yz': 'yz'}

    ... an iterable of strings or pairs of strings

    >>> identifier_mapping(['foo', ('bar', 'mitzvah')])
    {'foo': 'foo', 'bar': 'mitzvah'}

    ... a dict will be considered to be the mapping itself

    >>> identifier_mapping({'x': 'y', 'a': 'b'})
    {'x': 'y', 'a': 'b'}
    """
    if isinstance(x, str):
        x = x.split()
    if not isinstance(x, Mapping):

        def gen():
            for item in x:
                if isinstance(item, str):
                    yield item, item
                else:
                    yield item

        return dict(gen())
    else:
        return dict(**x)


FuncNodeAble = Union[FuncNode, Callable]


def func_node_transformer(
    fn: FuncNode,
    kwargs_transformers=(),
):
    """Get a modified ``FuncNode`` from an iterable of ``kwargs_trans`` modifiers."""
    func_node_kwargs = fn.to_dict()
    if callable(kwargs_transformers):
        kwargs_transformers = [kwargs_transformers]
    for trans in kwargs_transformers:
        if (new_kwargs := trans(func_node_kwargs)) is not None:
            func_node_kwargs = new_kwargs
    return FuncNode.from_dict(func_node_kwargs)


def func_nodes_to_code(
    func_nodes: Iterable[FuncNode],
    func_name: str = "generated_pipeline",
    *,
    favor_positional: bool = True,
) -> str:
    """Convert an iterable of FuncNodes back to executable Python code.

    This is the inverse operation of code_to_fnodes - it takes FuncNodes and generates
    Python code that would create equivalent FuncNodes when parsed.
    When favor_positional is True, any keyword argument with key equal to its value
    is moved to the positional arguments list:

        func(a=a, b=b, c=z, d=d)  ->  func(a, b, c=z, d=d)

    :param func_nodes: Iterable of FuncNode instances to convert to code
    :param func_name: Name for the generated function
    :param favor_positional: When True, transforms kwargs of the form key=key into positional args.
    :return: String containing Python code


    """

    def lines():
        yield f"def {func_name}():"
        for func_node in func_nodes:
            line = _func_node_to_assignment_line(
                func_node, favor_positional=favor_positional
            )
            yield f"    {line}"

    return "\n".join(lines())


def _func_node_to_assignment_line(func_node: FuncNode, favor_positional=True) -> str:
    """Convert a single FuncNode to a Python assignment line.

    :param func_node: The FuncNode to convert
    :param favor_positional: When True, transforms kwargs of the form key=key into positional args
    :return: String like "output = func_name(arg1, arg2=value)"
    """
    # Get the function name - use func_label if available, otherwise name
    func_name = getattr(func_node, "func_label", None) or func_node.name

    # Handle special cases for generated functions
    if func_name.endswith("__0") or func_name.endswith("__1"):
        # This is likely an itemgetter from tuple unpacking
        if hasattr(func_node.func, "keywords") and "keys" in func_node.func.keywords:
            keys = func_node.func.keywords["keys"]
            if len(keys) == 1:
                # Single item extraction
                source_var = list(func_node.bind.values())[0]
                return f"{func_node.out} = {source_var}[{keys[0]}]"

    # Build argument list
    args = []
    kwargs = []

    # Process bind dictionary to separate positional and keyword arguments
    for param, source in func_node.bind.items():
        if isinstance(param, int):
            # Positional argument
            args.append((param, source))
        else:
            # Keyword argument - check if favor_positional applies
            if favor_positional and param == source:
                # Convert to positional argument by using the parameter order from signature
                try:
                    param_index = list(func_node.sig.names).index(param)
                    args.append((param_index, source))
                except (ValueError, AttributeError):
                    # If we can't determine order, keep as keyword
                    kwargs.append((param, source))
            else:
                kwargs.append((param, source))

    # Sort positional arguments by their index
    args.sort(key=lambda x: x[0])

    # Build the argument string
    arg_parts = []

    # Add positional arguments
    for _, source in args:
        arg_parts.append(str(source))

    # Add keyword arguments
    for param, source in kwargs:
        arg_parts.append(f"{param}={source}")

    arg_string = ", ".join(arg_parts)

    # Handle tuple unpacking in output
    if "__" in func_node.out and not func_node.out.endswith(("__0", "__1")):
        # This might be a tuple output that was created from tuple unpacking
        output_parts = func_node.out.split("__")
        if len(output_parts) > 1:
            output = ", ".join(output_parts)
            return f"{output} = {func_name}({arg_string})"

    return f"{func_node.out} = {func_name}({arg_string})"
```

## caching.py

```python
"""Caching meshes"""

from functools import cached_property
from inspect import signature

from meshed.util import func_name, LiteralVal


def set_cached_property_attr(obj, name, value):
    """
    Helper to set cached properties.

    Reason: When adding cached_property dynamically (not just with the @cached_property)
    the name is not set correctly. This solves that.
    """
    cached_value = cached_property(value)
    cached_value.__set_name__(obj, name)
    setattr(obj, name, cached_value)


class LazyProps:
    """
    A class that makes all its attributes cached_property properties.

    Example:

    >>> class Klass(LazyProps):
    ...     a = 1
    ...     b = 2
    ...
    ...     # methods with one argument are cached
    ...     def c(self):
    ...         print("computing c...")
    ...         return self.a + self.b
    ...
    ...     d = lambda x: 4
    ...     e = LazyProps.Literal(lambda x: 4)
    ...
    ...     @LazyProps.Literal  # to mark that this method should not be cached
    ...     def method1(self):
    ...         return self.a * 7
    ...
    ...     # Methods with more than one argument are not cached
    ...     def method2(self, x):
    ...         return x + 1
    ...
    ...
    >>> k = Klass()
    >>> k.b
    2
    >>> k.c
    computing c...
    3
    >>> k.c  # note that c is not recomputed
    3
    >>> k.d  # d, a lambda with one argument, is treated as a cached property
    4
    >>> k.e()  # e is marked as a literal so is not a cached property, so need to call
    4
    >>> k.method1()  # method1 has one argument, but marked as a literal
    7
    >>> k.method2(10)  # method2 has more than one argument, so is not a cached property
    11
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for attr_name in (a for a in dir(cls) if not a.startswith("__")):
            attr_obj = getattr(cls, attr_name)
            if isinstance(attr_obj, LiteralVal):
                setattr(cls, attr_name, attr_obj.val)
            elif callable(attr_obj) and len(signature(attr_obj).parameters) == 1:
                set_cached_property_attr(cls, attr_name, attr_obj)

    Literal = LiteralVal  # just to have Literal available as LazyProps.Literal


def add_cached_property(cls, method, attr_name=None):
    """
    Add a method as a cached property to a class.
    """
    attr_name = attr_name or func_name(method)
    set_cached_property_attr(cls, attr_name, method)
    return cls


def add_cached_property_from_func(cls, func, attr_name=None):
    """
    Add a function cached property to a class.
    """
    params = list(signature(func).parameters)

    def method(self):
        return func(**{k: getattr(self, k) for k in params})

    method.__name__ = func.__name__
    method.__doc__ = func.__doc__

    return add_cached_property(cls, method, attr_name)


def with_cached_properties(funcs):
    """
    A decorator to add cached properties to a class.
    """

    def add_cached_properties(cls):
        for func in funcs:
            if not callable(func):
                func, attr_name = func  # assume it's a (func, attr_name) pair
            else:
                attr_name = None
            add_cached_property_from_func(cls, func, attr_name)
        return cls

    return add_cached_properties
```

## components.py

```python
"""
Specialized components for meshed.
"""

from i2 import Sig
from typing import Any
from collections.abc import Callable
from operator import itemgetter, attrgetter
from dataclasses import dataclass
from functools import partial


@dataclass
class Extractor:
    extractor_factory: Callable[[Any], Callable]
    extractor_params: Any
    # TODO: When migrating CI to 3.10+, can use `kw_only=True` here
    # name: str = field(default='extractor', kw_only=True)
    # input_name: str = field(default='x', kw_only=True)
    # But meanwhile, need an actual __init__ method:

    def __init__(
        self,
        extractor_factory: Callable[[Any], Callable],
        extractor_params: Any,
        *,
        name: str = "extractor",
        input_name: str = "x",
    ):
        self.extractor_factory = extractor_factory
        self.extractor_params = extractor_params
        self.name = name
        self.input_name = input_name
        self.__post_init__()

    def __post_init__(self):
        self.__name__ = self.name
        self.__signature__ = Sig(f"({self.input_name}, /)")
        self._call = self.extractor_factory(self.extractor_params)

    def __call__(self, x):
        return self._call(x)


def _itemgetter(items):
    if isinstance(items, str):
        items = [items]
    return itemgetter(*items)


def _attrgetter(attrs):
    if isinstance(attrs, str):
        attrs = [attrs]
    return attrgetter(*attrs)


Itemgetter = partial(Extractor, _itemgetter, name="itemgetter")
AttrGetter = partial(Extractor, _attrgetter, name="attrgetter")
```

## composition.py

```python
"""Specific use of FuncNode and DAG"""

from dataclasses import fields
from inspect import signature
from typing import Union
from collections.abc import Callable
from functools import partial

from i2 import Sig, kwargs_trans
from meshed.base import FuncNode, func_node_transformer
from meshed.dag import DAG
from meshed.util import Renamer, numbered_suffix_renamer, InvalidFunctionParameters

_func_node_fields = {x.name for x in fields(FuncNode)}

# TODO: How to enforce that it only has keys from _func_node_fields?
FuncNodeKwargs = dict
FuncNodeKwargsTrans = Callable[[FuncNodeKwargs], FuncNodeKwargs]


def is_func_node_kwargs_trans(func: Callable) -> bool:
    """Returns True iff the only required params of func are FuncNode field names.
    This ensures that the func will be able to be bound to FuncNode fields and
    therefore used as a func_node (kwargs) transformer.
    """
    return _func_node_fields.issuperset(Sig(func).required_names)


def func_node_kwargs_trans(func: Callable) -> FuncNodeKwargsTrans:
    if is_func_node_kwargs_trans(func):
        return func
    else:
        raise InvalidFunctionParameters(
            f"A FuncNodeKwargsTrans is expected to only have required params that are "
            f"also "
            f"FuncNode fields ({', '.join(_func_node_fields)}). "
            "Function {func} had signature: {Sig(func)}"
        )


def func_node_name_trans(
    name_trans: Callable[[str], str | None],
    *,
    also_apply_to_func_label: bool = False,
):
    """

    :param name_trans: A function taking a str and returning a str, or None (to indicate
        that no transformation should take place).
    :param also_apply_to_func_label:
    :return:
    """
    if not is_func_node_kwargs_trans(name_trans):
        if Sig(name_trans).n_required <= 1:

            def name_trans(name):
                return name_trans(name)

    kwargs_trans_kwargs = dict(name=name_trans)
    if also_apply_to_func_label:
        kwargs_trans_kwargs.update(func_label=name_trans)

    return partial(
        func_node_transformer,
        kwargs_transformers=partial(kwargs_trans, **kwargs_trans_kwargs),
    )


# TODO: Extract  ingress/egress boilerplate to wrapper
def suffix_ids(
    func_nodes,
    renamer: Renamer | str = numbered_suffix_renamer,
    *,
    also_apply_to_func_label: bool = False,
):
    if isinstance(func_nodes, DAG):
        egress = DAG
    else:
        egress = list
    if isinstance(renamer, str):
        suffix = renamer
        renamer = lambda name: f"{name}{suffix}"
    assert callable(suffix), f"suffix needs to be callable"
    func_node_trans = func_node_name_trans(
        renamer,
        also_apply_to_func_label=also_apply_to_func_label,
    )
    return egress(map(func_node_trans, func_nodes))


# ---------------------------------------------------------------------------------------
# lined, with meshed


def get_param(func):
    """
    Find the name of the parameter of a function with exactly one parameter.
    Raise an error if more or less parameters.
    :param func: callable, the function to inspect
    :return: str, the name of the single parameter of func
    """

    params = signature(func).parameters.keys()
    assert (
        len(params) == 1
    ), f"Your function has more than 1 parameter! Namely: {', '.join(params)}"
    for param in params:
        return param


def line_with_dag(*steps):
    """
    Emulate a Line object with a DAG
    :param steps: an iterable of callables, the steps of the pipeline. Each step should have exactly one parameter
    and the output of each step is fed into the next
    :return: a DAG instance computing the composition of all the functions in steps, in the provided order
    """

    step_counter = 0
    first_node = FuncNode(steps[0], out=f"step_{step_counter}")
    func_nodes = [first_node]
    for step in steps[1:]:
        step_node = FuncNode(
            step,
            out=f"step_{step_counter + 1}",
            bind={get_param(step): f"step_{step_counter}"},
        )
        step_counter += 1
        func_nodes.append(step_node)

    return DAG(func_nodes)
```

## dag.py

```python
"""
Making DAGs

In it's simplest form, consider this:

>>> from meshed import DAG
>>>
>>> def this(a, b=1):
...     return a + b
...
>>> def that(x, b=1):
...     return x * b
...
>>> def combine(this, that):
...     return (this, that)
...
>>>
>>> dag = DAG((this, that, combine))
>>> print(dag.synopsis_string())
a,b -> this_ -> this
x,b -> that_ -> that
this,that -> combine_ -> combine

But don't be fooled: There's much more to it!


FAQ and Troubleshooting
=======================

DAGs and Pipelines
------------------

>>> from functools import partial
>>> from meshed import DAG
>>> def chunker(sequence, chk_size: int):
...     return zip(*[iter(sequence)] * chk_size)
>>>
>>> my_chunker = partial(chunker, chk_size=3)
>>>
>>> vec = range(8)  # when appropriate, use easier to read sequences
>>> list(my_chunker(vec))
[(0, 1, 2), (3, 4, 5)]

Oh, that's just a ``my_chunker -> list`` pipeline!
A pipeline is a subset of DAG, so let me do this:

>>> dag = DAG([my_chunker, list])
>>> dag(vec)
Traceback (most recent call last):
...
TypeError: missing a required argument: 'sequence'

What happened here?
You're assuming that saying ``[my_chunker, list]`` is enough for DAG to know that
what you meant is for ``my_chunker`` to feed it's input to ``list``.
Sure, DAG has enough information to do so, but the default connection policy doesn't
assume that it's a pipeline you want to make.
In fact, the order you specify the functions doesn't have an affect on the connections
with the default connection policy.

See what the signature of ``dag`` is:

>>> from inspect import signature
>>> str(signature(dag))
'(iterable=(), /, sequence, *, chk_size: int = 3)'

So dag actually works just fine. Here's the proof:

>>> dag([1,2,3], vec)  # doctest: +SKIP
([1, 2, 3], <zip object at 0x104d7f080>)

It's just not what you might have intended.

Your best bet to get what you intended is to be explicit.

The way to be explicit is to not specify functions alone, but ``FuncNodes`` that
wrap them, along with the specification
the ``name`` the function will be referred to by,
the names that it's parameters should ``bind`` to (that is, where the function
will get it's import arguments from), and
the ``out`` name of where it should be it's output.

In the current case a fully specified DAG would look something like this:

>>> from meshed import FuncNode
>>> dag = DAG(
...     [
...         FuncNode(
...             func=my_chunker,
...             name='chunker',
...             bind=dict(sequence='sequence', chk_size='chk_size'),
...             out='chks'
...         ),
...         FuncNode(
...             func=list,
...             name='gather_chks_into_list',
...             bind=dict(iterable='chks'),
...             out='list_of_chks'
...         ),
...     ]
... )
>>> list(dag(vec))
[(0, 1, 2), (3, 4, 5)]

But really, if you didn't care about the names of things,
all you need in this case was to make sure that the output of ``my_chunker`` was
fed to ``list``, and therefore the following was sufficient:

>>> dag = DAG([
...     FuncNode(my_chunker, out='chks'),  # call the output of chunker "chks"
...     FuncNode(list, bind=dict(iterable='chks'))  # source list input from "chks"
... ])
>>> list(dag(vec))
[(0, 1, 2), (3, 4, 5)]

Connection policies are very useful when you want to define ways for DAG to
"just figure it out" for you.
That is, you want to tell the machine to adapt to your thoughts, not vice versa.
We support such technological expectations!
The default connection policy is there to provide one such ways, but
by all means, use another!

Does this mean that connection policies are not for production code?
Well, it depends. The Zen of Python (``import this``)
states "explicit is better than implicit", and indeed it's often
a good fallback rule.
But defining components and the way they should be assembled can go a long way
in achieving consistency, separation of concerns, adaptability, and flexibility.
All quite useful things. Also in production. Especially in production.
That said it is your responsiblity to use the right policy for your particular context.

"""

from functools import partial, wraps, cached_property
from collections import defaultdict

from dataclasses import dataclass, field
from itertools import chain
from operator import attrgetter, eq
from typing import (
    Union,
    Optional,
    Any,
    Tuple,
    KT,
    VT,
)
from collections.abc import Callable, MutableMapping, Iterable, Mapping
from warnings import warn

from i2 import double_up_as_factory, MultiFunc
from i2.signatures import (
    call_somewhat_forgivingly,
    call_forgivingly,
    Parameter,
    empty,
    Sig,
    sort_params,
    # SignatureComparator,
    CallableComparator,
)
from meshed.base import (
    FuncNode,
    dflt_configs,
    BindInfo,
    ch_func_node_func,
    ensure_func_nodes,
    _func_nodes_to_graph_dict,
    is_func_node,
    FuncNodeAble,
    func_node_transformer,
    # compare_signatures,
)

from meshed.util import (
    lambda_name,
    ValidationError,
    NotUniqueError,
    NotFound,
    NameValidationError,
    Renamer,
    _if_none_return_input,
    numbered_suffix_renamer,
    replace_item_in_iterable,
    InvalidFunctionParameters,
    extract_values,
    extract_items,
    ParameterMerger,
    conservative_parameter_merge,
)
from meshed.itools import (
    topological_sort,
    leaf_nodes,
    root_nodes,
    descendants,
    ancestors,
)

from meshed.viz import dot_lines_of_objs, add_new_line_if_none

FuncMapping = Union[Mapping[KT, Callable], Iterable[tuple[KT, Callable]]]


def order_subset_from_list(items, sublist):
    assert set(sublist).issubset(set(items)), f"{sublist} is not contained in {items}"
    d = {k: v for v, k in enumerate(items)}

    return sorted(sublist, key=lambda x: d[x])


def find_first_free_name(prefix, exclude_names=(), start_at=2):
    if prefix not in exclude_names:
        return prefix
    else:
        i = start_at
        while True:
            name = f"{prefix}__{i}"
            if name not in exclude_names:
                return name
            i += 1


def mk_mock_funcnode(arg, out):
    @Sig(arg)
    def func():
        pass

    # name = "_mock_" + str(arg) + "_" + str(out)  # f-string
    name = f"_mock_{str(arg)}_{str(out)}"  # f-string

    return FuncNode(func=func, out=out, name=name)


def mk_func_name(func, exclude_names=()):
    name = getattr(func, "__name__", "")
    if name == "<lambda>":
        name = lambda_name()  # make a lambda name that is a unique identifier
    elif name == "":
        if isinstance(func, partial):
            return mk_func_name(func.func, exclude_names)
        else:
            raise NameValidationError(f"Can't make a name for func: {func}")
    return find_first_free_name(name, exclude_names)


def mk_list_names_unique(nodes, exclude_names=()):
    names = [node.name for node in nodes]

    def gen():
        _exclude_names = exclude_names
        for name in names:
            if name not in _exclude_names:
                yield name
                _exclude_names = _exclude_names + (name,)
            else:
                found_name = find_first_free_name(f"{name}", _exclude_names)
                yield found_name
                _exclude_names = _exclude_names + (found_name,)

    return list(gen())


def mk_nodes_names_unique(nodes):
    new_names = mk_list_names_unique(nodes)
    for node, new_name in zip(nodes, new_names):
        node.name = new_name
    return nodes


def arg_names(func, func_name, exclude_names=()):
    names = Sig(func).names

    def gen():
        _exclude_names = exclude_names
        for name in names:
            if name not in _exclude_names:
                yield name
            else:
                found_name = find_first_free_name(
                    f"{func_name}__{name}", _exclude_names
                )
                yield found_name
                _exclude_names = _exclude_names + (found_name,)

    return list(gen())


def named_partial(func, *args, __name__=None, **keywords):
    """functools.partial, but with a __name__

    >>> f = named_partial(print, sep='\\n')
    >>> f.__name__
    'print'

    >>> f = named_partial(print, sep='\\n', __name__='now_partial_has_a_name')
    >>> f.__name__
    'now_partial_has_a_name'
    """
    f = partial(func, *args, **keywords)
    f.__name__ = __name__ or func.__name__
    return f


from i2.signatures import ch_func_to_all_pk


def hook_up(func, variables: MutableMapping, output_name=None):
    """Source inputs and write outputs to given variables mapping.

    Returns inputless and outputless function that will, when called,
    get relevant inputs from the provided variables mapping and write it's
    output there as well.

    :param variables: The MutableMapping (like... a dict) where the function
    should both read it's input and write it's output.
    :param output_name: The key of the variables mapping that should be used
    to write the output of the function
    :return: A function

    >>> def formula1(w, /, x: float, y=1, *, z: int = 1):
    ...     return ((w + x) * y) ** z

    >>> d = {}
    >>> f = hook_up(formula1, d)
    >>> # NOTE: update d, not d = dict(...), which would make a DIFFERENT d
    >>> d.update(w=2, x=3, y=4)  # not d = dict(w=2, x=3, y=4), which would
    >>> f()

    Note that there's no output. The output is in d
    >>> d
    {'w': 2, 'x': 3, 'y': 4, 'formula1': 20}

    Again...

    >>> d.clear()
    >>> d.update(w=1, x=2, y=3)
    >>> f()
    >>> d['formula1']
    9

    """
    _func = ch_func_to_all_pk(func)  # makes a position-keyword copy of func
    output_key = output_name
    if output_name is None:
        output_key = _func.__name__

    def source_from_decorated():
        variables[output_key] = call_somewhat_forgivingly(_func, (), variables)

    return source_from_decorated


def _separate_func_nodes_and_var_nodes(nodes):
    func_nodes = list()
    var_nodes = list()
    for node in nodes:
        if is_func_node(node):
            func_nodes.append(node)
        else:
            var_nodes.append(node)
    return func_nodes, var_nodes


_not_found = object()


def _find_unique_element(item, search_iterable, key: Callable[[Any, Any], bool]):
    """Find item in search_iterable, using key as the matching function,
    raising a NotFound error if no match and a NotUniqueError if more than one."""
    it = filter(lambda x: key(item, x), search_iterable)
    first = next(it, _not_found)
    if first == _not_found:
        raise NotFound(f"{item} wasn't found")
    else:
        the_next_match = next(it, _not_found)
        if the_next_match is not _not_found:
            raise NotUniqueError(f"{item} wasn't unique")
    return first


def modified_func_node(func_node, **modifications) -> FuncNode:
    modifiable_attrs = {"func", "name", "bind", "out"}
    assert not modifications.keys().isdisjoint(
        modifiable_attrs
    ), f"Can only modify these: {', '.join(modifiable_attrs)}"
    original_func_node_kwargs = {
        "func": func_node.func,
        "name": func_node.name,
        "bind": func_node.bind,
        "out": func_node.out,
    }
    return FuncNode(**dict(original_func_node_kwargs, **modifications))


from i2 import partialx


# TODO: doctests
def partialized_funcnodes(func_nodes, **keyword_defaults):
    for func_node in func_nodes:
        if argnames_to_be_bound := set(keyword_defaults).intersection(
            func_node.sig.names
        ):
            bindings = dict(extract_items(keyword_defaults, argnames_to_be_bound))
            # partialize the func and move defaulted params to the end
            partialized_func = partialx(
                func_node.func, **bindings, _allow_reordering=True
            )
            # get rid of kinds  # TODO: This is a bit extreme -- consider gentler touch
            nice_kinds_sig = Sig(partialized_func).ch_kinds_to_position_or_keyword()
            nice_kinds_partialized_func = nice_kinds_sig(partialized_func)
            yield modified_func_node(
                func_node, func=nice_kinds_partialized_func
            )  # TODO: A better way without partial?
        else:
            yield func_node


Scope = dict
VarNames = Iterable[str]
DagOutput = Any


def _name_attr_or_x(x):
    return getattr(x, "name", x)


def change_value_on_cond(d, cond, func):
    for k, v in d.items():
        if cond(k, v):
            d[k] = func(v)
    return d


def dflt_debugger_feedback(func_node, scope, output, step):
    print(f"{step} --------------------------------------------------------------")
    print(f"\t{func_node=}\n\t{scope=}")
    return output


# TODO: caching last scope isn't really the DAG's direct concern -- it's a debugging
#  concern. Perhaps a more general form would be to define a cache factory defaulting
#  to a dict, but that could be a "dict" that logs writes (even to an attribute of self)
@dataclass
class DAG:
    """
    >>> from meshed.dag import DAG, Sig
    >>>
    >>> def this(a, b=1):
    ...     return a + b
    >>> def that(x, b=1):
    ...     return x * b
    >>> def combine(this, that):
    ...     return (this, that)
    >>>
    >>> dag = DAG((this, that, combine))
    >>> print(dag.synopsis_string())
    a,b -> this_ -> this
    x,b -> that_ -> that
    this,that -> combine_ -> combine

    But what does it do?

    It's a callable, with a signature:

    >>> Sig(dag)  # doctest: +SKIP
    <Sig (a, x, b=1)>

    And when you call it, it executes the dag from the root values you give it and
    returns the leaf output values.

    >>> dag(1, 2, 3)  # (a+b,x*b) == (1+3,2*3) == (4, 6)
    (4, 6)
    >>> dag(1, 2)  # (a+b,x*b) == (1+1,2*1) == (2, 2)
    (2, 2)

    The above DAG was created straight from the functions, using only the names of the
    functions and their arguments to define how to hook the network up.

    But if you didn't write those functions specifically for that purpose, or you want
    to use someone else's functions, we got you covered.

    You can define the name of the node (the `name` argument), the name of the output
    (the `out` argument) and a mapping from the function's arguments names to
    "network names" (through the `bind` argument).
    The edges of the DAG are defined by matching `out` TO `bind`.

    """

    func_nodes: Iterable[FuncNode | Callable] = ()
    cache_last_scope: bool = field(default=True, repr=False)
    parameter_merge: ParameterMerger = field(
        default=conservative_parameter_merge, repr=False
    )
    # can return a prepopulated scope too!
    new_scope: Callable = field(default=dict, repr=False)
    name: str = None
    extract_output_from_scope: Callable[[Scope, VarNames], DagOutput] = field(
        default=extract_values, repr=False
    )

    def __post_init__(self):
        self.func_nodes = tuple(ensure_func_nodes(self.func_nodes))
        self.graph = _func_nodes_to_graph_dict(self.func_nodes)
        self.nodes = topological_sort(self.graph)
        # reorder the nodes to fit topological order
        self.func_nodes, self.var_nodes = _separate_func_nodes_and_var_nodes(self.nodes)
        # self.sig = Sig(dict(extract_items(sig.parameters, 'xz')))
        self.__signature__ = Sig(  # make a signature
            sort_params(  # with the sorted params (sorted to satisfy kind/default order)
                self.src_name_params(root_nodes(self.graph))
            )
        )

        # self.__signature__(self)  # to put the signature on the callable DAG
        # figure out the roots and leaves
        self.roots = tuple(
            self.__signature__.names
        )  # roots in the same order as signature
        leafs = leaf_nodes(self.graph)
        # But we want leafs in topological order
        self.leafs = tuple([name for name in self.nodes if name in leafs])
        self.last_scope = None
        self.__name__ = self.name or "DAG"

        self.bindings_cleaner()

    # TODO: No control of other DAG args (cache_last_scope etc.).
    @classmethod
    def from_funcs(cls, *funcs, **named_funcs):
        """

        :param funcs:
        :param named_funcs:
        :return:

        >>> dag = DAG.from_funcs(
        ...     lambda a: a * 2,
        ...     x=lambda: 10,
        ...     y=lambda x, _0: x + _0  # _0 refers to first arg (lambda a: a * 2)
        ... )
        >>> print(dag.synopsis_string())
        a -> _0_ -> _0
         -> x_ -> x
        x,_0 -> y_ -> y
        >>> dag(3)
        16

        """
        named_funcs = dict(MultiFunc(*funcs, **named_funcs))
        func_nodes = [
            FuncNode(name=name, func=f, out=name) for name, f in named_funcs.items()
        ]
        return cls(func_nodes)

    def bindings_cleaner(self):
        self.func_nodes = mk_nodes_names_unique(self.func_nodes)
        funcnodes_names = [node.name for node in self.func_nodes]
        func = lambda v: self._func_node_for[v].out
        cond = lambda k, v: v in funcnodes_names
        for node in self.func_nodes:
            node.bind = change_value_on_cond(node.bind, cond, func)

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def _get_kwargs(self, *args, **kwargs):
        """
        Get a dict of {argname: argval} pairs from positional and keyword arguments.
        """
        return self.__signature__.map_arguments(args, kwargs, apply_defaults=True)

    def _call(self, *args, **kwargs):
        # Get a dict of {argname: argval} pairs from positional and keyword arguments
        # How positionals are resolved is determined by self.__signature__
        # The result is the initial ``scope`` the func nodes will both read from
        # to get their arguments, and write their outputs to.
        scope = self._get_kwargs(*args, **kwargs)
        # Go through self.func_nodes in order and call them on scope (performing said
        # read_input -> call_func -> write_output operations)
        self.call_on_scope(scope)
        # From the scope, that may contain all intermediary results,
        # extract the desired final output and return it
        return self.extract_output_from_scope(scope, self.leafs)

    def _preprocess_scope(self, scope):
        """Take care of the stuff that needs to be taking care of before looping
        though the func_nodes and calling them on scope. Namely:

        - If scope is None, create a new one calling self.new_scope()
        - If self.cache_last_scope is True, remember the scope in self.last_scope

        """
        if scope is None:
            scope = self.new_scope()  # fresh new scope
        if self.cache_last_scope:
            self.last_scope = scope  # just to remember it, for debugging purposes ONLY!
        return scope

    def _call_func_nodes_on_scope_gen(self, scope):
        """Loop over ``func_nodes`` yielding ``func_node.call_on_scope(scope)``."""
        for func_node in self.func_nodes:
            yield func_node.call_on_scope(scope)

    def _call_func_nodes_on_scope(self, scope):
        """
        Loop over ``func_nodes`` calling func_node.call_on_scope on scope.
        (Really, just "consumes" the generator output by _call_func_nodes_on_scope_gen)
        """
        for _ in self._call_func_nodes_on_scope_gen(scope):
            pass

    def call_on_scope(self, scope=None):
        """Calls the func_nodes using scope (a dict or MutableMapping) both to
        source it's arguments and write it's results.

        Note: This method is only meant to be used as a backend to __call__, not as
        an actual interface method. Additional control/constraints on read and writes
        can be implemented by providing a custom scope for that. For example, one could
        log read and/or writes to specific keys, or disallow overwriting to an existing
        key (useful for pipeline sanity), etc.
        """
        scope = self._preprocess_scope(scope)
        self._call_func_nodes_on_scope(scope)

    def call_on_scope_iteratively(self, scope=None):
        """Calls the ``func_nodes`` using scope (a dict or MutableMapping) both to
        source it's arguments and write it's results.

        Use this function to control each func_node call step iteratively
        (through a generator)
        """
        scope = self._preprocess_scope(scope)
        yield from self._call_func_nodes_on_scope_gen(scope)

    # def clone(self, *args, **kwargs):
    #     """Use args, kwargs to make an instance, using self attributes for
    #     unspecified arguments.
    #     """

    def __getitem__(self, item):
        """Get a sub-dag from a specification of (var or fun) input and output nodes.

        ``dag[input_nodes:output_nodes]`` is the sub-dag made of intersection of all
        descendants of ``input_nodes``
        (inclusive) and ancestors of ``output_nodes`` (inclusive), where additionally,
        when a func node is contained, it takes with it the input and output nodes
        it needs.

        >>> def f(a): ...
        >>> def g(f): ...
        >>> def h(g): ...
        >>> def i(h): ...
        >>> dag = DAG([f, g, h, i])

        See what this dag looks like (it's a simple pipeline):

        >>> dag = DAG([f, g, h, i])
        >>> print(dag.synopsis_string())
        a -> f_ -> f
        f -> g_ -> g
        g -> h_ -> h
        h -> i_ -> i

        Get a subdag from ``g_`` (indicates the function here) to the end of ``dag``

        >>> subdag = dag['g_':]
        >>> print(subdag.synopsis_string())
        f -> g_ -> g
        g -> h_ -> h
        h -> i_ -> i

        From the beginning to ``h_``

        >>> print(dag[:'h_'].synopsis_string())
        a -> f_ -> f
        f -> g_ -> g
        g -> h_ -> h

        From ``g_`` to ``h_`` (both inclusive)

        >>> print(dag['g_':'h_'].synopsis_string())
        f -> g_ -> g
        g -> h_ -> h

        Above we used function (node names) to specify what we wanted, but we can also
        use names of input/output var-nodes. Do note the difference though.
        The nodes you specify to get a sub-dag are INCLUSIVE, but when you
        specify function nodes, you also get the input and output nodes of these
        functions.

        The ``dag['g_', 'h_']`` give us a sub-dag starting at ``f`` (the input node),
        but when we ask ``dag['g', 'h_']`` instead, ``g`` being the output node of
        function node ``g_``, we only get ``g -> h_ -> h``:

        >>> print(dag['g':'h'].synopsis_string())
        g -> h_ -> h

        If we wanted to include ``f`` we'd have to specify it:

        >>> print(dag['f':'h'].synopsis_string())
        f -> g_ -> g
        g -> h_ -> h

        Those were for simple pipelines, but let's now look at a more complex dag.

        We'll let the following examples self-comment:

        >>> def f(u, v): ...
        >>> def g(f): ...
        >>> def h(f, w): ...
        >>> def i(g, h): ...
        >>> def j(h, x): ...
        >>> def k(i): ...
        >>> def l(i, j): ...
        >>> dag = DAG([f, g, h, i, j, k, l])
        >>> print(dag.synopsis_string())
        u,v -> f_ -> f
        f -> g_ -> g
        f,w -> h_ -> h
        g,h -> i_ -> i
        h,x -> j_ -> j
        i -> k_ -> k
        i,j -> l_ -> l

        A little util to get consistent prints:

        >>> def print_sorted_synopsis(dag):
        ...     t = sorted(dag.synopsis_string().split('\\n'))
        ...     print('\\n'.join(t))

        >>> print_sorted_synopsis(dag[['u', 'f']:'h'])
        f,w -> h_ -> h
        u,v -> f_ -> f
        >>> print_sorted_synopsis(dag['u':'h'])
        f,w -> h_ -> h
        u,v -> f_ -> f
        >>> print_sorted_synopsis(dag[['u', 'f']:['h', 'g']])
        f -> g_ -> g
        f,w -> h_ -> h
        u,v -> f_ -> f
        >>> print_sorted_synopsis(dag[['x', 'g']:'k'])
        g,h -> i_ -> i
        i -> k_ -> k
        >>> print_sorted_synopsis(dag[['x', 'g']:['l', 'k']])
        g,h -> i_ -> i
        h,x -> j_ -> j
        i -> k_ -> k
        i,j -> l_ -> l

        >>>

        """
        return self._getitem(item)

    def _getitem(self, item):
        return DAG(
            func_nodes=self._ordered_subgraph_nodes(item),
            cache_last_scope=self.cache_last_scope,
            parameter_merge=self.parameter_merge,
        )

    def _ordered_subgraph_nodes(self, item):
        subgraph_nodes = self._subgraph_nodes(item)
        # TODO: When clone ready, use to do `constructor = type(self)` instead of DAG
        # constructor = type(self)  # instead of DAG
        initial_nodes = self.func_nodes
        ordered_subgraph_nodes = order_subset_from_list(initial_nodes, subgraph_nodes)
        return ordered_subgraph_nodes

    def _subgraph_nodes(self, item):
        ins, outs = self.process_item(item)
        _descendants = set(
            filter(FuncNode.has_as_instance, set(ins) | descendants(self.graph, ins))
        )
        _ancestors = set(
            filter(FuncNode.has_as_instance, set(outs) | ancestors(self.graph, outs))
        )
        subgraph_nodes = _descendants.intersection(_ancestors)
        return subgraph_nodes

    # TODO: Think about adding a ``_roll_in_orphaned_nodes=False`` argument:
    #   See https://github.com/i2mint/meshed/issues/14 for more information.
    def partial(
        self,
        *positional_dflts,
        _remove_bound_arguments=False,
        _consider_defaulted_arguments_as_bound=False,
        **keyword_dflts,
    ):
        """Get a curried version of the DAG.

        Like ``functools.partial``, but returns a DAG (not just a callable) and allows
        you to remove bound arguments as well as roll in orphaned_nodes.

        :param positional_dflts: Bind arguments positionally
        :param keyword_dflts: Bind arguments through their names
        :param _remove_bound_arguments: False -- set to True if you don't want bound
            arguments to show up in the signature.
        :param _consider_defaulted_arguments_as_bound: False -- set to True if
            you want to also consider arguments that already had defaults as bound
            (and be removed).
        :return:

        >>> def f(a, b):
        ...     return a + b
        >>> def g(c, d=4):
        ...     return c * d
        >>> def h(f, g):
        ...     return g - f
        >>> dag = DAG([f, g, h])
        >>> from inspect import signature
        >>> str(signature(dag))
        '(a, b, c, d=4)'
        >>> dag(1, 2, 3, 4)  # == (3 * 4) - (1 + 2) == 12 - 3 == 9
        9
        >>> dag(c=3, a=1, b=2, d=4)  # same as above
        9

        >>> new_dag = dag.partial(c=3)
        >>> isinstance(new_dag, DAG)  # it's a dag (not just a partialized callable!)
        True
        >>> str(signature(new_dag))
        '(a, b, c=3, d=4)'
        >>> new_dag(1, 2)  # same as dag(c=3, a=1, b=2, d=4), so:
        9
        """
        keyword_dflts = self.__signature__.map_arguments(
            args=positional_dflts,
            kwargs=keyword_dflts,
            apply_defaults=_consider_defaulted_arguments_as_bound,
            # positional_dflts and keyword_dflts usually don't cover all arguments, so:
            allow_partial=True,
            # we prefer to let the user know if they're trying to bind arguments
            # that don't exist, so:
            allow_excess=False,
            # we don't really care about kind, so:
            ignore_kind=True,
        )
        # TODO: mk_instance: What about other init args (cache_last_scope, ...)?
        mk_instance = type(self)
        func_nodes = partialized_funcnodes(self, **keyword_dflts)
        new_dag = mk_instance(func_nodes)
        if _remove_bound_arguments:
            new_sig = Sig(new_dag).remove_names(list(keyword_dflts))
            new_sig(new_dag)  # Change the signature of new_dag with bound args removed
        return new_dag

    def process_item(self, item):
        assert isinstance(item, slice), f"must be a slice, was: {item}"

        input_names, outs = item.start, item.stop

        empty_slice = slice(None)

        def ensure_variable_list(obj):
            if obj is None:
                return self.var_nodes
            if isinstance(obj, str):
                obj = obj.split()
            if isinstance(obj, (str, Callable)):
                # TODO: See if we can use _func_node_for instead
                return [self.get_node_matching(obj)]
            elif isinstance(obj, Iterable):
                # TODO: See if we can use _func_node_for instead
                return list(map(self.get_node_matching, obj))
            else:
                raise ValidationError(f"Unrecognized variables specification: {obj}")

        # assert len(item) == 2, f"Only items of size 1 or 2 are supported"
        input_names, outs = map(ensure_variable_list, [input_names, outs])
        return input_names, outs

    def get_node_matching(self, idx):
        if isinstance(idx, str):
            if idx in self.var_nodes:
                return idx
            return self._func_node_for[idx]
        elif isinstance(idx, Callable):
            return self._func_node_for[idx]
        raise NotFound(f"No matching node for idx: {idx}")

    # TODO: Reflect: Should we include functions as keys here? Makes existence of the
    #  item depend on unicity of the function in the DAG, therefore dynamic,
    #  so instable?
    #  Should this node indexing be controllable by user?
    @cached_property
    def _func_node_for(self):
        """A dictionary mapping identifiers and functions to their FuncNode instances
        in the DAG. The keys of this dictionary will include:

        - identifiers (names) of the ``FuncNode`` instances
        - ``out`` of ``FuncNode`` instances
        - The ``.func`` of the ``FuncNode`` instances if it's unique.

        >>> def foo(x): return x + 1
        >>> def bar(x): return x * 2
        >>> dag = DAG([
        ...     FuncNode(foo, out='foo_output'),
        ...     FuncNode(bar, name='B', out='b', bind={'x': 'foo_output'}),
        ... ])

        A ``FuncNode`` instance is indexed by both its identifier (``.name``) as well as
        the identifier of it's output (``.out``):

        >>> dag._func_node_for['foo_output']
        FuncNode(x -> foo -> foo_output)
        >>> dag._func_node_for['foo']
        FuncNode(x -> foo -> foo_output)
        >>> dag._func_node_for['b']
        FuncNode(x=foo_output -> B -> b)
        >>> dag._func_node_for['B']
        FuncNode(x=foo_output -> B -> b)

        If the function is hashable (most are) and unique within the ``DAG``, you
        can also find the ``FuncNode`` via the ``.func`` it's wrapping:

        >>> dag._func_node_for[foo]
        FuncNode(x -> foo -> foo_output)
        >>> dag._func_node_for[bar]
        FuncNode(x=foo_output -> B -> b)

        A word of warning though: The function index is provided as a convenience, but
        using identifiers is preferable since referencing via the function object
        depends on the other functions of the DAG, so could change if we add nodes.

        """
        d = dict()
        for func_node in self.func_nodes:
            d[func_node.out] = func_node
            d[func_node.name] = func_node

            try:
                if func_node.func not in d:
                    # if .func not in d already, remember the link
                    d[func_node.func] = func_node
                else:
                    # if .func was already in there, mark it for removal
                    # (but leaving the key present so that we know about the duplication)
                    d[func_node.func] = None
            except TypeError:
                # ignore (and don't include func) if not hashable
                pass

        # remove the items marked for removal and return
        return {k: v for k, v in d.items() if v is not None}

    def find_func_node(self, node, default=None):
        if isinstance(node, FuncNode):
            return node
        return self._func_node_for.get(node, default)

    def __iter__(self):
        """Yields the self.func_nodes
        Note: The raison d'etre of this ``__iter__`` is simply because if no custom one
        is provided, python defaults to yielding ``__getitem__[i]`` for integers,
        which leads to an error being raised.

        At least here we yield something sensible.

        A consequence of the `__iter__` being the iterable of func_nodes is that we
        can extend dags using the star operator. Consider the following dag;

        >>> def f(a): return a * 2
        >>> def g(f, b): return f + b
        >>> dag = DAG([f, g])
        >>> assert dag(2, 3) == 7

        Say you wanted to now take a, b, and the output og g, and feed it to another
        function...

        >>> def h(a, b, g): return f"{a=} {b=} {g=}"
        >>> extended_dag = DAG([*dag, h])
        >>> extended_dag(a=2, b=3)
        'a=2 b=3 g=7'
        """
        yield from self.func_nodes

    # Note: signature_comparator is position only to not conflict with any of the
    #  func_mapping keys.
    def ch_funcs(
        self,
        # func_comparator: CallableComparator = compare_signatures,
        ch_func_node_func: Callable[
            [FuncNode, Callable, CallableComparator], FuncNode
        ] = ch_func_node_func,
        /,
        **func_mapping: Callable,
    ) -> "DAG":
        """
        Change some of the functions in the DAG.
        More preciseluy get a copy of the DAG where in some of the functions have
        changed.

        :param name_and_func: ``name=func`` pairs where ``name`` is the
            ``FuncNode.name`` of the func nodes you want to change and func is the
            function you want to change it by.
        :return: A new DAG with the different functions.

        >>> from meshed import FuncNode, DAG
        >>> from i2 import Sig
        >>>
        >>> def f(a, b):
        ...     return a + b
        ...
        >>>
        >>> def g(a_plus_b, x):
        ...     return a_plus_b * x
        ...
        >>> f_node = FuncNode(func=f, out='a_plus_b')
        >>> g_node = FuncNode(func=g, bind={'x': 'b'})
        >>> d = DAG((f_node, g_node))
        >>> print(d.synopsis_string())
        a,b -> f -> a_plus_b
        b,a_plus_b -> g_ -> g
        >>> d(2, 3)  # (2 + 3) * 3 == 5 * 3
        15
        >>> dd = d.ch_funcs(f=lambda a, b: a - b)
        >>> dd(2, 3)  # (2 - 3) * 3 == -1 * 3
        -3

        You can reference the ``FuncNode`` you want to change through its ``.name`` or
        ``.out`` attribute (both are unique to this ``FuncNode`` in a ``DAG``).

        >>> from i2 import Sig
        >>>
        >>> dag = DAG([
        ...     FuncNode(lambda a, b: a + b, name='f'),
        ...     FuncNode(lambda y=1, z=2: y * z, name='g', bind={'z': 'f'})
        ... ])
        >>>
        >>> Sig(dag)
        <Sig (a, b, f=2, y=1)>
        >>>
        >>> dag.ch_funcs(g=lambda y=1, z=2: y / z)
        DAG(func_nodes=[FuncNode(a,b -> f -> _f), FuncNode(z=_f,y -> g -> _g)], name=None)

        But if you change the signature, even slightly you get an error.

        Here we didn't include the defaults:

        >>> dag.ch_funcs(g=lambda y, z: y / z)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: You can only change the func of a FuncNode with a another func if the signatures match.
          ...

        Here we include defaults, but ``z``'s is different:

        >>> dag.ch_funcs(g=lambda y=1, z=200: y / z)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: You can only change the func of a FuncNode with a another func if the signatures match.
          ...

        Here the defaults are exactly the same, but the order of parameters is
        different:

        >>> dag.ch_funcs(g=lambda z=2, y=1: y / z)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        Traceback (most recent call last):
          ...
        ValueError: You can only change the func of a FuncNode with a another func if the signatures match.
          ...

        This validation of the functions controlled by the ``func_comparator``
        argument. By default this is the ``compare_signatures`` which compares the
        signatures of the functions in the strictest way possible.
        The is the right choice for a default since it will get you out of trouble
        down the line.

        But it's also annoying in many situations, and in those cases you should
        specify the ``func_comparator`` that makes sense for your context.

        Since most of the time, you'll want to compare functions solely based on
        their signature, we provide a ``compare_signatures`` allows you to control the
        signature comparison through a ``signature_comparator`` argument.

        >>> from meshed import compare_signatures
        >>> from functools import partial
        >>> on_names = lambda sig1, sig2: list(sig1.parameters) == list(sig2.parameters)
        >>> same_names = partial(compare_signatures, signature_comparator=on_names)
        >>> ch_fnode = partial(ch_func_node_func, func_comparator=same_names)
        >>> d = dag.ch_funcs(ch_fnode, g=lambda y, z: y / z);
        >>> Sig(d)
        <Sig (a, b, y)>
        >>> d(2, 3, 4)
        0.8

        And this one works too:

        >>> d = dag.ch_funcs(ch_fnode, g=lambda y=1, z=200: y / z);

        But our ``same_names`` function compared names including their order.
        If we want a function with the signature ``(z=2, y=1)`` to be able to be
        "injected" we'll need a different comparator:

        >>> _names = lambda sig1, sig2: set(sig1.parameters) == set(sig2.parameters)
        >>> same_set_of_names = partial(
        ...     compare_signatures,
        ...     signature_comparator=(
        ...         lambda sig1, sig2: set(sig1.parameters) == set(sig2.parameters)
        ...     )
        ... )
        >>> ch_fnode2 = partial(ch_func_node_func, func_comparator=same_set_of_names)
        >>> d = dag.ch_funcs(ch_fnode2, g=lambda z=2, y=1: y / z);

        """
        return ch_funcs(
            self, func_mapping=func_mapping, ch_func_node_func=ch_func_node_func
        )

        # _validate_func_mapping(func_mapping, self)
        #
        # # def validate(func_mapping, func_nodes):
        #
        # # def ch_func(dag, key, func):
        # #     return DAG(
        # #         replace_item_in_iterable(
        # #             dag.func_nodes,
        # #             condition=lambda fn: dag._func_node_for.get(key, None) is not None,
        # #             replacement=lambda fn: ch_func_node_func(fn, func=func),
        # #         )
        # #     )
        #
        # # TODO: Change to use self._func_node_for
        # def ch_func(dag, key, func):
        #     return DAG(
        #         replace_item_in_iterable(
        #             dag.func_nodes,
        #             condition=lambda fn: fn.name == key or fn.out == key,
        #             replacement=lambda fn: _ch_func_node_func(fn, func=func),
        #         )
        #     )
        #
        # new_dag = self
        # for key, func in func_mapping.items():
        #     new_dag = ch_func(new_dag, key, func)
        # return new_dag

    # ------------ utils --------------------------------------------------------------

    @property
    def params_for_src(self):
        """The ``{src_name: list_of_params_using_that_src,...}`` dictionary.
        That is, a ``dict`` having lists of all ``Parameter`` objs that are used by a
        ``node.bind`` source (value of ``node.bind``) for each such source in the graph

        For each ``func_node``, ``func_node.bind`` gives us the
        ``{param: varnode_src_name}`` specification that tells us where (key of scope)
        to source the arguments of the ``func_node.func`` for each ``param`` of that
        function.

        What ``params_for_src`` is, is the corresponding inverse map.
        The ``{varnode_src_name: list_of_params}`` gathered by scanning each
        ``func_node`` of the DAG.
        """
        d = defaultdict(list)
        for node in self.func_nodes:
            for arg_name, src_name in node.bind.items():
                d[src_name].append(node.sig.parameters[arg_name])
        return dict(d)

    def src_name_params(self, src_names: Iterable[str] | None = None):
        """Generate Parameter instances that are needed to compute ``src_names``"""
        # see params_for_src property to see what d is
        d = self.params_for_src
        if src_names is None:  # if no src_names given, use the names of all var_nodes
            src_names = set(d)

        # For every src_name of the DAG that is in ``src_names``...
        for src_name in filter(src_names.__contains__, d):
            params = d[src_name]  # consider all the params that use it
            # make version of these params that have the same name (namely src_name)
            params_with_name_changed_to_src_name = [
                p.replace(name=src_name) for p in params
            ]
            if len(params_with_name_changed_to_src_name) == 1:
                # if there's only one param, yield it (there can be no conflict)
                yield params_with_name_changed_to_src_name[0]
            else:  # if there's more than one param, merge them
                # How to resolve conflicts (different defaults, annotations or kinds)
                # is determined by what ``parameter_merge`` specified, which is,
                # by default, strict (everything needs to be the same, or
                # ``parameter_merge`` with raise an error.)
                yield self.parameter_merge(*params_with_name_changed_to_src_name)

    # TODO: Find more representative (and possibly shorter) doctest:
    @property
    def graph_ids(self):
        """The dict representing the ``{from_node: to_nodes}`` graph.
        Like ``.graph``, but with node ids (names).

        >>> from meshed.dag import DAG
        >>> def add(a, b=1): return a + b
        >>> def mult(x, y=3): return x * y
        >>> def exp(mult, a): return mult ** a
        >>> assert DAG([add, mult, exp]).graph_ids == {
        ...     'a': ['add_', 'exp_'],
        ...     'b': ['add_'],
        ...     'add_': ['add'],
        ...     'x': ['mult_'],
        ...     'y': ['mult_'],
        ...     'mult_': ['mult'],
        ...     'mult': ['exp_'],
        ...     'exp_': ['exp']
        ... }

        """
        return {
            _name_attr_or_x(k): list(map(_name_attr_or_x, v))
            for k, v in self.graph.items()
        }

    def _prepare_other_for_addition(self, other):
        if isinstance(other, DAG):
            other = list(other.func_nodes)
        elif isinstance(other, int) and other == 0:
            # Note: This is so that we can use sum(dags) to get a union of dags without
            # having to specify the initial DAG() value of sum (which is 0 by default).
            other = DAG()
        else:
            other = list(DAG(other).func_nodes)

        return other

    def __radd__(self, other):
        """A union of DAGs. See ``__add__`` for more details.

        >>> dag = sum([DAG(list), DAG(tuple)])
        >>> print(dag.synopsis_string(bind_info='hybrid'))
        iterable -> list_ -> list
        iterable -> tuple_ -> tuple
        >>> dag([1,2,3])
        ([1, 2, 3], (1, 2, 3))

        """
        # We could have just returned self + other to be commutative, but perhaps
        # we would like to control some orders of things via the order of addition
        # (thinkg list addition versus set addition for example), so instead we write
        # the explicit code:
        return DAG(self._prepare_other_for_addition(other) + list(self.func_nodes))

    def __add__(self, other):
        """A union of DAGs.

        :param other: Another DAG or a valid object to make one with ``DAG(other)``.

        >>> dag = DAG(list) + DAG(tuple)
        >>> print(dag.synopsis_string(bind_info='hybrid'))
        iterable -> list_ -> list
        iterable -> tuple_ -> tuple
        >>> dag([1,2,3])
        ([1, 2, 3], (1, 2, 3))
        """
        return DAG(list(self.func_nodes) + self._prepare_other_for_addition(other))

    def copy(self, renamer=numbered_suffix_renamer):
        return DAG(ch_names(self.func_nodes, renamer=renamer))

    def add_edge(self, from_node, to_node, to_param=None):
        """Add an e

        :param from_node:
        :param to_node:
        :param to_param:
        :return: A new DAG with the edge added

        >>> def f(a, b): return a + b
        >>> def g(c, d=1): return c * d
        >>> def h(x, y=1): return x ** y
        >>>
        >>> three_funcs = DAG([f, g, h])
        >>> assert (
        ...     three_funcs(x=1, c=2, a=3, b=4)
        ...     == (7, 2, 1)
        ...     == (f(a=3, b=4), g(c=2), h(x=1))
        ...     == (3 + 4, 2*1, 1** 1)
        ... )
        >>> print(three_funcs.synopsis_string())
        a,b -> f_ -> f
        c,d -> g_ -> g
        x,y -> h_ -> h
        >>> hg = three_funcs.add_edge('h', 'g')
        >>> assert (
        ...     hg(a=3, b=4, x=1)
        ...     == (7, 1)
        ...     == (f(a=3, b=4), g(c=h(x=1)))
        ...     == (3 + 4, 1 * (1 ** 1))
        ... )
        >>> print(hg.synopsis_string())
        a,b -> f_ -> f
        x,y -> h_ -> h
        h,d -> g_ -> g
        >>>
        >>> fhg = three_funcs.add_edge('h', 'g').add_edge('f', 'h')
        >>> assert (
        ...     fhg(a=3, b=4)
        ...     == 7
        ...     == g(h(f(3, 4)))
        ...     == ((3 + 4) * 1) ** 1
        ... )
        >>> print(fhg.synopsis_string())
        a,b -> f_ -> f
        f,y -> h_ -> h
        h,d -> g_ -> g

        The from and to nodes can be expressed by the ``FuncNode`` ``name`` (identifier)
        or ``out``, or even the function itself if it's used only once in the ``DAG``.

        >>> fhg = three_funcs.add_edge(h, 'g').add_edge('f_', 'h')
        >>> assert fhg(a=3, b=4) == 7

        By default, the edge will be added from ``from_node.out`` to the first
        parameter of the function of ``to_node``.
        But if you want otherwise, you can specify the parameter the edge should be
        connected to.
        For example, see below how we connect the outputs of ``g`` and ``h`` to the
        parameters ``a`` and ``b`` of ``f`` respectively:

        >>> f_of_g_and_h = (
        ...     DAG([f, g, h])
        ...     .add_edge(g, f, to_param='a')
        ...     .add_edge(h, f, 'b')
        ... )
        >>> assert (
        ...     f_of_g_and_h(x=2, c=3, y=2, d=2)
        ...     == 10
        ...     == f(g(c=3, d=2), h(x=2, y=2))
        ...     == 3 * 2 + 2 ** 2
        ... )
        >>>
        >>> print(f_of_g_and_h.synopsis_string())
        c,d -> g_ -> g
        x,y -> h_ -> h
        g,h -> f_ -> f

        See Also ``DAG.add_edges`` to add multiple edges at once

        """
        # resolve from_node and to_node into FuncNodes
        from_node, to_node = map(self.find_func_node, (from_node, to_node))
        if to_node is None and callable(to_node):
            to_node = FuncNode(
                to_node
            )  # TODO: Automatically avoid clashing with dag identifiers (?)

        # if to_param is None, take the first parameter of to_node as the one
        if to_param is None:
            if not to_node.bind:
                raise InvalidFunctionParameters(
                    "You can't add an edge TO a FuncNode whose function has no "
                    "parameters. "
                    f"You attempted to add an edge between {from_node=} and {to_node=}."
                )
            else:
                # first param of .func (i.e. first key of .bind)
                to_param = next(iter(to_node.bind))

        existing_bind = to_node.bind[to_param]
        if any(existing_bind == fn.out for fn in self.func_nodes):
            raise ValueError(
                f"The {to_node} node is already sourcing '{to_param}' from '"
                f"{existing_bind}'."
                "Delete that edge to be able before you add a new one"
            )

        new_to_node_dict = to_node.to_dict()
        new_bind = new_to_node_dict["bind"].copy()
        new_bind[to_param] = from_node.out  # this is the actual edge creation
        new_to_node = FuncNode.from_dict(dict(new_to_node_dict, bind=new_bind))
        return DAG(
            replace_item_in_iterable(
                self.func_nodes,
                condition=lambda x: x == to_node,
                replacement=lambda x: new_to_node,
            )
        )

    # TODO: There are optimization and pre-validation opportunities here!
    def add_edges(self, edges):
        """Adds multiple edges by applying ``DAG.add_edge`` multiple times.

        :param edges: An iterable of ``(from_node, to_node)`` pairs or
            ``(from_node, to_node, param)`` triples.
        :return: A new dag with the said edges added.

        >>> def f(a, b): return a + b
        >>> def g(c, d=1): return c * d
        >>> def h(x, y=1): return x ** y
        >>> fhg = DAG([f, g, h]).add_edges([(h, 'g'), ('f_', 'h')])
        >>> assert fhg(a=3, b=4) == 7
        """
        dag = self
        for edge in edges:
            dag = dag.add_edge(*edge)
        return dag

    def debugger(self, feedback: Callable = dflt_debugger_feedback):
        r"""
        Utility to debug DAGs by computing each step sequentially, with feedback.

        :param feedback: A callable that defines what feedback is given, usually used to
            print/log some information and output some information for every step.
            Must be a function with signature ``(func_node, scope, output, step)`` or
            a subset thereof.
        :return:

        >>> from inspect import signature
        >>>
        >>> def f(a, b):
        ...     return a + b
        ...
        >>> def g(c, d=4):
        ...     return c * d
        ...
        >>> def h(f, g):
        ...     return g - f
        ...
        >>> dag2 = DAG([f, g, h], name='arithmetic')
        >>> dag2
        DAG(func_nodes=[FuncNode(a,b -> f_ -> f), FuncNode(c,d -> g_ -> g), FuncNode(f,g -> h_ -> h)], name='arithmetic')
        >>> str(signature(dag2))
        '(a, b, c, d=4)'
        >>> dag2(1,2,3)
        9
        >>>
        >>> debugger = dag2.debugger()
        >>> str(signature(debugger))
        '(a, b, c, d=4)'
        >>> d = debugger(1,2,3)
        >>> next(d)  # doctest: +NORMALIZE_WHITESPACE
        0 --------------------------------------------------------------
            func_node=FuncNode(a,b -> f_ -> f)
            scope={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'f': 3}
        3
        >>> next(d)  # doctest: +NORMALIZE_WHITESPACE
        1 --------------------------------------------------------------
            func_node=FuncNode(c,d -> g_ -> g)
            scope={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'f': 3, 'g': 12}
        12

        ... and so on. You can also choose to run every step all at once, collecting
        the ``feedback`` outputs of each step in a list, like this:

        >>> feedback_outputs = list(debugger(1,2,3))  # doctest: +NORMALIZE_WHITESPACE
        0 --------------------------------------------------------------
            func_node=FuncNode(a,b -> f_ -> f)
            scope={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'f': 3}
        1 --------------------------------------------------------------
            func_node=FuncNode(c,d -> g_ -> g)
            scope={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'f': 3, 'g': 12}
        2 --------------------------------------------------------------
            func_node=FuncNode(f,g -> h_ -> h)
            scope={'a': 1, 'b': 2, 'c': 3, 'd': 4, 'f': 3, 'g': 12, 'h': 9}

        """

        # TODO: Add feedback callable validation
        @Sig(self)
        def launch_debugger(*args, **kwargs):
            scope = self._get_kwargs(*args, **kwargs)
            for step, func_node in enumerate(self.func_nodes):
                output = func_node.call_on_scope(scope)
                kwargs = dict(
                    func_node=func_node, scope=scope, output=output, step=step
                )
                yield call_forgivingly(feedback, **kwargs)

        return launch_debugger

    # ------------ display -------------------------------------------------------------

    def to_code(self):
        return dag_to_code(self)

    def synopsis_string(self, bind_info: BindInfo = "var_nodes"):
        return "\n".join(
            func_node.synopsis_string(bind_info) for func_node in self.func_nodes
        )

    # TODO: Give more control (merge with lined)
    def dot_digraph_body(
        self,
        start_lines=(),
        *,
        end_lines=(),
        vnode_shape: str = dflt_configs["vnode_shape"],
        fnode_shape: str = dflt_configs["fnode_shape"],
        func_display: bool = dflt_configs["func_display"],
    ):
        """Make lines for dot (graphviz) specification of DAG

        >>> def add(a, b=1): return a + b
        >>> def mult(x, y=3): return x * y
        >>> def exp(mult, a): return mult ** a
        >>> func_nodes = [
        ...     FuncNode(add, out='x'), FuncNode(mult, name='the_product'), FuncNode(exp)
        ... ]

        #
        # >>> assert list(DAG(func_nodes).dot_digraph_body()) == [
        # ]
        """
        if isinstance(start_lines, str):
            start_lines = start_lines.split()  # TODO: really? split on space?
        if isinstance(end_lines, str):
            end_lines = end_lines.split()
        kwargs = dict(
            vnode_shape=vnode_shape, fnode_shape=fnode_shape, func_display=func_display
        )
        yield from dot_lines_of_objs(
            self.func_nodes, start_lines=start_lines, end_lines=end_lines, **kwargs
        )

    @wraps(dot_digraph_body)
    def dot_digraph_ascii(self, *args, **kwargs):
        """Get an ascii art string that represents the pipeline"""
        from meshed.util import dot_to_ascii

        return dot_to_ascii("\n".join(self.dot_digraph_body(*args, **kwargs)))

    @wraps(dot_digraph_body)
    def dot_digraph(self, *args, **kwargs):
        try:
            import graphviz
        except (ModuleNotFoundError, ImportError) as e:
            raise ModuleNotFoundError(
                f"{e}\nYou may not have graphviz installed. "
                f"See https://pypi.org/project/graphviz/."
            )
        # Note: Since graphviz 0.18, need to have a newline in body lines!
        body = list(map(add_new_line_if_none, self.dot_digraph_body(*args, **kwargs)))
        return graphviz.Digraph(body=body)

    # NOTE: "sig = property(__signature__)" is not working. So, doing the following instead.
    @property
    def sig(self):
        return self.__signature__

    @sig.setter
    def sig(self, value):
        self.__signature__ = value

    def find_funcs(self, filt: Callable[[FuncNode], bool] = None) -> Iterable[Callable]:
        return (func_node.func for func_node in filter(filt, self.func_nodes))


def call_func(func, kwargs):
    kwargs = {k.__name__: v for k, v in kwargs.items()}
    return Sig(func).source_kwargs(kwargs)


def print_dag_string(dag: DAG, bind_info: BindInfo = "hybrid"):
    print(dag.synopsis_string(bind_info=bind_info))


# --------------------------------------------------------------------------------------
# dag tools

# from typing import Iterable, Union
# from i2 import Sig
from meshed.util import extract_dict
from meshed.base import func_nodes_to_code


def dag_to_code(dag):
    """
    Convert a DAG to code.

    >>> from meshed import code_to_dag
    >>> @code_to_dag
    ... def dag():
    ...     a = func1(x, y)
    ...     b = func2(a, z)
    ...     c = func3(a, w=b)
    >>>

    Original DAG:

    >>> print(dag.synopsis_string())  # doctest: +NORMALIZE_WHITESPACE
    x,y -> func1 -> a
    a,z -> func2 -> b
    a,b -> func3 -> c
    <BLANKLINE>

    Generated code using dag_to_code function:

    >>> code = dag_to_code(dag)
    >>> print(code)  # doctest: +NORMALIZE_WHITESPACE
    def dag():
        a = func1(x, y)
        b = func2(a, z)
        c = func3(a, w=b)
    <BLANKLINE>

    Test round-trip conversion:

    >>> dag2 = code_to_dag(code)
    >>> print(dag2.synopsis_string())  # doctest: +NORMALIZE_WHITESPACE
    x,y -> func1 -> a
    a,z -> func2 -> b
    a,b -> func3 -> c
    <BLANKLINE>
    >>> # Verify they're equivalent:
    >>> dag.synopsis_string() == dag2.synopsis_string()
    True


    """
    return func_nodes_to_code(dag.func_nodes, dag.name)


def parametrized_dag_factory(dag: DAG, param_var_nodes: str | Iterable[str]):
    """
    Constructs a factory for sub-DAGs derived from the input DAG, with values of
    specific 'parameter' variable nodes precomputed and fixed. These precomputed nodes,
    and their ancestor nodes (unless required elsewhere), are omitted from the sub-DAG.

    The factory function produced by this operation requires arguments corresponding to
    the ancestor nodes of the parameter variable nodes. These arguments are used to
    compute the values of the parameter nodes.

    This function reflects the typical structure of a class in object-oriented
    programming, where initialization arguments are used to set certain fixed values
    (attributes), which are then leveraged in subsequent methods.

    >>> import i2
    >>> from meshed import code_to_dag
    >>> @code_to_dag
    ... def testdag():
    ...     a = criss(aa, aaa)
    ...     b = cross(aa, bb)
    ...     c = apple(a, b)
    ...     d = sauce(a, b)
    ...     e = applesauce(c, d)
    >>>
    >>> dag_factory = parametrized_dag_factory(testdag, 'a')
    >>> print(f"{i2.Sig(dag_factory)}")
    (aa, aaa)
    >>> d = dag_factory(aa=1, aaa=2)
    >>> print(f"{i2.Sig(d)}")
    (b)
    >>> d(b='bananna')
    'applesauce(c=apple(a=criss(aa=1, aaa=2), b=bananna), d=sauce(a=criss(aa=1, aaa=2), b=bananna))'

    """

    if isinstance(param_var_nodes, str):
        param_var_nodes = param_var_nodes.split()
    # The dag is split into two parts:
    #   Part whose role it is to compute the param_var_nodes from root nodes
    param_dag = dag[:param_var_nodes]
    #   Part that computes the rest based on these (and remaining root nodes)
    computation_dag = dag[param_var_nodes:]
    # Get the intersection of the two parts on the var nodes
    common_var_nodes = set(param_dag.var_nodes) & set(computation_dag.var_nodes)

    @Sig(param_dag)
    def dag_factory(*parametrization_args, **parametrization_kwargs):
        # use the param_dag to compute the values of the parameter var nodes
        # (and what ever else happens to be in the leaves, but we'll remove that later)
        _ = param_dag(*parametrization_args, **parametrization_kwargs)
        # Get the values for all nodes that are common to param_dag and computation_dag
        # (There may be more than just param_var_nodes!)
        common_var_node_values = extract_dict(param_dag.last_scope, common_var_nodes)
        # By fixing those values, you now have a the computation_dag you want
        # Note: Also, remove the bound arguments
        # (i.e. the arguments that were used to compute the values)
        # so that the user doesn't change those and get inconsistencies!
        d = computation_dag.partial(
            **common_var_node_values, _remove_bound_arguments=True
        )
        # Remember the var nodes that parametrized the dag
        # TODO: Is this a good idea? Meant for debugging really.
        d._common_var_node_values = common_var_node_values
        return d

    return dag_factory


# --------------------------------------------------------------------------------------
# reordering funcnodes

from meshed.util import uncurry, pairs

mk_mock_funcnode_from_tuple = uncurry(mk_mock_funcnode)


def funcnodes_from_pairs(pairs):
    return list(map(mk_mock_funcnode_from_tuple, pairs))


def reorder_on_constraints(funcnodes, outs):
    extra_nodes = funcnodes_from_pairs(pairs(outs))
    funcnodes += extra_nodes
    graph = _func_nodes_to_graph_dict(funcnodes)
    nodes = topological_sort(graph)
    print("after ordering:", nodes)
    ordered_nodes = [node for node in nodes if node not in extra_nodes]
    func_nodes, var_nodes = _separate_func_nodes_and_var_nodes(ordered_nodes)

    return func_nodes, var_nodes


def attribute_vals(objs: Iterable, attrs: Iterable[str], egress=None):
    """Extract attributes from an iterable of objects
    >>> list(attribute_vals([print, map], attrs=['__name__', '__module__']))
    [('print', 'builtins'), ('map', 'builtins')]
    """
    if isinstance(attrs, str):
        attrs = attrs.split()
    val_tuples = map(attrgetter(*attrs), objs)
    if egress:
        return egress(val_tuples)
    else:
        return val_tuples


names_and_outs = partial(attribute_vals, attrs=("name", "out"), egress=chain)

DagAble = Union[DAG, Iterable[FuncNodeAble]]


# TODO: Extract hardcoded ".name or .out" condition so indexing/condition can be
#  controlled by user.
def _validate_func_mapping(func_mapping: FuncMapping, func_nodes: DagAble):
    """Validates a ``FuncMapping`` against an iterable of ``FuncNodes``.

    That is, it assures that:

    - The keys of ``func_mapping`` are all ``FuncNode`` identifiers (i.e. appear as a
    ``.name`` or ``.out`` of one of the ``func_nodes``.

    - The values of ``func_mapping`` are all callable.

    >>> def f(a, b):
    ...     return a + b
    >>> def g(a_plus_b, x):
    ...     return a_plus_b * x
    ...
    >>> func_nodes = [
    ...     FuncNode(func=f, out='a_plus_b'), FuncNode(func=g, bind={'x': 'b'})
    ... ]
    >>> _validate_func_mapping(
    ...     dict(f=lambda a, b: a - b, g=lambda a_plus_b, x: x), func_nodes
    ... )

    You can use the ``.name`` or ``.out`` to index the func_node:

    >>> _validate_func_mapping(dict(f=lambda a, b: a - b), func_nodes)
    >>> _validate_func_mapping(dict(a_plus_b=lambda a, b: a - b), func_nodes)

    If you mention a key that doesn't correspond to one of the elements of
    ``func_nodes``, you'll be told off.

    >>> _validate_func_mapping(dict(not_a_key=lambda a, b: a - b), func_nodes)
    Traceback (most recent call last):
      ...
    KeyError: "These identifiers weren't found in func_nodes: not_a_key"

    If you mention a value that is not callable, you'll also be told off:

    >>> _validate_func_mapping(dict(f='hello world'), func_nodes)
    Traceback (most recent call last):
      ...
    TypeError: These values of func_src weren't callable: hello world

    """
    allowed_identifiers = set(
        chain.from_iterable(names_and_outs(DAG(func_nodes).func_nodes))
    )
    if not_allowed := (func_mapping.keys() - allowed_identifiers):
        raise KeyError(
            f"These identifiers weren't found in func_nodes: {', '.join(not_allowed)}"
        )
    if not_callable := set(filter(lambda x: not callable(x), func_mapping.values())):
        raise TypeError(
            f"These values of func_src weren't callable: {', '.join(not_callable)}"
        )


FuncMappingValidator = Callable[[FuncMapping, DagAble], None]


# TODO: Redesign. Is terrible both in interface and code.
# TODO: Merge with DAG, or with Mesh (when it exists)
# TODO: Make it work with any FuncNode Iterable
# TODO: extract egress functionality to decorator?
@double_up_as_factory
def ch_funcs(
    func_nodes: DagAble = None,
    *,
    func_mapping: FuncMapping = (),
    validate_func_mapping: FuncMappingValidator | None = _validate_func_mapping,
    # TODO: Design. Don't like the fact that ch_func_node_func needs a slot for
    #  func_comparator, which is then given later. Perhaps only ch_func_node_func should
    #  should be given (and it contains the func_comparator)
    ch_func_node_func: Callable[
        [FuncNode, Callable, CallableComparator], FuncNode
    ] = ch_func_node_func,
    # func_comparator: CallableComparator = compare_signatures,
):
    """Function (and decorator) to change the functions of func_nodes according to
    the specification of a func_mapping whose keys are ``.name`` or ``.out`` values
    of the nodes of ``func_nodes`` and the values are the callable we want to replace
    them by.

    A constrained version of ``ch_funcs`` is used as a method of ``DAG``.
    The present function is given to provide more control.

    """
    func_mapping = dict(func_mapping)
    if validate_func_mapping:
        validate_func_mapping(func_mapping, func_nodes)

    # def validate(func_mapping, func_nodes):

    # def ch_func(dag, key, func):
    #     return DAG(
    #         replace_item_in_iterable(
    #             dag.func_nodes,
    #             condition=lambda fn: dag._func_node_for.get(key, None) is not None,
    #             replacement=lambda fn: ch_func_node_func(fn, func=func),
    #         )
    #     )

    # TODO: Optimize (for example, use self._func_node_for)
    def ch_func(dag, key, func):
        condition = lambda fn: fn.name == key or fn.out == key  # TODO: interface ctrl?
        replacement = lambda fn: ch_func_node_func(
            fn,
            func,
        )
        return DAG(
            replace_item_in_iterable(
                dag.func_nodes,
                condition=condition,
                replacement=replacement,
            )
        )

    new_dag = DAG(func_nodes)
    for key, func in func_mapping.items():
        new_dag = ch_func(new_dag, key, func)
    return new_dag

    # def transformed_func_nodes():
    #     for fn in func_nodes:
    #         if (
    #             new_func := func_mapping.get(fn.out, func_mapping.get(fn.name, None))
    #         ) is not None:
    #             new_fn_kwargs = dict(fn.to_dict(), func=new_func)
    #             yield FuncNode.from_dict(new_fn_kwargs)
    #         else:
    #             yield fn

    # # If func_nodes are input as a DAG (which is an iterable of FuncNodes!),
    # # make sure to return a DAG as well -- if not, return a list of FuncNodes
    # if isinstance(func_nodes, DAG):
    #     return DAG(transformed_func_nodes())
    # else:
    #     return list(transformed_func_nodes())


change_funcs = ch_funcs  # back-compatibility


# TODO: Include as method of DAG?
# TODO: extract egress functionality to decorator
@double_up_as_factory
def ch_names(func_nodes: DagAble = None, *, renamer: Renamer = numbered_suffix_renamer):
    """Renames variables and functions of a ``DAG`` or iterable of ``FuncNodes``.

    :param func_nodes: A ``DAG`` of iterable of ``FuncNodes``
    :param renamer: A function taking an old name and returning the new one, or:
        - A dictionary ``{old_name: new_name, ...}`` mapping old names to new ones
        - A string, which will be appended to all identifiers of the ``func_nodes``
    :return: func_nodes with some or all identifiers changed. If the input ``func_nodes``
    is an iterable of ``FuncNodes``, a list of func_nodes will be returned, and if the
    input ``func_nodes`` is a ``DAG`` instance, a ``DAG`` will be returned.

    >>> from meshed.makers import code_to_dag
    >>> from meshed.dag import print_dag_string
    >>>
    >>> @code_to_dag
    ... def dag():
    ...     b = f(a)
    ...     c = g(x=a)
    ...     d = h(b, y=c)


    This is what the dag looks like:

    >>> print_dag_string(dag)
    a -> f -> b
    x=a -> g -> c
    b,y=c -> h -> d

    Now, if rename the vars of the ``dag`` without further specifying how, all of our
    nodes (names) will be suffixed with a ``_1``

    >>> new_dag = ch_names(dag)
    >>> print_dag_string(new_dag)
    a=a_1 -> f_1 -> b_1
    x=a_1 -> g_1 -> c_1
    b=b_1,y=c_1 -> h_1 -> d_1

    If any nodes are already suffixed by ``_`` followed by a number, the default
    renamer (``numbered_suffix_renamer``) will increment that number:

    >>> another_new_data = ch_names(new_dag)
    >>> print_dag_string(another_new_data)
    a=a_2 -> f_2 -> b_2
    x=a_2 -> g_2 -> c_2
    b=b_2,y=c_2 -> h_2 -> d_2

    If we specify a string for the ``renamer`` argument, it will be used to suffix all
    the nodes.

    >>> print_dag_string(ch_names(dag, renamer='_copy'))
    a=a_copy -> f_copy -> b_copy
    x=a_copy -> g_copy -> c_copy
    b=b_copy,y=c_copy -> h_copy -> d_copy

    Finally, for full functionality on renaming, you can use a function

    >>> print_dag_string(ch_names(dag, renamer=lambda x: f"{x.upper()}"))
    a=A -> F -> B
    x=A -> G -> C
    b=B,y=C -> H -> D

    In all the above our input was a ``DAG`` so we got a ``DAG`` back, but if we enter
    an iterable of ``FuncNode`` instances, we'll get a list of the same back.
    Also, know that if your function returns ``None`` for a given identifier, it will
    have the effect of not changing that identifier.

    >>> ch_names(dag.func_nodes, renamer=lambda x: x.upper() if x in 'abc' else None)
    [FuncNode(a=A -> f -> B), FuncNode(x=A -> g -> C), FuncNode(b=B,y=C -> h -> d)]

    If you want to rename the nodes with an explicit mapping, you can do so by
    specifying this mapping as your renamer

    >>> substitutions = {'a': 'alpha', 'b': 'bravo'}
    >>> print_dag_string(ch_names(dag, renamer=substitutions))
    a=alpha -> f -> bravo
    x=alpha -> g -> c
    b=bravo,y=c -> h -> d

    """
    if isinstance(func_nodes, DAG):
        egress = DAG
    else:
        egress = list
    renamer = renamer or numbered_suffix_renamer
    if isinstance(renamer, str):
        suffix = renamer
        renamer = lambda name: f"{name}{suffix}"
    elif isinstance(renamer, Mapping):
        old_to_new_map = dict(renamer)
        renamer = old_to_new_map.get
    assert callable(renamer), f"Could not be resolved into a callable: {renamer}"
    ktrans = partial(_rename_node, renamer=renamer)
    func_node_trans = partial(func_node_transformer, kwargs_transformers=ktrans)
    return egress(map(func_node_trans, func_nodes))


def _rename_node(fn_kwargs, renamer: Renamer = numbered_suffix_renamer):
    fn_kwargs = fn_kwargs.copy()
    # decorate renamer so if the original returns None the decorated will return input
    renamer = _if_none_return_input(renamer)
    fn_kwargs["name"] = renamer(fn_kwargs["name"])
    fn_kwargs["out"] = renamer(fn_kwargs["out"])
    fn_kwargs["bind"] = {
        param: renamer(var_id) for param, var_id in fn_kwargs["bind"].items()
    }
    return fn_kwargs


rename_nodes = ch_names  # back-compatibility
```

## examples/__init__.py

```python
"""Examples of using meshed."""

from meshed.examples.online_marketing import funcs as online_marketing_funcs
from meshed.examples.vaccine_vs_no_vaccine import funcs as vaccine_vs_no_vaccine_funcs
from meshed.examples.price_elasticity import funcs as price_elasticity_funcs


funcs = {
    "online_marketing": online_marketing_funcs,
    "vaccine_vs_no_vaccine": vaccine_vs_no_vaccine_funcs,
    "price_elasticity": price_elasticity_funcs,
}
```

## examples/online_marketing.py

```python
"""Online marketing funnel: impressions and clicks to sales and profit.

                           ┌──────────────────────┐
                           │ click_per_impression │
                           └──────────────────────┘
                             │
                             ▼
   ┌─────────────────┐     ┌──────────────────────┐
   │   impressions   │ ──▶ │        clicks        │
   └─────────────────┘     └──────────────────────┘
┌────┘                       │
│                            ▼
│  ┌─────────────────┐     ┌──────────────────────┐
│  │ sales_per_click │ ──▶ │        sales         │
│  └─────────────────┘     └──────────────────────┘
│                            │
│                            ▼
│                          ┌──────────────────────┐     ┌──────────────────┐
│                          │       revenue        │ ◀── │ revenue_per_sale │
│                          └──────────────────────┘     └──────────────────┘
│                            │
│                            ▼
│                          ┌──────────────────────┐
│                          │        profit        │ ◀┐
│                          └──────────────────────┘  │
│                          ┌──────────────────────┐  │
│                          │ cost_per_impression  │  │
│                          └──────────────────────┘  │
│                            │                       │
│                            ▼                       │
│                          ┌──────────────────────┐  │
└────────────────────────▶ │         cost         │ ─┘
                           └──────────────────────┘

"""


def cost(impressions, cost_per_impression):
    return impressions * cost_per_impression


def clicks(impressions, click_per_impression):
    return impressions * click_per_impression


def sales(clicks, sales_per_click):
    return clicks * sales_per_click


def revenue(sales, revenue_per_sale):
    return sales * revenue_per_sale


def profit(revenue, cost):
    return revenue - cost


funcs = (cost, clicks, sales, revenue, profit)
```

## examples/price_elasticity.py

```python
"""price elasticity relates price to revenue, expense, and profit
                   ┌─────────┐
                   │  base   │
                   └─────────┘
                     │
                     │
                     ▼
┌────────────┐     ┌─────────────────────────┐
│ elasticity │ ──▶ │          sold           │ ─┐
└────────────┘     └─────────────────────────┘  │
                     │               ▲          │
                     │               │          │
                     ▼               │          │
┌────────────┐     ┌─────────┐     ┌─────────┐  │
│    cost    │ ──▶ │ expense │     │  price  │  │
└────────────┘     └─────────┘     └─────────┘  │
                     │               │          │
                     │               │          │
                     ▼               ▼          │
                   ┌─────────┐     ┌─────────┐  │
                   │ profit  │ ◀── │ revenue │ ◀┘
                   └─────────┘     └─────────┘

"""


def profit(revenue, expense):
    return revenue - expense


def revenue(price, sold):
    return price * sold


def expense(cost, sold):
    return cost * sold


def sold(price, elasticity, base=1e6):
    return base * price ** (1 - elasticity)


funcs = (profit, revenue, expense, sold)
```

## examples/vaccine_vs_no_vaccine.py

```python
"""Simple model relating vaccination to death toll, involving exposure and infection rate

                        ┌──────────────────────┐
                        │   death_vax_factor   │
                        └──────────────────────┘
                          │
                          ▼
┌─────────────────┐     ┌──────────────────────────────────┐
│ die_if_infected │ ──▶ │               die                │
└─────────────────┘     └──────────────────────────────────┘
                          │                       ▲    ▲
                          ▼                       │    │
┌─────────────────┐     ┌──────────────────────┐  │  ┌─────┐
│   population    │ ──▶ │      death_toll      │  │  │ vax │
└─────────────────┘     └──────────────────────┘  │  └─────┘
                                                  │    │
                        ┌──────────────────────┐  │    │
                        │ infection_vax_factor │  │    │
                        └──────────────────────┘  │    │
                          │                       │    │
                          ▼                       │    │
                        ┌──────────────────────┐  │    │
                     ┌▶ │       infected       │ ─┘    │
                     │  └──────────────────────┘       │
                     │    ▲                            │
                     │    └────────────────────────────┘
                     │  ┌──────────────────────┐
                     │  │       exposed        │
                     │  └──────────────────────┘
                     │    │
                     │    ▼
                     │  ┌──────────────────────┐
                     └─ │          r           │
                        └──────────────────────┘
                          ▲
                          │
                        ┌──────────────────────┐
                        │   infect_if_expose   │
                        └──────────────────────┘
"""

DFLT_VAX = 0.5


def _factor(vax, vax_factor):
    assert 0 <= vax <= 1, f"vax should be between 0 and 1: Was {vax}"
    return vax * vax_factor + (1 - vax)


def r(exposed: float = 6, infect_if_expose: float = 1 / 5):
    return exposed * infect_if_expose


def infected(r: float = 1.2, vax: float = DFLT_VAX, infection_vax_factor: float = 0.15):
    return r * _factor(vax, infection_vax_factor)


def die(
    infected: float,
    die_if_infected: float = 0.05,
    vax: float = DFLT_VAX,
    death_vax_factor: float = 0.05,
):
    return infected * die_if_infected * _factor(vax, death_vax_factor)


def death_toll(die: float, population: int = 1e6):
    return int(die * population)


funcs = (r, infected, die, death_toll)
```

## ext/__init__.py

```python
"""vendors"""
```

## ext/gk.py

```python
"""
This module is meant to explore a different representation of a computation graph
and a different way of executing it.
It is based on Yahoo's graphkit library. The library hasn't been maintained since 2018,
so vendored and modified here).
One of the main differences is that we got rid of the networkx dependency,
which was used to represent the computation graph.
Instead, this module uses meshed's itools library to represent the computation graph.

# Yahoo's graphkit library is under Apache License 2.0:
# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

NOTE: This module is only meant to an exploratory "extension". It is not planned to be maintained.
"""

# ---------- base --------------------------------------------------------------


class Data:
    """
    This wraps any data that is consumed or produced
    by a Operation. This data should also know how to serialize
    itself appropriately.
    This class an "abstract" class that should be extended by
    any class working with data in the HiC framework.
    """

    def __init__(self, **kwargs):
        pass

    def get_data(self):
        raise NotImplementedError

    def set_data(self, data):
        raise NotImplementedError


from dataclasses import dataclass, field


@dataclass
class Operation:
    """
    This is an abstract class representing a data transformation. To use this,
    please inherit from this class and customize the ``.compute`` method to your
    specific application.

    Names may be given to this layer and its inputs and outputs. This is
    important when connecting layers and data in a Network object, as the
    names are used to construct the graph.
    :param str name: The name the operation (e.g. conv1, conv2, etc..)
    :param list needs: Names of input data objects this layer requires.
    :param list provides: Names of output data objects this provides.
    :param dict params: A dict of key/value pairs representing parameters
                        associated with your operation. These values will be
                        accessible using the ``.params`` attribute of your object.
                        NOTE: It's important that any values stored in this
                        argument must be pickelable.
    """

    name: str = field(default="None")
    needs: list = field(default=None)
    provides: list = field(default=None)
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        This method is a hook for you to override. It gets called after this
        object has been initialized with its ``needs``, ``provides``, ``name``,
        and ``params`` attributes. People often override this method to implement
        custom loading logic required for objects that do not pickle easily, and
        for initialization of c++ dependencies.
        """
        pass

    def __eq__(self, other):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return bool(self.name is not None and self.name == getattr(other, "name", None))

    def __hash__(self):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return hash(self.name)

    def compute(self, inputs):
        """
        This method must be implemented to perform this layer's feed-forward
        computation on a given set of inputs.
        :param list inputs:
            A list of :class:`Data` objects on which to run the layer's
            feed-forward computation.
        :returns list:
            Should return a list of :class:`Data` objects representing
            the results of running the feed-forward computation on
            ``inputs``.
        """

        raise NotImplementedError

    def _compute(self, named_inputs, outputs=None):
        inputs = [named_inputs[d] for d in self.needs]
        results = self.compute(inputs)

        results = zip(self.provides, results)
        if outputs:
            outputs = set(outputs)
            results = filter(lambda x: x[0] in outputs, results)

        return dict(results)

    def __getstate__(self):
        """
        This allows your operation to be pickled.
        Everything needed to instantiate your operation should be defined by the
        following attributes: params, needs, provides, and name
        No other piece of state should leak outside of these 4 variables
        """

        result = {}
        # this check should get deprecated soon. its for downward compatibility
        # with earlier pickled operation objects
        if hasattr(self, "params"):
            result["params"] = self.__dict__["params"]
        result["needs"] = self.__dict__["needs"]
        result["provides"] = self.__dict__["provides"]
        result["name"] = self.__dict__["name"]

        return result

    def __setstate__(self, state):
        """
        load from pickle and instantiate the detector
        """
        for k in iter(state):
            self.__setattr__(k, state[k])
        self.__postinit__()

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        return "{}(name='{}', needs={}, provides={})".format(
            self.__class__.__name__,
            self.name,
            self.needs,
            self.provides,
        )


class NetworkOperation(Operation):
    def __init__(self, **kwargs):
        self.net = kwargs.pop("net")
        Operation.__init__(self, **kwargs)

        # set execution mode to single-threaded sequential by default
        self._execution_method = "sequential"

    def _compute(self, named_inputs, outputs=None):
        return self.net.compute(outputs, named_inputs, method=self._execution_method)

    def __call__(self, *args, **kwargs):
        return self._compute(*args, **kwargs)

    def set_execution_method(self, method):
        """
        Determine how the network will be executed.
        Args:
            method: str
                If "parallel", execute graph operations concurrently
                using a threadpool.
        """
        options = ["parallel", "sequential"]
        assert method in options
        self._execution_method = method

    def plot(self, filename=None, show=False):
        self.net.plot(filename=filename, show=show)

    def __getstate__(self):
        state = Operation.__getstate__(self)
        state["net"] = self.__dict__["net"]
        return state


# ------------ modifiers -------------------------------------------------------

"""
This sub-module contains input/output modifiers that can be applied to
arguments to ``needs`` and ``provides`` to let GraphKit know it should treat
them differently.

Copyright 2016, Yahoo Inc.
Licensed under the terms of the Apache License, Version 2.0. See the LICENSE
file associated with the project for terms.
"""


class optional(str):
    """
    Input values in ``needs`` may be designated as optional using this modifier.
    If this modifier is applied to an input value, that value will be input to
    the ``operation`` if it is available.  The function underlying the
    ``operation`` should have a parameter with the same name as the input value
    in ``needs``, and the input value will be passed as a keyword argument if
    it is available.

    Here is an example of an operation that uses an optional argument::

        from graphkit import operation, compose
        from graphkit.modifiers import optional

        # Function that adds either two or three numbers.
        def myadd(a, b, c=0):
            return a + b + c

        # Designate c as an optional argument.
        graph = compose('mygraph')(
            operator(name='myadd', needs=['a', 'b', optional('c')], provides='sum')(myadd)
        )

        # The graph works with and without 'c' provided as input.
        assert graph({'a': 5, 'b': 2, 'c': 4})['sum'] == 11
        assert graph({'a': 5, 'b': 2})['sum'] == 7

    """

    pass


# ------------ network ------------------------------------------------------

# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

import time
import os

# import networkx as nx
from meshed import itools as gr

from io import StringIO

# uses base.Operation


class DataPlaceholderNode(str):
    """
    A node for the Network graph that describes the name of a Data instance
    produced or required by a layer.
    """

    def __repr__(self):
        return 'DataPlaceholderNode("%s")' % self


class DeleteInstruction(str):
    """
    An instruction for the compiled list of evaluation steps to free or delete
    a Data instance from the Network's cache after it is no longer needed.
    """

    def __repr__(self):
        return 'DeleteInstruction("%s")' % self


class Network:
    """
    This is the main network implementation. The class contains all of the
    code necessary to weave together operations into a directed-acyclic-graph (DAG)
    and pass data through.
    """

    def __init__(self, **kwargs):
        """ """

        # directed graph of layer instances and data-names defining the net.
        self.graph = dict()
        self._debug = kwargs.get("debug", False)

        # this holds the timing information for eache layer
        self.times = {}

        # a compiled list of steps to evaluate layers *in order* and free mem.
        self.steps = []

        # This holds a cache of results for the _find_necessary_steps
        # function, this helps speed up the compute call as well avoid
        # a multithreading issue that is occuring when accessing the
        # graph in networkx
        self._necessary_steps_cache = {}

    def add_op(self, operation):
        """
        Adds the given operation and its data requirements to the network graph
        based on the name of the operation, the names of the operation's needs, and
        the names of the data it provides.

        :param Operation operation: Operation object to add.
        """

        # assert layer and its data requirements are named.
        assert operation.name, "Operation must be named"
        assert operation.needs is not None, "Operation's 'needs' must be named"
        assert operation.provides is not None, "Operation's 'provides' must be named"

        # assert layer is only added once to graph
        assert operation not in gr.nodes(self.graph), "Operation may only be added once"

        # add nodes and edges to graph describing the data needs for this layer
        for n in operation.needs:
            gr.add_edge(self.graph, DataPlaceholderNode(n), operation)

        # add nodes and edges to graph describing what this layer provides
        for p in operation.provides:
            gr.add_edge(self.graph, operation, DataPlaceholderNode(p))

        # clear compiled steps (must recompile after adding new layers)
        self.steps = []

    def list_layers(self):
        assert self.steps, "network must be compiled before listing layers."
        return [(s.name, s) for s in self.steps if isinstance(s, Operation)]

    def show_layers(self):
        """Shows info (name, needs, and provides) about all layers in this network."""
        for name, step in self.list_layers():
            print("layer_name: ", name)
            print("\t", "needs: ", step.needs)
            print("\t", "provides: ", step.provides)
            print("")

    def compile(self):
        """Create a set of steps for evaluating layers
        and freeing memory as necessary"""

        # clear compiled steps
        self.steps = []

        # create an execution order such that each layer's needs are provided.
        ordered_nodes = list(gr.topological_sort(self.graph))

        # add Operations evaluation steps, and instructions to free data.
        for i, node in enumerate(ordered_nodes):

            if isinstance(node, DataPlaceholderNode):
                continue

            elif isinstance(node, Operation):

                # add layer to list of steps
                self.steps.append(node)

                # Add instructions to delete predecessors as possible.  A
                # predecessor may be deleted if it is a data placeholder that
                # is no longer needed by future Operations.
                for predecessor in gr.predecessors(self.graph, node):
                    if self._debug:
                        print("checking if node %s can be deleted" % predecessor)
                    predecessor_still_needed = False
                    for future_node in ordered_nodes[i + 1 :]:
                        if isinstance(future_node, Operation):
                            if predecessor in future_node.needs:
                                predecessor_still_needed = True
                                break
                    if not predecessor_still_needed:
                        if self._debug:
                            print("  adding delete instruction for %s" % predecessor)
                        self.steps.append(DeleteInstruction(predecessor))

            else:
                raise TypeError("Unrecognized network graph node")

    def _find_necessary_steps(self, outputs, inputs):
        """
        Determines what graph steps need to pe run to get to the requested
        outputs from the provided inputs.  Eliminates steps that come before
        (in topological order) any inputs that have been provided.  Also
        eliminates steps that are not on a path from he provided inputs to
        the requested outputs.

        :param list outputs:
            A list of desired output names.  This can also be ``None``, in which
            case the necessary steps are all graph nodes that are reachable
            from one of the provided inputs.

        :param dict inputs:
            A dictionary mapping names to values for all provided inputs.

        :returns:
            Returns a list of all the steps that need to be run for the
            provided inputs and requested outputs.
        """

        # return steps if it has already been computed before for this set of inputs and outputs
        outputs = (
            tuple(sorted(outputs)) if isinstance(outputs, (list, set)) else outputs
        )
        inputs_keys = tuple(sorted(inputs.keys()))
        cache_key = (inputs_keys, outputs)
        if cache_key in self._necessary_steps_cache:
            return self._necessary_steps_cache[cache_key]

        graph = self.graph
        if not outputs:

            # If caller requested all outputs, the necessary nodes are all
            # nodes that are reachable from one of the inputs.  Ignore input
            # names that aren't in the graph.
            necessary_nodes = set()
            for input_name in iter(inputs):
                if gr.has_node(graph, input_name):
                    necessary_nodes |= gr.descendants(graph, input_name)

        else:

            # If the caller requested a subset of outputs, find any nodes that
            # are made unecessary because we were provided with an input that's
            # deeper into the network graph.  Ignore input names that aren't
            # in the graph.
            unnecessary_nodes = set()
            for input_name in iter(inputs):
                if gr.has_node(graph, input_name):
                    unnecessary_nodes |= gr.ancestors(graph, input_name)

            # Find the nodes we need to be able to compute the requested
            # outputs.  Raise an exception if a requested output doesn't
            # exist in the graph.
            necessary_nodes = set()
            for output_name in outputs:
                if not gr.has_node(graph, output_name):
                    raise ValueError(
                        "graphkit graph does not have an output "
                        "node named %s" % output_name
                    )
                necessary_nodes |= gr.ancestors(graph, output_name)

            # Get rid of the unnecessary nodes from the set of necessary ones.
            necessary_nodes -= unnecessary_nodes

        necessary_steps = [step for step in self.steps if step in necessary_nodes]

        # save this result in a precomputed cache for future lookup
        self._necessary_steps_cache[cache_key] = necessary_steps

        # Return an ordered list of the needed steps.
        return necessary_steps

    def compute(self, outputs, named_inputs, method=None):
        """
        Run the graph. Any inputs to the network must be passed in by name.

        :param list output: The names of the data node you'd like to have returned
                            once all necessary computations are complete.
                            If you set this variable to ``None``, all
                            data nodes will be kept and returned at runtime.

        :param dict named_inputs: A dict of key/value pairs where the keys
                                  represent the data nodes you want to populate,
                                  and the values are the concrete values you
                                  want to set for the data node.


        :returns: a dictionary of output data objects, keyed by name.
        """

        # assert that network has been compiled
        assert self.steps, "network must be compiled before calling compute."
        assert (
            isinstance(outputs, (list, tuple)) or outputs is None
        ), "The outputs argument must be a list"

        # choose a method of execution
        if method == "parallel":
            return self._compute_thread_pool_barrier_method(named_inputs, outputs)
        else:
            return self._compute_sequential_method(named_inputs, outputs)

    def _compute_thread_pool_barrier_method(
        self, named_inputs, outputs, thread_pool_size=10
    ):
        """
        This method runs the graph using a parallel pool of thread executors.
        You may achieve lower total latency if your graph is sufficiently
        sub divided into operations using this method.
        """
        from multiprocessing.dummy import Pool

        # if we have not already created a thread_pool, create one
        if not hasattr(self, "_thread_pool"):
            self._thread_pool = Pool(thread_pool_size)
        pool = self._thread_pool

        cache = {}
        cache.update(named_inputs)
        necessary_nodes = self._find_necessary_steps(outputs, named_inputs)

        # this keeps track of all nodes that have already executed
        has_executed = set()

        # with each loop iteration, we determine a set of operations that can be
        # scheduled, then schedule them onto a thread pool, then collect their
        # results onto a memory cache for use upon the next iteration.
        while True:

            # the upnext list contains a list of operations for scheduling
            # in the current round of scheduling
            upnext = []
            for node in necessary_nodes:
                # only delete if all successors for the data node have been executed
                if isinstance(node, DeleteInstruction):
                    if ready_to_delete_data_node(node, has_executed, self.graph):
                        if node in cache:
                            cache.pop(node)

                # continue if this node is anything but an operation node
                if not isinstance(node, Operation):
                    continue

                if (
                    ready_to_schedule_operation(node, has_executed, self.graph)
                    and node not in has_executed
                ):
                    upnext.append(node)

            # stop if no nodes left to schedule, exit out of the loop
            if len(upnext) == 0:
                break

            done_iterator = pool.imap_unordered(
                lambda op: (op, op._compute(cache)), upnext
            )
            for op, result in done_iterator:
                cache.update(result)
                has_executed.add(op)

        if not outputs:
            return cache
        else:
            return {k: cache[k] for k in iter(cache) if k in outputs}

    def _compute_sequential_method(self, named_inputs, outputs):
        """
        This method runs the graph one operation at a time in a single thread
        """
        # start with fresh data cache
        cache = {}

        # add inputs to data cache
        cache.update(named_inputs)

        # Find the subset of steps we need to run to get to the requested
        # outputs from the provided inputs.
        all_steps = self._find_necessary_steps(outputs, named_inputs)

        self.times = {}
        for step in all_steps:

            if isinstance(step, Operation):

                if self._debug:
                    print("-" * 32)
                    print("executing step: %s" % step.name)

                # time execution...
                t0 = time.time()

                # compute layer outputs
                layer_outputs = step._compute(cache)

                # add outputs to cache
                cache.update(layer_outputs)

                # record execution time
                t_complete = round(time.time() - t0, 5)
                self.times[step.name] = t_complete
                if self._debug:
                    print("step completion time: %s" % t_complete)

            # Process DeleteInstructions by deleting the corresponding data
            # if possible.
            elif isinstance(step, DeleteInstruction):

                if outputs and step not in outputs:
                    # Some DeleteInstruction steps may not exist in the cache
                    # if they come from optional() needs that are not privoded
                    # as inputs.  Make sure the step exists before deleting.
                    if step in cache:
                        if self._debug:
                            print("removing data '%s' from cache." % step)
                        cache.pop(step)

            else:
                raise TypeError("Unrecognized instruction.")

        if not outputs:
            # Return the whole cache as output, including input and
            # intermediate data nodes.
            return cache

        else:
            # Filter outputs to just return what's needed.
            # Note: list comprehensions exist in python 2.7+
            return {k: cache[k] for k in iter(cache) if k in outputs}

    def plot(self, filename=None, show=False):
        """
        Plot the graph.

        params:
        :param str filename:
            Write the output to a png, pdf, or graphviz dot file. The extension
            controls the output format.

        :param boolean show:
            If this is set to True, use matplotlib to show the graph diagram
            (Default: False)

        :returns:
            An instance of the pydot graph

        """
        from contextlib import suppress

        with suppress(ModuleNotFoundError, ImportError):
            import pydot
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            assert self.graph is not None

            def get_node_name(a):
                if isinstance(a, DataPlaceholderNode):
                    return a
                return a.name

            g = pydot.Dot(graph_type="digraph")

            # draw nodes
            for nx_node in gr.nodes(self.graph):
                if isinstance(nx_node, DataPlaceholderNode):
                    node = pydot.Node(name=nx_node, shape="rect")
                else:
                    node = pydot.Node(name=nx_node.name, shape="circle")
                g.add_node(node)

            # draw edges
            for src, dst in gr.edges(self.graph):
                src_name = get_node_name(src)
                dst_name = get_node_name(dst)
                edge = pydot.Edge(src=src_name, dst=dst_name)
                g.add_edge(edge)

            # save plot
            if filename:
                basename, ext = os.path.splitext(filename)
                with open(filename, "w") as fh:
                    if ext.lower() == ".png":
                        fh.write(g.create_png())
                    elif ext.lower() == ".dot":
                        fh.write(g.to_string())
                    elif ext.lower() in [".jpg", ".jpeg"]:
                        fh.write(g.create_jpeg())
                    elif ext.lower() == ".pdf":
                        fh.write(g.create_pdf())
                    elif ext.lower() == ".svg":
                        fh.write(g.create_svg())
                    else:
                        raise Exception(
                            "Unknown file format for saving graph: %s" % ext
                        )

            # display graph via matplotlib
            if show:
                png = g.create_png()
                sio = StringIO(png)
                img = mpimg.imread(sio)
                plt.imshow(img, aspect="equal")
                plt.show()

            return g


def ready_to_schedule_operation(op, has_executed, graph):
    """
    Determines if a Operation is ready to be scheduled for execution based on
    what has already been executed.

    Args:
        op:
            The Operation object to check
        has_executed: set
            A set containing all operations that have been executed so far
        graph:
            The networkx graph containing the operations and data nodes
    Returns:
        A boolean indicating whether the operation may be scheduled for
        execution based on what has already been executed.
    """
    dependencies = set(
        filter(lambda v: isinstance(v, Operation), gr.ancestors(graph, op.needs))
    )
    return dependencies.issubset(has_executed)


def ready_to_delete_data_node(name, has_executed, graph):
    """
    Determines if a DataPlaceholderNode is ready to be deleted from the
    cache.

    Args:
        name:
            The name of the data node to check
        has_executed: set
            A set containing all operations that have been executed so far
        graph:
            The networkx graph containing the operations and data nodes
    Returns:
        A boolean indicating whether the data node can be deleted or not.
    """
    data_node = get_data_node(name, graph)
    return set(gr.successors(graph, data_node)).issubset(has_executed)


def get_data_node(name, graph):
    """
    Gets a data node from a graph using its name
    """
    for node in gr.nodes(graph):
        if node == name and isinstance(node, DataPlaceholderNode):
            return node
    return None


# ------------ functional ---------------------------------------------------------------

# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

from itertools import chain

# uses Operation, NetworkOperation from base
# uses Network from network


class FunctionalOperation(Operation):
    def __init__(self, **kwargs):
        self.fn = kwargs.pop("fn")
        Operation.__init__(self, **kwargs)

    def _compute(self, named_inputs, outputs=None):
        inputs = [named_inputs[d] for d in self.needs if not isinstance(d, optional)]

        # Find any optional inputs in named_inputs.  Get only the ones that
        # are present there, no extra `None`s.
        optionals = {
            n: named_inputs[n]
            for n in self.needs
            if isinstance(n, optional) and n in named_inputs
        }

        # Combine params and optionals into one big glob of keyword arguments.
        kwargs = {k: v for d in (self.params, optionals) for k, v in d.items()}
        result = self.fn(*inputs, **kwargs)
        if len(self.provides) == 1:
            result = [result]

        result = zip(self.provides, result)
        if outputs:
            outputs = set(outputs)
            result = filter(lambda x: x[0] in outputs, result)

        return dict(result)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __getstate__(self):
        state = Operation.__getstate__(self)
        state["fn"] = self.__dict__["fn"]
        return state


class operation(Operation):
    """
    This object represents an operation in a computation graph.  Its
    relationship to other operations in the graph is specified via its
    ``needs`` and ``provides`` arguments.

    :param function fn:
        The function used by this operation.  This does not need to be
        specified when the operation object is instantiated and can instead
        be set via ``__call__`` later.

    :param str name:
        The name of the operation in the computation graph.

    :param list needs:
        Names of input data objects this operation requires.  These should
        correspond to the ``args`` of ``fn``.

    :param list provides:
        Names of output data objects this operation provides.

    :param dict params:
        A dict of key/value pairs representing constant parameters
        associated with your operation.  These can correspond to either
        ``args`` or ``kwargs`` of ``fn`.
    """

    def __init__(self, fn=None, **kwargs):
        self.fn = fn
        Operation.__init__(self, **kwargs)

    def _normalize_kwargs(self, kwargs):

        # Allow single value for needs parameter
        if "needs" in kwargs and type(kwargs["needs"]) == str:
            assert kwargs["needs"], "empty string provided for `needs` parameters"
            kwargs["needs"] = [kwargs["needs"]]

        # Allow single value for provides parameter
        if "provides" in kwargs and type(kwargs["provides"]) == str:
            assert kwargs["provides"], "empty string provided for `needs` parameters"
            kwargs["provides"] = [kwargs["provides"]]

        assert kwargs["name"], "operation needs a name"
        assert type(kwargs["needs"]) == list, "no `needs` parameter provided"
        assert type(kwargs["provides"]) == list, "no `provides` parameter provided"
        assert hasattr(
            kwargs["fn"], "__call__"
        ), "operation was not provided with a callable"

        if type(kwargs["params"]) is not dict:
            kwargs["params"] = {}

        return kwargs

    def __call__(self, fn=None, **kwargs):
        """
        This enables ``operation`` to act as a decorator or as a functional
        operation, for example::

            @operator(name='myadd1', needs=['a', 'b'], provides=['c'])
            def myadd(a, b):
                return a + b

        or::

            def myadd(a, b):
                return a + b
            operator(name='myadd1', needs=['a', 'b'], provides=['c'])(myadd)

        :param function fn:
            The function to be used by this ``operation``.

        :return:
            Returns an operation class that can be called as a function or
            composed into a computation graph.
        """

        if fn is not None:
            self.fn = fn

        total_kwargs = {}
        total_kwargs.update(vars(self))
        total_kwargs.update(kwargs)
        total_kwargs = self._normalize_kwargs(total_kwargs)

        return FunctionalOperation(**total_kwargs)

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        return "{}(name='{}', needs={}, provides={}, fn={})".format(
            self.__class__.__name__,
            self.name,
            self.needs,
            self.provides,
            self.fn.__name__,
        )


class compose:
    """
    This is a simple class that's used to compose ``operation`` instances into
    a computation graph.

    :param str name:
        A name for the graph being composed by this object.

    :param bool merge:
        If ``True``, this compose object will attempt to merge together
        ``operation`` instances that represent entire computation graphs.
        Specifically, if one of the ``operation`` instances passed to this
        ``compose`` object is itself a graph operation created by an
        earlier use of ``compose`` the sub-operations in that graph are
        compared against other operations passed to this ``compose``
        instance (as well as the sub-operations of other graphs passed to
        this ``compose`` instance).  If any two operations are the same
        (based on name), then that operation is computed only once, instead
        of multiple times (one for each time the operation appears).
    """

    def __init__(self, name=None, merge=False):
        assert name, "compose needs a name"
        self.name = name
        self.merge = merge

    def __call__(self, *operations):
        """
        Composes a collection of operations into a single computation graph,
        obeying the ``merge`` property, if set in the constructor.

        :param operations:
            Each argument should be an operation instance created using
            ``operation``.

        :return:
            Returns a special type of operation class, which represents an
            entire computation graph as a single operation.
        """
        assert len(operations), "no operations provided to compose"

        # If merge is desired, deduplicate operations before building network
        if self.merge:
            merge_set = set()
            for op in operations:
                if isinstance(op, NetworkOperation):
                    net_ops = filter(lambda x: isinstance(x, Operation), op.net.steps)
                    merge_set.update(net_ops)
                else:
                    merge_set.add(op)
            operations = list(merge_set)

        def order_preserving_uniquifier(seq, seen=None):
            seen = seen if seen else set()
            seen_add = seen.add
            return [x for x in seq if not (x in seen or seen_add(x))]

        provides = order_preserving_uniquifier(
            chain(*[op.provides for op in operations])
        )
        needs = order_preserving_uniquifier(
            chain(*[op.needs for op in operations]), set(provides)
        )

        # compile network
        net = Network()
        for op in operations:
            net.add_op(op)
        net.compile()

        return NetworkOperation(
            name=self.name, needs=needs, provides=provides, params={}, net=net
        )
```

## ext/gk_tests.py

```python
"""
Tests for gk.py
"""

import pytest

from contextlib import suppress

import math
import pickle

from pprint import pprint
from operator import add

with suppress(ModuleNotFoundError, ImportError):
    from numpy.testing import assert_raises

    from meshed.ext.gk import (
        operation,
        compose,
        Operation,
        optional,
        Network,
    )

    def test_network():

        # Sum operation, late-bind compute function
        sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum_ab")(add)

        # sum_op1 is callable
        print(sum_op1(1, 2))

        # Multiply operation, decorate in-place
        @operation(name="mul_op1", needs=["sum_ab", "b"], provides="sum_ab_times_b")
        def mul_op1(a, b):
            return a * b

        # mul_op1 is callable
        print(mul_op1(1, 2))

        # Pow operation
        @operation(
            name="pow_op1",
            needs="sum_ab",
            provides=["sum_ab_p1", "sum_ab_p2", "sum_ab_p3"],
            params={"exponent": 3},
        )
        def pow_op1(a, exponent=2):
            return [math.pow(a, y) for y in range(1, exponent + 1)]

        print(pow_op1._compute({"sum_ab": 2}, ["sum_ab_p2"]))

        # Partial operation that is bound at a later time
        partial_op = operation(
            name="sum_op2", needs=["sum_ab_p1", "sum_ab_p2"], provides="p1_plus_p2"
        )

        # Bind the partial operation
        sum_op2 = partial_op(add)

        # Sum operation, early-bind compute function
        sum_op_factory = operation(add)

        sum_op3 = sum_op_factory(name="sum_op3", needs=["a", "b"], provides="sum_ab2")

        # sum_op3 is callable
        print(sum_op3(5, 6))

        # compose network
        net = compose(name="my network")(sum_op1, mul_op1, pow_op1, sum_op2, sum_op3)

        #
        # Running the network
        #

        # get all outputs
        pprint(net({"a": 1, "b": 2}))

        # get specific outputs
        pprint(net({"a": 1, "b": 2}, outputs=["sum_ab_times_b"]))

        # start with inputs already computed
        pprint(net({"sum_ab": 1, "b": 2}, outputs=["sum_ab_times_b"]))

        # visualize network graph
        # net.plot(show=True)

    def test_network_simple_merge():

        sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
        sum_op2 = operation(name="sum_op2", needs=["a", "b"], provides="sum2")(add)
        sum_op3 = operation(name="sum_op3", needs=["sum1", "c"], provides="sum3")(add)
        net1 = compose(name="my network 1")(sum_op1, sum_op2, sum_op3)
        pprint(net1({"a": 1, "b": 2, "c": 4}))

        sum_op4 = operation(name="sum_op1", needs=["d", "e"], provides="a")(add)
        sum_op5 = operation(name="sum_op2", needs=["a", "f"], provides="b")(add)
        net2 = compose(name="my network 2")(sum_op4, sum_op5)
        pprint(net2({"d": 1, "e": 2, "f": 4}))

        net3 = compose(name="merged")(net1, net2)
        pprint(net3({"c": 5, "d": 1, "e": 2, "f": 4}))

    def test_network_deep_merge():

        sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
        sum_op2 = operation(name="sum_op2", needs=["a", "b"], provides="sum2")(add)
        sum_op3 = operation(name="sum_op3", needs=["sum1", "c"], provides="sum3")(add)
        net1 = compose(name="my network 1")(sum_op1, sum_op2, sum_op3)
        pprint(net1({"a": 1, "b": 2, "c": 4}))

        sum_op4 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
        sum_op5 = operation(name="sum_op4", needs=["sum1", "b"], provides="sum2")(add)
        net2 = compose(name="my network 2")(sum_op4, sum_op5)
        pprint(net2({"a": 1, "b": 2}))

        net3 = compose(name="merged", merge=True)(net1, net2)
        pprint(net3({"a": 1, "b": 2, "c": 4}))

    def test_input_based_pruning():
        # Tests to make sure we don't need to pass graph inputs if we're provided
        # with data further downstream in the graph as an input.

        sum1 = 2
        sum2 = 5

        # Set up a net such that if sum1 and sum2 are provided directly, we don't
        # need to provide a and b.
        sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
        sum_op2 = operation(name="sum_op2", needs=["a", "b"], provides="sum2")(add)
        sum_op3 = operation(name="sum_op3", needs=["sum1", "sum2"], provides="sum3")(
            add
        )
        net = compose(name="test_net")(sum_op1, sum_op2, sum_op3)

        results = net({"sum1": sum1, "sum2": sum2})

        # Make sure we got expected result without having to pass a or b.
        assert "sum3" in results
        assert results["sum3"] == add(sum1, sum2)

    def test_output_based_pruning():
        # Tests to make sure we don't need to pass graph inputs if they're not
        # needed to compute the requested outputs.

        c = 2
        d = 3

        # Set up a network such that we don't need to provide a or b if we only
        # request sum3 as output.
        sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
        sum_op2 = operation(name="sum_op2", needs=["c", "d"], provides="sum2")(add)
        sum_op3 = operation(name="sum_op3", needs=["c", "sum2"], provides="sum3")(add)
        net = compose(name="test_net")(sum_op1, sum_op2, sum_op3)

        results = net({"c": c, "d": d}, outputs=["sum3"])

        # Make sure we got expected result without having to pass a or b.
        assert "sum3" in results
        assert results["sum3"] == add(c, add(c, d))

    def test_input_output_based_pruning():
        # Tests to make sure we don't need to pass graph inputs if they're not
        # needed to compute the requested outputs or of we're provided with
        # inputs that are further downstream in the graph.

        c = 2
        sum2 = 5

        # Set up a network such that we don't need to provide a or b d if we only
        # request sum3 as output and if we provide sum2.
        sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
        sum_op2 = operation(name="sum_op2", needs=["c", "d"], provides="sum2")(add)
        sum_op3 = operation(name="sum_op3", needs=["c", "sum2"], provides="sum3")(add)
        net = compose(name="test_net")(sum_op1, sum_op2, sum_op3)

        results = net({"c": c, "sum2": sum2}, outputs=["sum3"])

        # Make sure we got expected result without having to pass a, b, or d.
        assert "sum3" in results
        assert results["sum3"] == add(c, sum2)

    def test_pruning_raises_for_bad_output():
        # Make sure we get a ValueError during the pruning step if we request an
        # output that doesn't exist.

        # Set up a network that doesn't have the output sum4, which we'll request
        # later.
        sum_op1 = operation(name="sum_op1", needs=["a", "b"], provides="sum1")(add)
        sum_op2 = operation(name="sum_op2", needs=["c", "d"], provides="sum2")(add)
        sum_op3 = operation(name="sum_op3", needs=["c", "sum2"], provides="sum3")(add)
        net = compose(name="test_net")(sum_op1, sum_op2, sum_op3)

        # Request two outputs we can compute and one we can't compute.  Assert
        # that this raises a ValueError.
        assert_raises(
            ValueError,
            net,
            {"a": 1, "b": 2, "c": 3, "d": 4},
            outputs=["sum1", "sum3", "sum4"],
        )

    def test_optional():
        # Test that optional() needs work as expected.

        # Function to add two values plus an optional third value.
        def addplusplus(a, b, c=0):
            return a + b + c

        sum_op = operation(
            name="sum_op1",
            needs=["a", "b", optional("c")],
            provides="sum",
        )(addplusplus)

        net = compose(name="test_net")(sum_op)

        # Make sure output with optional arg is as expected.
        named_inputs = {"a": 4, "b": 3, "c": 2}
        results = net(named_inputs)
        assert "sum" in results
        assert results["sum"] == sum(named_inputs.values())

        # Make sure output without optional arg is as expected.
        named_inputs = {"a": 4, "b": 3}
        results = net(named_inputs)
        assert "sum" in results
        assert results["sum"] == sum(named_inputs.values())

    def test_deleted_optional():
        # Test that DeleteInstructions included for optionals do not raise
        # exceptions when the corresponding input is not prodided.

        # Function to add two values plus an optional third value.
        def addplusplus(a, b, c=0):
            return a + b + c

        # Here, a DeleteInstruction will be inserted for the optional need 'c'.
        sum_op1 = operation(
            name="sum_op1",
            needs=["a", "b", optional("c")],
            provides="sum1",
        )(addplusplus)
        sum_op2 = operation(name="sum_op2", needs=["sum1", "sum1"], provides="sum2")(
            add
        )
        net = compose(name="test_net")(sum_op1, sum_op2)

        # DeleteInstructions are used only when a subset of outputs are requested.
        results = net({"a": 4, "b": 3}, outputs=["sum2"])
        assert "sum2" in results

    # Skip this test since it requires a long time to run
    @pytest.mark.skip(reason="This test takes a long time to run")
    def test_parallel_execution():
        import time

        def fn(x):
            time.sleep(1)
            print("fn %s" % (time.time() - t0))
            return 1 + x

        def fn2(a, b):
            time.sleep(1)
            print("fn2 %s" % (time.time() - t0))
            return a + b

        def fn3(z, k=1):
            time.sleep(1)
            print("fn3 %s" % (time.time() - t0))
            return z + k

        pipeline = compose(name="l", merge=True)(
            # the following should execute in parallel under threaded execution mode
            operation(name="a", needs="x", provides="ao")(fn),
            operation(name="b", needs="x", provides="bo")(fn),
            # this should execute after a and b have finished
            operation(name="c", needs=["ao", "bo"], provides="co")(fn2),
            operation(name="d", needs=["ao", optional("k")], provides="do")(fn3),
            operation(name="e", needs=["ao", "bo"], provides="eo")(fn2),
            operation(name="f", needs="eo", provides="fo")(fn),
            operation(name="g", needs="fo", provides="go")(fn),
        )

        t0 = time.time()
        pipeline.set_execution_method("parallel")
        result_threaded = pipeline({"x": 10}, ["co", "go", "do"])
        print("threaded result")
        print(result_threaded)

        t0 = time.time()
        pipeline.set_execution_method("sequential")
        result_sequential = pipeline({"x": 10}, ["co", "go", "do"])
        print("sequential result")
        print(result_sequential)

        # make sure results are the same using either method
        assert result_sequential == result_threaded

    def test_multi_threading():
        import time
        import random
        from multiprocessing.dummy import Pool

        def op_a(a, b):
            time.sleep(random.random() * 0.02)
            return a + b

        def op_b(c, b):
            time.sleep(random.random() * 0.02)
            return c + b

        def op_c(a, b):
            time.sleep(random.random() * 0.02)
            return a * b

        pipeline = compose(name="pipeline", merge=True)(
            operation(name="op_a", needs=["a", "b"], provides="c")(op_a),
            operation(name="op_b", needs=["c", "b"], provides="d")(op_b),
            operation(name="op_c", needs=["a", "b"], provides="e")(op_c),
        )

        def infer(i):
            # data = open("616039-bradpitt.jpg").read()
            outputs = ["c", "d", "e"]
            results = pipeline({"a": 1, "b": 2}, outputs)
            assert tuple(sorted(results.keys())) == tuple(sorted(outputs)), (
                outputs,
                results,
            )
            return results

        N = 100
        for i in range(20, 200):
            pool = Pool(i)
            pool.map(infer, range(N))
            pool.close()

    ####################################
    # Backwards compatibility
    ####################################

    # Classes must be defined as members of __main__ for pickleability

    # We first define some basic operations
    class Sum(Operation):
        def compute(self, inputs):
            a = inputs[0]
            b = inputs[1]
            return [a + b]

    class Mul(Operation):
        def compute(self, inputs):
            a = inputs[0]
            b = inputs[1]
            return [a * b]

    # This is an example of an operation that takes a parameter.
    # It also illustrates an operation that returns multiple outputs
    class Pow(Operation):
        def compute(self, inputs):

            a = inputs[0]
            outputs = []
            for y in range(1, self.params["exponent"] + 1):
                p = math.pow(a, y)
                outputs.append(p)
            return outputs

    def test_backwards_compatibility():

        sum_op1 = Sum(name="sum_op1", provides=["sum_ab"], needs=["a", "b"])
        mul_op1 = Mul(
            name="mul_op1", provides=["sum_ab_times_b"], needs=["sum_ab", "b"]
        )
        pow_op1 = Pow(
            name="pow_op1",
            needs=["sum_ab"],
            provides=["sum_ab_p1", "sum_ab_p2", "sum_ab_p3"],
            params={"exponent": 3},
        )
        sum_op2 = Sum(
            name="sum_op2",
            provides=["p1_plus_p2"],
            needs=["sum_ab_p1", "sum_ab_p2"],
        )

        net = Network()
        net.add_op(sum_op1)
        net.add_op(mul_op1)
        net.add_op(pow_op1)
        net.add_op(sum_op2)
        net.compile()

        # try the pickling part
        pickle.dumps(net)

        #
        # Running the network
        #

        # get all outputs
        pprint(net.compute(outputs=None, named_inputs={"a": 1, "b": 2}))

        # get specific outputs
        pprint(net.compute(outputs=["sum_ab_times_b"], named_inputs={"a": 1, "b": 2}))

        # start with inputs already computed
        pprint(
            net.compute(outputs=["sum_ab_times_b"], named_inputs={"sum_ab": 1, "b": 2})
        )
```

## itools.py

```python
"""Functions that provide iterators of g elements where g is any
adjacency Mapping representation.

"""

from typing import (
    Any,
    List,
    TypeVar,
    Union,
    Optional,
)
from collections.abc import Mapping, Sized, MutableMapping, Iterable, Callable
from itertools import product, chain
from functools import wraps, reduce, partial
from collections import defaultdict
from random import sample, randint
from operator import or_

from i2.signatures import Sig

N = TypeVar("N")
Graph = Mapping[N, Iterable[N]]
MutableGraph = MutableMapping[N, Iterable[N]]


def _import_or_raise(module_name, pip_install_name: str | bool | None = None):
    try:
        return __import__(module_name)
    except ImportError as e:
        if pip_install_name is True:
            pip_install_name = module_name  # use the module name as the install name
        if pip_install_name:
            msg = f"You can install it by running: `pip install {pip_install_name}`"
        else:
            msg = "Please install it first."
        raise ImportError(f"Could not import {module_name}. {msg}") from e


def random_graph(n_nodes: int = 7):
    """Get a random graph.

    >>> random_graph()  # doctest: +SKIP
    {0: [6, 3, 5, 2],
     1: [3, 2, 0, 6],
     2: [5, 6, 4, 0],
     3: [1, 0, 5, 6, 3],
     4: [],
     5: [1, 5, 3, 6],
     6: [4, 3, 1]}
    >>> random_graph(3)  # doctest: +SKIP
    {0: [0], 1: [0], 2: []}
    """
    nodes = range(n_nodes)

    def gen():
        for src in nodes:
            n_dst = randint(0, n_nodes - 1)
            dst = sample(nodes, n_dst)
            yield src, list(dst)

    return dict(gen())


def graphviz_digraph(d: Graph):
    """Makes a graphviz graph using the links specified by dict d"""
    graphviz = _import_or_raise("graphviz", "graphviz")
    dot = graphviz.Digraph()
    for k, v in d.items():
        for vv in v:
            dot.edge(vv, k)
    return dot


def _handle_exclude_nodes(func: Callable):
    sig = Sig(func)

    @wraps(func)
    def _func(*args, **kwargs):
        kwargs = sig.map_arguments(args, kwargs, apply_defaults=True)
        try:
            _exclude_nodes = kwargs["_exclude_nodes"]
        except KeyError:
            raise RuntimeError(f"{func} doesn't have a _exclude_nodes argument")

        if _exclude_nodes is None:
            _exclude_nodes = set()
        elif not isinstance(_exclude_nodes, set):
            _exclude_nodes = set(_exclude_nodes)

        kwargs["_exclude_nodes"] = _exclude_nodes
        args, kwargs = sig.mk_args_and_kwargs(kwargs)
        return func(*args, **kwargs)

    return _func


def add_edge(g: MutableGraph, node1, node2):
    """Add an edge FROM node1 TO node2"""
    if node1 in g:
        g[node1].append(node2)
    else:
        g[node1] = [node2]


def edges(g: Graph):
    """Generates edges of graph, i.e. ``(from_node, to_node)`` tuples.

    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> assert sorted(edges(g)) == [
    ...     ('a', 'c'), ('b', 'c'), ('b', 'e'), ('c', 'a'), ('c', 'b'), ('c', 'd'),
    ...     ('c', 'e'), ('d', 'c'), ('e', 'c'), ('e', 'z')]
    """
    for src in g:
        for dst in g[src]:
            yield src, dst


def nodes(g: Graph):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> sorted(nodes(g))
    ['a', 'b', 'c', 'd', 'e', 'f', 'z']
    """
    seen = set()
    for src in g:
        if src not in seen:
            yield src
            seen.add(src)
        for dst in g[src]:
            if dst not in seen:
                yield dst
                seen.add(dst)


def has_node(g: Graph, node, check_adjacencies=True):
    """Returns True if the graph has given node

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2]
    ... }
    >>> has_node(g, 0)
    True
    >>> has_node(g, 2)
    True

    Note that 2 was found, though it's not a key of ``g``.
    This shows that we don't have to have an explicit ``{2: []}`` in ``g``
    to be able to see that it's a node of ``g``.
    The function will go through the values of the mapping to try to find it
    if it hasn't been found before in the keys.

    This can be inefficient, so if that matters, you can express your
    graph ``g`` so that all nodes are explicitly declared as keys, and
    use ``check_adjacencies=False`` to tell the function not to look into
    the values of the ``g`` mapping.

    >>> has_node(g, 2, check_adjacencies=False)
    False
    >>> g = {
    ...     0: [1, 2],
    ...     1: [2],
    ...     2: []
    ... }
    >>> has_node(g, 2, check_adjacencies=False)
    True

    """
    if node in g:
        return True

    if check_adjacencies:
        # look in the adjacencies (because a leaf might not show up as a
        # {leaf: []} item in g!
        for adjacencies in g.values():
            if node in adjacencies:
                return True

    return False  # if not found before


@_handle_exclude_nodes
def successors(g: Graph, node, _exclude_nodes=None):
    """Iterator of nodes that have directed paths FROM node

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [1, 4],
    ...     3: [4]}
    >>> assert set(successors(g, 1)) == {1, 2, 3, 4}
    >>> assert set(successors(g, 3)) == {4}
    >>> assert set(successors(g, 4)) == set()

    Notice that 1 is a successor of 1 here because there's a 1-2-1 directed path
    """
    direct_successors = set(g.get(node, [])) - _exclude_nodes
    yield from direct_successors
    _exclude_nodes.update(direct_successors)
    for successor_node in direct_successors:
        yield from successors(g, successor_node, _exclude_nodes)


def predecessors(g: Graph, node):
    """Iterator of nodes that have directed paths TO node

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [1, 4],
    ...     3: [4]}
    >>> set(predecessors(g, 4))
    {0, 1, 2, 3}
    >>> set(predecessors(g, 2))
    {0, 1, 2}
    >>> set(predecessors(g, 0))
    set()

    Notice that 2 is a predecessor of 2 here because of the presence
    of a 2-1-2 directed path.
    """
    yield from successors(edge_reversed_graph(g), node)


def _split_if_str(x):
    """
    If source is a string, the `str.split()` of it will be returned.

    This is to be used in situations where we deal with lists of strings and
    want to avoid mistaking a single string input with an iterable of characters.

    For example, if a user specifies ``'abc'`` as an argument, this could have the
    same effect as specifying  ``['a', 'b', 'c']``,
    which often not what's intended, but rather ``['abc']`` is intended).

    If the user actually wants ``['a', 'b', 'c']``, they can specify it by doing
    ``list('abc')`` explicitly.
    """
    if isinstance(x, str):
        return x.split()
    else:
        return x


def children(g: Graph, source: Iterable[N]):
    """Set of all nodes (not in source) adjacent FROM 'source' in 'g'

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [1, 4],
    ...     3: [4]
    ... }
    >>> children(g, [2, 3])
    {1, 4}
    >>> children(g, [4])
    set()
    """
    source = _split_if_str(source)
    source = set(source)
    _children = set()
    for node in source:
        _children.update(g.get(node, set()))
    return _children - source


def parents(g: Graph, source: Iterable[N]):
    """Set of all nodes (not in source) adjacent TO 'source' in 'g'

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [1, 4],
    ...     3: [4]
    ... }
    >>> parents(g, [2, 3])
    {0, 1}
    >>> parents(g, [0])
    set()
    """
    return children(edge_reversed_graph(g), source)


@_handle_exclude_nodes
def ancestors(g: Graph, source: Iterable[N], _exclude_nodes=None):
    """Set of all nodes (not in source) reachable TO `source` in `g`.

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [4],
    ...     3: [4]
    ... }
    >>> ancestors(g, [2, 3])
    {0, 1}
    >>> ancestors(g, [0])
    set()
    """
    source = _split_if_str(source)
    assert isinstance(source, Iterable)
    source = set(source) - _exclude_nodes
    _parents = (set(parents(g, source)) - source) - _exclude_nodes
    if not _parents:
        return set()
    else:
        _ancestors_of_parent = ancestors(g, _parents, _exclude_nodes)

        return _parents | _ancestors_of_parent


def descendants(g: Graph, source: Iterable[N], _exclude_nodes=None):
    """Returns the set of all nodes reachable FROM `source` in `g`.

    >>> g = {
    ...     0: [1, 2],
    ...     1: [2, 3, 4],
    ...     2: [4],
    ...     3: [4]
    ... }
    >>> descendants(g, [2, 3])
    {4}
    >>> descendants(g, [4])
    set()
    """
    return ancestors(edge_reversed_graph(g), source, _exclude_nodes)


# TODO: Can serious be optimized, and hasn't been tested much: Revise
def root_nodes(g: Graph):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> sorted(root_nodes(g))
    ['f']

    Note that `f` is present: Isolated nodes are considered both as
    root and leaf nodes both.
    """
    nodes_having_parents = set(chain.from_iterable(g.values()))
    return set(g) - set(nodes_having_parents)


# TODO: Can be made much more efficient, by looking at the ancestors code itself
def root_ancestors(graph: dict, nodes: str | Iterable[str]):
    """
    Returns the roots of the sub-dag that contribute to compute the given nodes.
    """
    if isinstance(nodes, str):
        nodes = nodes.split()
    get_ancestors = partial(ancestors, graph)
    ancestors_of_nodes = reduce(or_, map(get_ancestors, nodes), set())
    return ancestors_of_nodes & set(root_nodes(graph))


# TODO: Can serious be optimized, and hasn't been tested much: Revise
def leaf_nodes(g: Graph):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> sorted(leaf_nodes(g))
    ['f', 'z']

    Note that `f` is present: Isolated nodes are considered both as
    root and leaf nodes both.
    """
    return root_nodes(edge_reversed_graph(g))


def isolated_nodes(g: Graph):
    """Nodes that
    >>> g = dict(a='c', b='ce', c=list('abde'), d='c', e=['c', 'z'], f={})
    >>> set(isolated_nodes(g))
    {'f'}
    """
    for src in g:
        if not next(
            iter(g[src]), False
        ):  # Note: Slower than just `not g[src]`, but safer
            yield src


def find_path(g: Graph, src, dst, path=None):
    """find a path from src to dst nodes in graph

    >>> g = dict(a='c', b='ce', c=list('abde'), d='c', e=['c', 'z'], f={})
    >>> find_path(g, 'a', 'c')
    ['a', 'c']
    >>> find_path(g, 'a', 'b')
    ['a', 'c', 'b']
    >>> find_path(g, 'a', 'z')
    ['a', 'c', 'b', 'e', 'z']
    >>> assert find_path(g, 'a', 'f') == None

    """
    if path == None:
        path = []
    path = path + [src]
    if src == dst:
        return path
    if src not in g:
        return None
    for node in g[src]:
        if node not in path:
            extended_path = find_path(g, node, dst, path)
            if extended_path:
                return extended_path
    return None


def reverse_edges(g: Graph):
    """Generator of reversed edges. Like edges but with inverted edges.

    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> assert sorted(reverse_edges(g)) == [
    ...     ('a', 'c'), ('b', 'c'), ('c', 'a'), ('c', 'b'), ('c', 'd'), ('c', 'e'),
    ...     ('d', 'c'), ('e', 'b'), ('e', 'c'), ('z', 'e')]

    NOTE: Not to be confused with  ``edge_reversed_graph`` which inverts the direction
    of edges.
    """
    for src, dst_nodes in g.items():
        yield from product(dst_nodes, src)


def has_cycle(g: Graph) -> list[N]:
    """
        Returns a list representing a cycle in the graph if any. An empty list indicates no cycle.

        :param g: The graph to check for cycles, represented as a dictionary where keys are nodes
                  and values are lists of nodes pointing to the key node (parents of the key node).

        Example usage:
        >>> g = dict(e=['c', 'd'], c=['b'], d=['b'], b=['a'])
        >>> has_cycle(g)
        []

        >>> g['a'] = ['e']  # Introducing a cycle
        >>> has_cycle(g)
        ['e', 'c', 'b', 'a', 'e']

    Design notes:

    - **Graph Representation**: The graph is interpreted such that each key is a child node,
    and the values are lists of its parents. This representation requires traversing
    the graph in reverse, from child to parent, to detect cycles.
    I regret this design choice, which was aligned with the original problem that was
    being solved, but which doesn't follow the usual representation of a graph.
    - **Consistent Return Type**: The function systematically returns a list. A non-empty
    list indicates a cycle (showing the path of the cycle), while an empty list indicates
    the absence of a cycle.
    - **Depth-First Search (DFS)**: The function performs a DFS on the graph to detect
    cycles. It uses a recursion stack (rec_stack) to track the path being explored and
    a visited set (visited) to avoid re-exploring nodes.
    - **Cycle Detection and Path Reconstruction**: When a node currently in the recursion
    stack is encountered again, a cycle is detected. The function then reconstructs the
    cycle path from the current path explored, including the start and end node to
    illustrate the cycle closure.
    - **Efficient Backtracking**: After exploring a node's children, the function
    backtracks by removing the node from the recursion stack and the current path,
    ensuring accurate path tracking for subsequent explorations.

    """
    visited = set()  # Tracks visited nodes to avoid re-processing
    rec_stack = set()  # Tracks nodes currently in the recursion stack to detect cycles

    def _has_cycle(node, path):
        """
        Helper function to perform DFS on the graph and detect cycles.
        :param node: Current node being processed
        :param path: Current path taken from the start node to the current node
        :return: List representing the cycle, empty if no cycle is found
        """
        if node in rec_stack:
            # Cycle detected, return the cycle path including the current node for closure
            cycle_start_index = path.index(node)
            return path[cycle_start_index:] + [node]
        if node in visited:
            # Node already processed and didn't lead to a cycle, skip
            return []

        # Mark the current node as visited and add to the recursion stack
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        # Explore all parent nodes
        for parent in g.get(node, []):
            cycle_path = _has_cycle(parent, path)
            if cycle_path:
                # Cycle found in the path of the parent node
                return cycle_path

        # Current path didn't lead to a cycle, backtrack
        rec_stack.remove(node)
        path.pop()

        return []

    # Iterate over all nodes to ensure disconnected components are also checked
    for node in g:
        cycle_path = _has_cycle(node, [])
        if cycle_path:
            # Return the first cycle found
            return cycle_path

    # No cycle found in any component of the graph
    return []


def out_degrees(g: Graph):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> assert dict(out_degrees(g)) == (
    ...     {'a': 1, 'b': 2, 'c': 4, 'd': 1, 'e': 2, 'f': 0}
    ... )
    """
    for src, dst_nodes in g.items():
        yield src, len(dst_nodes)


def in_degrees(g: Graph):
    """
    >>> g = dict(a='c', b='ce', c='abde', d='c', e=['c', 'z'], f={})
    >>> assert dict(in_degrees(g)) == (
    ... {'a': 1, 'b': 1, 'c': 4,  'd': 1, 'e': 2, 'f': 0, 'z': 1}
    ... )
    """
    return out_degrees(edge_reversed_graph(g))


def copy_of_g_with_some_keys_removed(g: Graph, keys: Iterable):
    keys = _split_if_str(keys)
    return {k: v for k, v in g.items() if k not in keys}


def _topological_sort_helper(g, parent, visited, stack):
    """A recursive function to service topological_sort"""

    visited.add(parent)  # Mark the current node as visited.

    # Recurse for all the vertices adjacent to this node
    for child in reversed(g.get(parent, [])):
        if child not in visited:
            _topological_sort_helper(g, child, visited, stack)

    # Push current node to stack which stores result
    stack.insert(0, parent)
    # print(f"  Inserted {parent}: {stack=}")


def topological_sort(g: Graph):
    """Return the list of nodes in topological sort order.

    This order is such that a node parents will all occur before;
        If order[i] is parent of order[j] then i < j

    This is often used to compute the order of computation in a DAG.

    >>> g = {
    ...     0: [4, 2],
    ...     4: [3, 1],
    ...     2: [3],
    ...     3: [1]
    ... }
    >>>
    >>> list(topological_sort(g))
    [0, 4, 2, 3, 1]

    Here's an ascii art of the graph, to verify that the topological sort is
    indeed as expected.

    .. code-block::
    ┌───┐     ┌───┐     ┌───┐     ┌───┐
    │ 0 │ ──▶ │ 2 │ ──▶ │ 3 │ ──▶ │ 1 │
    └───┘     └───┘     └───┘     └───┘
      │                   ▲         ▲
      │                   │         │
      ▼                   │         │
    ┌───┐                 │         │
    │ 4 │ ────────────────┼─────────┘
    └───┘                 │
      │                   │
      └───────────────────┘
    """
    visited = set()
    stack = []

    # Call the recursive helper function to accumulate topological sorts
    # starting from all vertices one by one
    for parent in reversed(g):
        if parent not in visited:
            # print(f"Processing {parent}")
            _topological_sort_helper(g, parent, visited, stack)

    return stack


def edge_reversed_graph(
    g: Graph,
    dst_nodes_factory: Callable[[], Iterable[N]] = list,
    dst_nodes_append: Callable[[Iterable[N], N], None] = list.append,
) -> Graph:
    """Invert the from/to direction of the edges of the graph.

    >>> g = dict(a='c', b='cd', c='abd', e='')
    >>> assert edge_reversed_graph(g) == {
    ...     'c': ['a', 'b'], 'd': ['b', 'c'], 'a': ['c'], 'b': ['c'], 'e': []}
    >>> reverse_g_with_sets = edge_reversed_graph(g, set, set.add)
    >>> assert reverse_g_with_sets == {
    ...     'c': {'a', 'b'}, 'd': {'b', 'c'}, 'a': {'c'}, 'b': {'c'}, 'e': set([])}

    Testing border cases
    >>> assert edge_reversed_graph(dict(e='', a='e')) == {'e': ['a'], 'a': []}
    >>> assert edge_reversed_graph(dict(a='e', e='')) == {'e': ['a'], 'a': []}
    """
    # Pattern: Groupby logic

    d = defaultdict(dst_nodes_factory)
    for src, dst_nodes in g.items():
        d.setdefault(src, dst_nodes_factory())  # add node if not present
        for dst in dst_nodes:  # empty iterable does nothing
            dst_nodes_append(d[dst], src)
    return d


# A possibly faster way to find descendant of a node in a directed ACYCLIC graph
#
# def find_descendants(d, key):
#     """
#     >>> g = dict(a='c', b='ce', c='de', e=['z'], x=['y', 'w'], tt='y')
#     >>> sorted(find_descendants(g, 'a'))
#     ['a', 'c', 'd', 'e', 'z']
#     """
#
#     yield key
#     try:
#         direct_neighbors = d[key]
#         for n in direct_neighbors:
#             yield from find_descendants(d, n)
#     except KeyError:
#         pass


def filter_dict_with_list_values(d, condition):
    return {k: list(filter(condition, d[k])) for k, v in d.items()}


def filter_dict_on_keys(d, condition):
    return {k: v for k, v in d.items() if condition(k, v)}


def nodes_of_graph(graph):
    return {*graph.keys(), *graph.values()}


def subtract_subgraph(graph, subgraph):
    subnodes = nodes_of_graph(subgraph)
    is_not_subnode = lambda x: x not in subnodes
    not_only_subnodes = lambda x: set(x).issubset(set(subnodes))
    is_not_empty = lambda k, v: len(v) > 0
    key_not_subnode = lambda k, v: k not in subnodes
    graph = filter_dict_with_list_values(graph, is_not_subnode)
    graph = filter_dict_on_keys(graph, is_not_empty)
    graph = filter_dict_on_keys(graph, key_not_subnode)

    return graph
```

## makers.py

```python
r"""Makers

This module contains tools to make meshed objects in different ways.

Let's start with an example where we have some code representing a user story:

>>> def user_story():
...     wfs = call(src_to_wf, data_src)
...     chks_iter = map(chunker, wfs)
...     chks = chain(chks_iter)
...     fvs = map(featurizer, chks)
...     model_outputs = map(model, fvs)

If the code is compliant (has only function calls and assignments of their result),
we can extract ``FuncNode`` factories from these lines (uses AST behind the scenes).

>>> from meshed.makers import src_to_func_node_factory
>>> fnodes_factories = list(src_to_func_node_factory(user_story))

Each factory is a curried version of ``FuncNode``, set up to be able to make a ``DAG``
equivalent to the user story, once we provide the necessary functions (``call``,
``map``, and ``chain``).

>>> from functools import partial
>>> assert all(
... isinstance(x, partial) and issubclass(x.func, FuncNode) for x in fnodes_factories
... )

See that the ``FuncNode`` factories are all set up with
``name`` (id),
``out`` (output variable name),
``bind`` (names of the variables where the function will source it's arguments), and
``func_label`` (which can be used when displaying the DAG, or as a key to the function
to use).

>>> assert [x.keywords for x in fnodes_factories] == [
...  {'name': 'call',
...   'out': 'wfs',
...   'bind': {0: 'src_to_wf', 1: 'data_src'},
...   'func_label': 'call'},
...  {'name': 'map',
...   'out': 'chks_iter',
...   'bind': {0: 'chunker', 1: 'wfs'},
...   'func_label': 'map'},
...  {'name': 'chain',
...   'out': 'chks',
...   'bind': {0: 'chks_iter'},
...   'func_label': 'chain'},
...  {'name': 'map_04',
...   'out': 'fvs',
...   'bind': {0: 'featurizer', 1: 'chks'},
...   'func_label': 'map'},
...  {'name': 'map_05',
...   'out': 'model_outputs',
...   'bind': {0: 'model', 1: 'fvs'},
...   'func_label': 'map'}
... ]

What can we do with that?

Well, provide the functions, so the DAG can actually compute.

You can do it yourself, or get a little help with ``mk_fnodes_from_fn_factories``.

>>> from meshed.dag import DAG
>>> from meshed.makers import mk_fnodes_from_fn_factories
>>> fnodes = list(mk_fnodes_from_fn_factories(fnodes_factories))
>>> dag = DAG(fnodes)
>>> print(dag.synopsis_string())
src_to_wf,data_src -> call -> wfs
chunker,wfs -> map -> chks_iter
chks_iter -> chain -> chks
featurizer,chks -> map_04 -> fvs
model,fvs -> map_05 -> model_outputs

Wait! But we didn't actually provide the functions we wanted to use!
What happened?!?
What happened is that ``mk_fnodes_from_fn_factories`` just made some for us.
It used the convenient ``meshed.util.mk_place_holder_func`` which makes a function
(that happens to actually compute something and be picklable).

>>> from inspect import signature
>>> str(signature(dag))
'(src_to_wf, data_src, chunker, featurizer, model)'

We can actually call the ``dag`` and get something meaningful:

>>> dag(1, 2, 3, 4, 5)
'map(model=5, fvs=map(featurizer=4, chks=chain(chks_iter=map(chunker=3, wfs=call(src_to_wf=1, data_src=2)))))'

If you don't want ``mk_fnodes_from_fn_factories`` to do that (because you are in
prod and need to make sure as much as possible is explicitly as expected, you can
simply use a different ``factory_to_func`` argument. The default one is:

>>> from meshed.makers import dlft_factory_to_func

which you can also reuse to make your own.
See below how we provide a ``name_to_func_map`` to specify how ``func_label``s should
map to actual functions, and set ``use_place_holder_fallback=False`` to make
sure that we don't ever fallback on a placeholder function as we did above.

>>> def _call(x, y):
...     # would use operator.methodcaller('__call__') but doesn't have a __name__
...     return x + y
>>> def _map(x, y):
...     return [x, y]
>>> def _chain(iterable):
...     return sum(iterable)
>>>
>>> factory_to_func = partial(
...     dlft_factory_to_func,
...     name_to_func_map={'map': _map, 'chain': _chain, 'call': _call},
...     use_place_holder_fallback=False
... )
>>>
>>> fnodes = list(mk_fnodes_from_fn_factories(fnodes_factories, factory_to_func))
>>> dag = DAG(fnodes)

On the surface, we get the same dag as we had before -- at least from the point of view
of the dag signature, names, and relationships between these names:

>>> print(dag.synopsis_string())
src_to_wf,data_src -> call -> wfs
chunker,wfs -> map -> chks_iter
chks_iter -> chain -> chks
featurizer,chks -> map_04 -> fvs
model,fvs -> map_05 -> model_outputs
>>> str(signature(dag))
'(src_to_wf, data_src, chunker, featurizer, model)'

But see below that the dag is now using the functions we specified:

>>> # dag(src_to_wf=1, data_src=2, chunker=3, featurizer=4, model=5)
>>> # will trigger this:
>>> # src_to_wf=1, data_src=2 -> call -> wfs == 1 + 2 == 3
>>> # chunker=3 , wfs=3 -> map -> chks_iter == [3, 3]
>>> # chks_iter=6 -> chain -> chks == 3 + 3 == 6
>>> # featurizer=4, chks=6 -> map_04 -> fvs == [4, 6]
>>> # model=5, fvs=[4, 6] -> map_05 -> model_outputs == [5, [4, 6]]
>>> dag(1, 2, 3, 4, 5)
[5, [4, 6]]

"""

import ast
import re
import inspect
from operator import itemgetter, attrgetter
from typing import (
    Tuple,
    Optional,
    TypeVar,
    Dict,
    Union,
)
from collections.abc import Iterator, Iterable, Callable, Mapping
from functools import partial


from i2 import Sig, name_of_obj, partialx, double_up_as_factory
from i2.signatures import name_of_obj


from meshed.dag import DAG
from meshed.base import FuncNode
from meshed.util import mk_place_holder_func, ordered_set_operations


T = TypeVar("T")

# Some restrictions exist and need to be clarified or removed (i.e. more cases handled)
# For example,
# * can't reuse a variable (would lead to same node)
# * x = y (or x, y = tup) not handled (but could easily by binding)
# We don't need these cases to be handled, only x = func(...) forms lead to Turing (I
# think...):
# Further other cases are not handled, but we don't want to handle ALL of python
# -- just a sufficiently expressive subset.


def attr_dict(obj):
    return {a: getattr(obj, a) for a in dir(obj) if not a.startswith("_")}


def is_from_ast_module(o):
    return getattr(type(o), "__module__", "").startswith("_ast")


def _ast_info_str(x):
    if hasattr(x, "lineno"):
        return f"lineno={x.lineno}"
    else:
        return "lineno=unknown"


def _itemgetter(sequence, keys=()):
    if len(keys) == 1:
        key = keys[0]
        return sequence[key]
    return tuple(sequence[i] for i in keys)


def signed_itemgetter(*keys):
    """Like ``operator.itemgetter``, except has a signature, which we needed"""
    return partialx(_itemgetter, keys=keys, _rm_partialize=True)


def _ast_unparse(node):
    try:
        import astunparse

        return astunparse.unparse(node)
    except (ImportError, ModuleNotFoundError):
        return "<to see the code, pip install astunparse>"


def _error_handler(body, info=None):
    info = _ast_info_str(body)
    if isinstance(body, ast.If):
        raise ValueError(
            f"At {info}: You cannot have if statements. "
            f"Replace them with functional equivalents. ({body=})\n"
            f"{_ast_unparse(body)}"
        )
    else:
        raise ValueError(
            f"Couldn't find a handler for parsing body with {info} ({body=})\n"
            f"{_ast_unparse(body)}"
        )


def parse_body(body, *, body_index=None):
    info = _ast_info_str(body)
    if isinstance(body, (ast.Assign, ast.AnnAssign)):
        return parse_assignment(body, info=info)
    elif isinstance(body, ast.Expr) and isinstance(body.value, ast.Call):
        dummy_var = ast.Name(id=f"_{body_index}", ctx=ast.Store())
        new_assign = ast.Assign(targets=[dummy_var], value=body.value)
        return parse_assignment(new_assign)
    elif isinstance(body, ast.Return):
        return None  # ignore  # TODO: Would like to actually use this
    elif isinstance(body, ast.Expr) and isinstance(body.value, (ast.Str, ast.Constant)):
        return None  # ignore  # TODO: Would like to actually use this
    else:
        return _error_handler(body)


# Note: generalize? glom?
def parse_assignment(body: ast.Assign, info=None) -> tuple:
    # TODO: Make this validation better (at least more help in raised error)
    # TODO: extract validating code out as validation functions?
    info = _ast_info_str(body)
    if not isinstance(body, (ast.Assign, ast.AnnAssign)):
        raise ValueError(f"All commands should be assignments, this one wasn't: {info}")

    if isinstance(body, ast.Assign):
        target = body.targets
    elif isinstance(body, ast.AnnAssign):
        target = [body.target]  # ast.AnnAssign has a single target, not a list

    assert len(target) == 1, f"Only one target allowed: {info}"

    target = target[0]
    assert isinstance(
        target, (ast.Name, ast.Tuple)
    ), f"Should be a ast.Name or ast.Tuple: {info}"

    value = body.value
    assert isinstance(value, ast.Call), (
        f"Only assigned function calls are allowed:" f" {info}"
    )

    return target, value


# TODO: Evolve this: Perhaps it can be used to centralize this concern:
def _extract_value_from_ast_element(ast_element):
    if isinstance(ast_element, ast.Name):
        return ast_element.id
    else:
        return ast_element.value


def parsed_to_node_kwargs(target_value) -> Iterator[dict]:
    """Extract FuncNode kwargs (name, out, and bind) from ast (target,value) pairs

    :param target_value: A (target, value) pair
    :return: A ``{name:..., out:..., bind:...}`` dict (meant to be used to curry FuncNode

    Where can you make make target_values? With the ``parse_assignment_steps`` function.

    >>> from meshed.makers import parse_assignment_steps
    >>> def foo():
    ...     x = func1(a, b=2)
    ...     y = func2(x, func1, c=3, d=x)
    >>> for target_value in parse_assignment_steps(foo):
    ...     for d in parsed_to_node_kwargs(target_value):
    ...         print(d)
    {'name': 'func1', 'out': 'x', 'bind': {0: 'a', 'b': 2}}
    {'name': 'func2', 'out': 'y', 'bind': {0: 'x', 1: 'func1', 'c': 3, 'd': 'x'}}

    """
    # Note: ast.Tuple has names in 'elts' attribute,
    # and could be handled, but would need to lead to multiple nodes
    target, value = target_value
    args = value.args
    bind_from_args = {i: k.id for i, k in enumerate(args)}
    kwargs = {x.arg: _extract_value_from_ast_element(x.value) for x in value.keywords}
    if isinstance(target, ast.Name):
        yield dict(
            name=value.func.id, out=target.id, bind=dict(bind_from_args, **kwargs)
        )
    elif isinstance(target, ast.Tuple):
        assign_to_names = tuple(map(attrgetter("id"), target.elts))
        # yield the function call information, assigning to a single variable
        # TODO: Long. Better way? (careful: need global uniqueness!)
        func_output_name = "__".join(assign_to_names)
        yield dict(
            name=value.func.id,
            out=func_output_name,
            bind=dict(bind_from_args, **kwargs),
        )
        # then, yield instructions to extract variable into several
        for i, assign_to_name in enumerate(assign_to_names):
            yield dict(
                func=signed_itemgetter(i),
                name=f"{assign_to_name}__{i}",
                out=assign_to_name,
                bind={0: func_output_name},
                func_label=f"[{i}]",
            )
        # raise ValueError(f"You're here: {target=}")
    else:
        raise TypeError(f"Should be a ast.Name or ast.Tuple. Was: {target}")


FuncNodeFactory = Callable[[Callable], FuncNode]


def node_kwargs_to_func_node_factory(node_kwargs) -> FuncNodeFactory:
    return partial(FuncNode, **node_kwargs)


def _ensure_src_string(src):
    if callable(src):
        src = inspect.getsource(src)
    return src


def _remove_indentation(src):
    m = re.match(r"\s+", src)
    if m is not None:
        indent = m.group(0)
        indent_length = len(indent)

        def gen():
            for line in src.split("\n"):
                if line.startswith(indent):
                    yield line[indent_length:]

        return "\n".join(gen())
    else:
        raise RuntimeError(f"I found no indent!")


def robust_ast_parse(src):
    try:
        return ast.parse(src)
    except IndentationError:
        return robust_ast_parse(_remove_indentation(src))


def parse_steps(src):
    """
    Parse source code and generate tuples of information about it.

    :param src: The source string or a python object whose code string can be extracted.
    :return: And generator of "target_values"

    >>> from meshed.makers import parse_steps
    >>> def foo():
    ...     x = func1(a, b=2)
    ...     y = func2(x, c=3)
    >>> target_values = list(parse_steps(foo))

    Let's look at the first target_value to see what it contains:

    >>> name, call = target_values[0]  # a 2-tuple
    >>> assert isinstance(name, ast.Name)  # the first element is a ast Name object
    >>> sorted(vars(name))
    ['col_offset', 'ctx', 'end_col_offset', 'end_lineno', 'id', 'lineno']
    >>> name.id
    'x'
    >>> assert isinstance(call, ast.Call)  # the first element is a ast Call object
    >>> sorted(vars(call))
    ['args', 'col_offset', 'end_col_offset', 'end_lineno', 'func', 'keywords', 'lineno']
    >>> call.args[0].id
    'a'
    >>> call.keywords[0].arg
    'b'
    >>> call.keywords[0].value.value
    2

    Basically, these ast objects contain all we need to know about the (parsed) source.

    """
    src = _ensure_src_string(src)
    root = robust_ast_parse(src)
    assert len(root.body) == 1
    func_body = root.body[0]
    # TODO: work with func_body.args to get info on interface (name, args, kwargs,
    #  return etc.)
    #     return func_body
    for body_index, body in enumerate(func_body.body):
        if (parsed_body := parse_body(body, body_index=body_index)) is not None:
            yield parsed_body


parse_assignment_steps = parse_steps  # backcompatible

iterize = lambda func: partial(map, func)


FuncNodeFactory = Callable[..., FuncNode]
FactoryToFunc = Callable[[FuncNodeFactory], Callable]


def src_to_func_node_factory(
    src, exclude_names=None
) -> Iterator[FuncNode | FuncNodeFactory]:
    """
    :param src: Callable or string of callable.
    :param exclude_names: Names to exclude when making func_nodes
    :return:
    """
    exclude_names = set(exclude_names or set())
    for i, target_value in enumerate(parse_assignment_steps(src), 1):
        for node_kwargs in parsed_to_node_kwargs(target_value):
            node_kwargs["func_label"] = node_kwargs["name"]
            if node_kwargs["name"] in exclude_names:
                # need to keep names uniques, so add a prefix to (hope) to get uniqueness
                node_kwargs["name"] += f"_{i:02.0f}"
            exclude_names.add(node_kwargs["name"])
            yield node_kwargs_to_func_node_factory(node_kwargs)


dlft_factory_to_func: FactoryToFunc


# TODO: A bit strange to ask a factory for information to get a func that it needs
#  to make itself. Do we gain much over simply saying "factory, make yourself"?
def dlft_factory_to_func(
    factory: partial,
    name_to_func_map: dict[str, Callable] | None = None,
    use_place_holder_fallback=True,
):
    """Get a function for the given factory, using"""
    # TODO: Add extra validation (like n_args of return func against bind)
    name_to_func_map = name_to_func_map or dict()

    factory_kwargs = factory.keywords
    name = (
        factory_kwargs["func_label"] or factory_kwargs["name"] or factory_kwargs["out"]
    )
    if name in name_to_func_map:
        return name_to_func_map[name]
    elif use_place_holder_fallback:
        arg_names = [
            k if isinstance(k, str) else v for k, v in factory_kwargs["bind"].items()
        ]
        return mk_place_holder_func(arg_names, name=name)
    else:
        raise KeyError(f"name not found in name_to_func_map: {name}")


def mk_fnodes_from_fn_factories(
    fnodes_factories: Iterable[FuncNodeFactory],
    factory_to_func: FactoryToFunc = dlft_factory_to_func,
) -> Iterator[FuncNode]:
    """Make func nodes from func node factories and a specification of how to make the
    nodes from these.

    :param fnodes_factories: An iterable of FuncNodeFactory
    :param factory_to_func: A function that will give you a function given a
        FuncNodeFactory input (where it will draw the information it needs to know
        what kind of function to make).
    :return:
    """
    # TODO: Might be a cleaner design for this...
    for fnode_factory in fnodes_factories:
        sig = Sig(fnode_factory)
        if sig.n_required == 1 and sig.names[0] == "func":
            # first making sure the fnode_factory is exactly as expected for this case,
            # get a function for this fnode_factory, then use it to make the fnode
            func = factory_to_func(fnode_factory)
            yield fnode_factory(func)
        elif sig.n_required == 0:
            # if fnode_factory has no (required) arguments, just call the factory:
            yield fnode_factory()
        else:
            # if couldn't figure it out from the last two cases, freak out!
            raise ValueError(
                f"The fnode_factory didn't have the expected format, so I'm freaking "
                f"out. It's supposed to be a no-arguments-required-callable or a "
                f"functools.partial that needs only a func to make the func node. "
                f"This is the offending fnode_factory: {fnode_factory}"
            )


class dlft_factory_to_func_mapping(Mapping):
    def __getitem__(self, item):
        return dlft_factory_to_func(item)


def _code_to_fnodes(src, func_src=dlft_factory_to_func):
    # Make all the funodes, but partial ones that don't have the func defined yet
    fnodes_factories = list(src_to_func_node_factory(src))
    # "Inject" the actual functions
    return mk_fnodes_from_fn_factories(fnodes_factories, func_src)


def _extract_name_from_single_func_def(src: str, default=None):
    t = robust_ast_parse(src)
    if (body := getattr(t, "body")) is not None:
        first_element = next(iter(body))
        if (
            isinstance(first_element, ast.FunctionDef)
            and (name := getattr(first_element, "name")) is not None
        ):
            return name
    return default


FuncSource = Union[Callable[[str], Callable], Mapping[str, Callable]]


# TODO: Make code_to_fnodes more flexible (not need to be enclosed in a function
#  definition)
@double_up_as_factory
def code_to_fnodes(
    src=None,
    *,
    func_src: FuncSource = dlft_factory_to_func,
    use_place_holder_fallback=False,
) -> tuple[FuncNode]:
    """Get func_nodes from src code"""
    func_src = _ensure_func_src(func_src, use_place_holder_fallback)
    # Pass on to _code_to_fnodes to get func nodes iterable needed to make DAG
    return tuple(_code_to_fnodes(src, func_src))


def _reconfigure_signature_according_to_src(src, dag):
    if callable(src):
        new_dag_sig = Sig(dag).modified(_allow_reordering=True, **Sig(src).parameters)
        return new_dag_sig(dag)
    return dag


@double_up_as_factory
def code_to_dag(
    src=None,
    *,
    func_src: FuncSource = dlft_factory_to_func,
    use_place_holder_fallback=False,
    name: str = None,
) -> DAG:
    """Get a ``meshed.DAG`` from src code

    This function parses Python code and creates a DAG that represents the
    computational flow. The inverse operation is available through ``dag_to_code``
    which can convert a DAG back to executable Python code.

    See also: ``dag_to_code`` for the inverse operation.
    """
    fnodes = code_to_fnodes(
        src, func_src=func_src, use_place_holder_fallback=use_place_holder_fallback
    )
    dag = DAG(fnodes, name=_ensure_name(name, src))
    dag._code_to_dag_src = src
    return _reconfigure_signature_according_to_src(src, dag)


def code_to_digraph(src):
    return code_to_dag(src).dot_digraph()


simple_code_to_digraph = code_to_digraph  # back-compatability alias

# import re
# from typing import Tuple, Iterable, Iterator
# from meshed import FuncNode, code_to_dag, code_to_fnodes, DAG

extract_tokens = re.compile(r"\w+").findall


def triples_to_fnodes(triples: Iterable[tuple[str, str, str]]) -> Iterable[FuncNode]:
    """Converts an iterable of func call triples to an iterable of ``FuncNode``s.
    (Which in turn can be converted to a ``DAG``.)

    Note how the python identifiers are extracted (on the basis of "an unbroken
    sequence of alphanumerical (and underscore) characters", ignoring all other
    characters).

    >>> from meshed import DAG
    >>> dag = DAG(
    ...     triples_to_fnodes(
    ...     [
    ...         ('alpha bravo', 'charlie', 'delta echo'),
    ...         (' foxtrot  &^$#', 'golf', '  alpha,  echo'),
    ...     ])
    ... )
    >>> print(dag.synopsis_string())
    delta,echo -> charlie -> alpha__bravo
    alpha__bravo -> alpha__0 -> alpha
    alpha__bravo -> bravo__1 -> bravo
    alpha,echo -> golf -> foxtrot
    """
    code = "\n\t".join(_triple_to_func_call_str(*triple) for triple in triples)
    code = f"def main():\n\t{code}"
    return code_to_fnodes(code)


def _triple_to_func_call_str(
    outputs: str | Iterable[str],
    func_name: str,
    inputs: str | Iterable[str],
) -> str:
    """Converts a `(outputs, func_name, inputs)` triple to a function call string.

    >>> _triple_to_func_call_str(('a', 'b'), 'func', ('c', 'd'))
    'a, b = func(c, d)'
    """
    outputs = _ensure_tokens_iterable(outputs)
    inputs = _ensure_tokens_iterable(inputs)
    return f"{', '.join(outputs)} = {func_name}({', '.join(inputs)})"


def _ensure_tokens_iterable(tokens):
    if isinstance(tokens, str):
        return extract_tokens(tokens)
    else:
        return tokens


def _ensure_func_src(
    func_src: FuncSource, use_place_holder_fallback=False
) -> Callable[[str], Callable]:
    if isinstance(func_src, Mapping):
        name_to_func_map = func_src
        func_src = partial(
            dlft_factory_to_func,
            name_to_func_map=name_to_func_map,
            use_place_holder_fallback=use_place_holder_fallback,
        )
    assert isinstance(func_src, Callable), f"func_src should be callable, or a mapping"
    return func_src


def _ensure_name(name, src):
    if name is None:
        if isinstance(src, str):
            name = _extract_name_from_single_func_def(src, "dag_made_from_code_parsing")
        else:
            name = name_of_obj(src)
    return name


# SB stuff, not used, so comment-out deprecating
# class AssignNodeVisitor(ast.NodeVisitor):
#     def __init__(self):
#         self.store = []
#
#     def visit_Assign(self, node):
#         self.store.append(parse_assignment(node))
#         return node
#
#
# def retrieve_assignments(src):
#     if callable(src):
#         src = inspect.getsource(src)
#     nodes = ast.parse(src)
#     visitor = AssignNodeVisitor()
#     visitor.visit(nodes)
#
#     return visitor.store


def lined_dag(funcs):
    dag = DAG(funcs)
    if not funcs:
        return dag
    else:
        names = [name_of_obj(func) for func in funcs]
        pairs = zip(names, names[1:])
        dag = dag.add_edges(pairs)
        return dag


from collections.abc import Callable, Mapping, Iterable

NamedFuncs = Mapping[str, Callable]


def named_funcs_to_func_nodes(named_funcs: NamedFuncs) -> Iterable[FuncNode]:
    """Make ``FuncNode``s from keyword arguments, using the key as the ``.out`` of the
    ``FuncNode`` and the value as the ``.func`` of the ``FuncNode``.

    Example use: To get from ``Slabs`` to ``DAG``.

    >>> from meshed import DAG
    >>> func_nodes = list(named_funcs_to_func_nodes(dict(
    ...     a=lambda x: x + 1,
    ...     b=lambda a: a + 2,
    ...     c=lambda a, b: a * b)
    ... ))
    >>> dag = DAG(func_nodes)
    >>> dag(x=3)
    24

    The inverse of this function is ``func_nodes_to_named_funcs``.

    >>> named_funcs = func_nodes_to_named_funcs(dag.func_nodes)
    >>> dag2 = DAG(named_funcs_to_func_nodes(named_funcs))
    >>> assert dag2(x=3) == dag(x=3) == 24

    """
    return (
        FuncNode(func, name=f"{out}_", out=out) for out, func in named_funcs.items()
    )


def func_nodes_to_named_funcs(func_nodes: Iterable[FuncNode]) -> NamedFuncs:
    """Make some components (kwargs) based on the ``.out`` and ``.func`` of the
    ``FuncNode``s.

    Example use: To get from ``DAG`` to ``Slabs``.

    >>> from meshed import DAG, FuncNode
    >>> dag = DAG([
    ...     FuncNode(lambda x: x + 1, out='a'),
    ...     FuncNode(lambda a: a + 2, out='b',),
    ...     FuncNode(lambda a, b: a * b, out='c'),
    ... ])
    >>> dag(x=10)
    143
    >>> named_funcs = func_nodes_to_named_funcs(dag.func_nodes)
    >>> isinstance(named_funcs, dict)
    True
    >>> list(named_funcs)
    ['a', 'b', 'c']
    >>> callable(named_funcs['a'])
    True
    >>> assert dag.find_func_node('a').func(3) == named_funcs['a'](3) ==4

    The inverse of this function is ``named_funcs_to_func_nodes``.

    >>> func_nodes = list(named_funcs_to_func_nodes(named_funcs))
    >>> dag2 = DAG(func_nodes)
    >>> assert dag2(x=3) == dag(x=3) == 24


    """
    return {node.out: node.func for node in func_nodes}


# --------------------------------------------------------------------------------------
"""
This section contains some ideas around making a two-way interaction between meshed 
and a GUI that will enable the construction of meshes as well as rendering them, 
and possibly running them.
"""

# from typing import Callable
# from meshed.dag import DAG, FuncNode
# from meshed.util import mk_place_holder_func
# from functools import partial

# TODO: Add default func_to_jdict and dict_to_func that uses mk_place_holder_func to
#  jdict will be only signature (jdict) and deserializing it will be just placeholder
Jdict = dict  # json-serializable dictionary


def fnode_to_jdict(
    fnode: FuncNode, *, func_to_jdict: Callable[[Callable], Jdict] = None
):
    jdict = {
        "name": fnode.name,
        "func_label": fnode.func_label,
        "bind": fnode.bind,
        "out": fnode.out,
    }
    if func_to_jdict is not None:
        jdict["func"] = func_to_jdict(fnode.func)
    return jdict


def jdict_to_fnode(jdict: dict, *, jdict_to_func: Callable[[Jdict], Callable] = None):
    if jdict_to_func is not None:
        return FuncNode(
            func=jdict_to_func(jdict["func"]),
            name=jdict["name"],
            func_label=jdict["func_label"],
            bind=jdict["bind"],
            out=jdict["out"],
        )
    else:
        raise NotImplementedError("Need a function")


def dag_to_jdict(dag: DAG, *, func_to_jdict: Callable = None):
    """
    Will produce a json-serializable dictionary from a dag.
    """
    fnode_to_jdict_ = partial(fnode_to_jdict, func_to_jdict=func_to_jdict)
    return {
        "name": dag.name,
        "func_nodes": list(map(fnode_to_jdict_, dag.func_nodes)),
    }


def jdict_to_dag(jdict: dict, *, jdict_to_func: Callable = None):
    """
    Will produce a dag from a json-serializable dictionary.
    """
    jdict_to_fnode_ = partial(jdict_to_fnode, jdict_to_func=jdict_to_func)
    return DAG(
        name=jdict["name"],
        func_nodes=list(map(jdict_to_fnode_, jdict["func_nodes"])),
    )
```

## scrap/__init__.py

```python
"""For scrap only"""
```

## scrap/annotations_to_meshes.py

```python
"""
Code related to work on the "From annotated functions to meshes" discussion:

https://github.com/i2mint/meshed/discussions/55


"""

import typing
from typing import Dict, Protocol, TypeVar, Any
from collections.abc import Callable
from collections.abc import Callable as CallableGenericAlias
from inspect import signature, Signature, Parameter
from functools import wraps

import builtins
import re

_camel_pattern = re.compile(r"(?<!^)(?=[A-Z])")
PK = Parameter.POSITIONAL_OR_KEYWORD


def _camel_to_snake(x):
    return _camel_pattern.sub("_", x)


_is_lower = lambda x: x == x.lower()
_builtin_lower_names = set(map(_camel_to_snake, filter(_is_lower, dir(builtins))))


Annotation, ArgPosition, MethodName, Argname = Any, int, str, str
MkArgname = Callable[[Annotation, ArgPosition, MethodName], Argname]


def try_annotation_name(
    arg_annotation: Annotation, arg_position: ArgPosition, method_name: MethodName
) -> Argname:
    argname = getattr(arg_annotation, "__name__", None)
    if argname is None or argname in _builtin_lower_names:
        argname = f"arg_{arg_position:02.0f}"
    return argname.lower()


def _is_callable_type_annot(x):
    return isinstance(x, CallableGenericAlias)


def callable_annots_to_signature(
    callable_annots: CallableGenericAlias, mk_argname: MkArgname = try_annotation_name
) -> Signature:
    """Produces a signature from a Callable type annotation

    >>> from typing import Callable, NewType
    >>> MyType = NewType('MyType', str)
    >>> sig = callable_annots_to_signature(Callable[[MyType, str], str])
    >>> import inspect
    >>> isinstance(sig, inspect.Signature)
    True
    >>> list(sig.parameters.keys())
    ['self', 'mytype', 'arg_01']
    >>> sig.parameters['arg_01'].annotation
    <class 'str'>

    """
    origin = typing.get_origin(callable_annots)
    if not _is_callable_type_annot(origin):
        raise ValueError("The provided type is not a Callable generic alias")

    input_annots, return_annot = typing.get_args(callable_annots)
    arg_params = [
        Parameter(mk_argname(annot, i, None), PK, annotation=annot)
        for i, annot in enumerate(input_annots)
    ]
    self_param = Parameter("self", PK)
    return Signature([self_param] + arg_params, return_annotation=return_annot)


def func_types_to_protocol(
    func_types: dict[str, CallableGenericAlias],
    name: str = None,
    *,
    mk_argname: MkArgname = try_annotation_name,
) -> typing.Protocol:
    """Produces a typing.Protocol based on a dictionary of
    `(method_name, Callable_type)` specification"""

    class TempProtocol(Protocol):
        pass

    if name:
        TempProtocol.__name__ = name

    for method_name, callable_type in func_types.items():
        sig = callable_annots_to_signature(callable_type, mk_argname)

        def method_stub(*args, **kwargs):
            pass

        method_stub.__signature__ = sig
        setattr(TempProtocol, method_name, method_stub)

    return TempProtocol


def test_func_types_to_protocol():
    from typing import Any, NewType
    from collections.abc import Iterable, Callable

    Group = NewType("Group", str)
    Item = NewType("Item", Any)

    class Groups:
        add_item_to_group: Callable[[Item, Group], Any]
        add_items_to_group: Callable[[Iterable[Item], Group], Any]
        list_groups: Callable[[], Iterable[Group]]
        items_for_group: Callable[[Group], Iterable[Item]]

    class ExpectedGroupsProtocol(Protocol):
        def add_items_to_group(self, item: Item, group: Group) -> Any:
            pass

        def add_items_to_group(self, items: Iterable[Item], group: Group) -> Any:
            pass

        def list_groups(self) -> Iterable[Group]:
            pass

        def items_for_group(self, group: Group) -> Iterable[Item]:
            pass

    ActualGroupsProtocol = func_types_to_protocol(Groups.__annotations__)
    # TODO: assert that ActualGroupsProtocol and ExpectedGroupsProtocol are equal in
    #  some way (but == doesn't work, and is not meant to!)


def func_types_to_scaffold(
    func_types: dict[str, CallableGenericAlias], name: str = None
) -> str:
    """Produces a scaffold class containing the said methods, with given annotations"""

    if name is None:
        name = "GeneratedClass"

    methods = []
    for method_name, callable_type in func_types.items():
        sig = callable_annots_to_signature(callable_type)
        arg_str = ", ".join(
            (
                f"{param.name}: {param.annotation.__name__}"
                if param.annotation != Parameter.empty
                else f"{param.name}"
            )
            for param in sig.parameters.values()
        )
        return_annotation = (
            sig.return_annotation.__name__
            if sig.return_annotation != Parameter.empty
            else "None"
        )
        method_str = (
            f"def {method_name}({arg_str}) -> {return_annotation}:\n    \tpass\n"
        )
        methods.append(method_str)

    class_str = f"\nclass {name}:\n    " + "\n    ".join(methods)
    return class_str


_expected_scaffold = """
class GeneratedClass:
    def add_item_to_group(self, item: Item, group: Group) -> Any:
    	pass

    def add_items_to_group(self, iterable: Iterable, group: Group) -> Any:
    	pass

    def list_groups(self) -> Iterable:
    	pass

    def items_for_group(self, group: Group) -> Iterable:
    	pass
"""


def test_func_types_to_scaffold():
    from typing import Any, NewType
    from collections.abc import Iterable, Callable

    Group = NewType("Group", str)
    Item = NewType("Item", Any)

    class Groups:
        add_item_to_group: Callable[[Item, Group], Any]
        add_items_to_group: Callable[[Iterable[Item], Group], Any]
        list_groups: Callable[[], Iterable[Group]]
        items_for_group: Callable[[Group], Iterable[Item]]

    actual_scaffold = func_types_to_scaffold(Groups.__annotations__)
    assert actual_scaffold == _expected_scaffold
```

## scrap/cached_dag.py

```python
from collections.abc import Mapping
from functools import partial
from collections import ChainMap
from meshed.itools import edge_reversed_graph, descendants
from i2 import Sig, Param, sort_params


class NotAllowed(Exception):
    """To use to indicate that something is not allowed"""


class OverWritesNotAllowedError(NotAllowed):
    """Error to raise when a writes to existing keys are not allowed"""


def get_first_item_and_assert_unicity(seq):
    seq_length = len(seq)
    if seq_length:
        assert seq_length == 1, (
            f"There should be one and one only item in the " f"sequence: {seq}"
        )
        return seq[0]
    else:
        return None


def func_node_names_and_outs(dag):
    for func_node in dag.func_nodes:
        yield func_node.name, func_node.out


class NoOverwritesDict(dict):
    """
    A dict where you're not allowed to write to a key that already has a value in it.

    >>> d = NoOverwritesDict(a=1, b=2)
    >>> d
    {'a': 1, 'b': 2}

    Writing is allowed, in new keys

    >>> d['c'] = 3
    >>> d
    {'a': 1, 'b': 2, 'c': 3}

    It's also okay to write into an existing key if the value it holds is identical.
    In fact, the write doesn't even happen.

    >>> d['b'] = 2

    But if we try to write a different value...

    >>> d['b'] = 22  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    cached_dag.OverWritesNotAllowedError: The b key already exists and you're not allowed to change its value

    """

    def __setitem__(self, key, value):
        if key not in self:
            super().__setitem__(key, value)
        elif value != self[key]:
            raise OverWritesNotAllowedError(
                f"The {key} key already exists and you're not allowed to change its "
                f"value"
            )
        # else, don't even write the value since it's the same


NoSuchKey = type("NoSuchKey", (), {})


# TODO: Cache validation and invalidation
# TODO: Continue constructing uppward towards lazyprop-using class (instances are
#  varnodes)
class CachedDag:
    """
    Wraps a DAG, using it to compute any of it's var nodes from it's dependents,
    with the capability of caching intermediate var nodes for later reuse.

    >>> def add(a, b=1):
    ...     return a + b
    >>> def mult(x, y=2):
    ...     return x * y
    >>> def subtract(a, b=4):
    ...     return a - b
    >>> from meshed import code_to_dag
    >>>
    >>> @code_to_dag(func_src=locals())
    ... def dag(w, ww, www):
    ...     x = mult(w, ww)
    ...     y = add(x, www)
    ...     z = subtract(x, y)
    >>> print(dag.dot_digraph_ascii())  # doctest: +SKIP

    .. code-block::
                        w

                     │
                     │
                     ▼
                   ┌──────────┐
         ww=   ──▶ │   mult   │
                   └──────────┘
                     │
                     │
                     ▼

                        x       ─┐
                                 │
                     │           │
                     │           │
                     ▼           │
                   ┌──────────┐  │
         www=  ──▶ │   add    │  │
                   └──────────┘  │
                     │           │
                     │           │
                     ▼           │
                                 │
                        y=       │
                                 │
                     │           │
                     │           │
                     ▼           │
                   ┌──────────┐  │
                   │ subtract │ ◀┘
                   └──────────┘
                     │
                     │
                     ▼

                        z

    >>> from inspect import signature
    >>> g = CachedDag(dag)
    >>> signature(g)
    <Signature (k, /, **input_kwargs)>


    We can get ``ww`` because it has a default:

    (TODO: This (and further tests) stopped working since code_to_dag was enhanced
    with the ability to use the wrapped function's signature to determine the
    signature of the output dag. Need to fix this.)

    >>> g('ww')
    2

    But we can't get ``y`` because we don't have what it depends on:

    >>> g('y')
    Traceback (most recent call last):
        ...
    TypeError: The input_kwargs of a dag call is missing 1 required argument: 'w'

    It needs a ``w?``! No, it needs an ``x``! But to get an ``x`` you need a ``w``,
    and...

    >>> g('x')
    Traceback (most recent call last):
        ...
    TypeError: The input_kwargs of a dag call is missing 1 required argument: 'w'

    So let's give it a w!

    >>> g('x', w=3)  # == 3 * 2 ==
    6

    And now this works:

    >>> g('x')
    6

    because

    >>> g.cache
    {'x': 6}

    and this will work too:

    >>> g('y')
    7
    >>> g.cache
    {'x': 6, 'y': 7}

    But this is something we need to handle better!

    >>> g('x', w=10)
    6

    This is happending because there's already a x in the cache, and it takes precedence.
    This would be okay if consider CachedDag as a low level object that is never
    actually used by a user.
    But we need to protect the user from such effects!

    First, we probably should cache inputs too.

    The we can:
    - Make  computation take precedence over cache, overwriting the existing cache
        with the new resulting values

    - Allow the user to declare the entire cache, or just some variables in it,
    as write-once, to avoid creating bugs with the above proposal.

    - Cache multiple paths (lru_cache style) for different input combinations

    """

    def __init__(self, dag, cache=True, name=None):
        self.dag = dag
        self.reversed_graph = edge_reversed_graph(dag.graph_ids)
        self.roots = set(self.dag.roots)
        self.leafs = set(self.dag.leafs)
        self.var_nodes = set(self.dag.var_nodes)
        self.func_node_of_id = {fn.out: fn for fn in self.dag.func_nodes}
        self.name = name
        self.out_of_func_node_name = dict(func_node_names_and_outs(self.dag))
        self._dag_sig = Sig(self.dag)
        self.defaults = self._dag_sig.defaults
        if cache is True:
            self.cache = NoOverwritesDict()
        elif not isinstance(cache, Mapping):
            raise NotImplementedError(
                "This type of cache is not implemented (must resolve to a Mapping): "
                f"{cache=}"
            )
        self._cache = ChainMap(self.defaults, self.cache)

    @property
    def __name__(self):
        return self.name or self.dag.__name__

    def __iter__(self):
        yield from self.reversed_graph

    def func_node_id(self, k):
        func_node_name = get_first_item_and_assert_unicity(self.reversed_graph[k])
        if func_node_name is not None:
            return self.out_of_func_node_name[func_node_name]

    # TODO: Consider having args and kwargs instead of just input_kwargs.
    #   or making it (k, /, *args, **kwargs)
    def __call__(self, k, /, **input_kwargs):
        #         print(f"Calling ({k=},{input_kwargs=})\t{self.cache=}")
        input_kwargs = dict(input_kwargs)
        if intersection := (input_kwargs.keys() & self.cache.keys()):
            # TODO: Can give the user a more informative/correct message, since the
            #  user has more options than just the root nodes: They some combination of
            #  intermediates would also satisfy requirements.
            raise ValueError(
                f"input_kwargs can't contain any keys that are already in cache! "
                f"These names were in both: {intersection}"
            )
        _cache = ChainMap(input_kwargs, self._cache)
        if k in _cache:
            return _cache[k]
        input_kwargs = dict(input_kwargs)
        func_node_id = self.func_node_id(k)
        #         print(f"{func_node_id=}")
        if func_node_id:
            if (output := self.cache.get(func_node_id)) is not None:
                return output
            else:
                func_node = self.func_node_of_id[func_node_id]
                input_sources = {
                    src: self(src, **input_kwargs) for src in func_node.bind.values()
                }
                #                 inputs = dict(input_sources, **input_kwargs)  #
                # TODO: do we need to include **self.defaults in the middle?
                inputs = ChainMap(_cache, input_sources)
                #                 print(f"Computing {func_node_id}: ", end=" ")
                output = func_node.call_on_scope(inputs, write_output_into_scope=False)
                self.cache[func_node_id] = output
                #                 print(f"result -> {output}")
                return output
        else:  # k is a root node
            assert k in self.roots, f"Was expecting this to be a root node: {k}"
            inputs = ChainMap(input_kwargs, self._cache)
            if (output := inputs.get(k, NoSuchKey)) is not NoSuchKey:
                return output
            else:
                raise TypeError(
                    f"The input_kwargs of a {self.__name__} call is missing 1 required "
                    f"argument: '{k}'"
                )

    def _call(self, k, /, **kwargs):
        return self(k, **kwargs)

    def roots_for(self, node):
        """
        The set of roots that lead to ``node``.

        >>> from meshed.makers import code_to_dag
        >>> @code_to_dag
        ... def dag():
        ...     x = mult(w, ww)
        ...     y = add(x, www)
        ...     z = subtract(x, y)
        >>> print(dag.synopsis_string())
        w,ww -> mult -> x
        x,www -> add -> y
        x,y -> subtract -> z
        >>> g = CachedDag(dag)
        >>> sorted(g.roots_for('x'))
        ['w', 'ww']
        >>> sorted(g.roots_for('y'))
        ['w', 'ww', 'www']
        """
        return set(
            filter(self.roots.__contains__, descendants(self.reversed_graph, node))
        )

    def _signature_for_node_method(self, node):
        def gen():
            for name in filter(lambda x: x not in self.cache, self.roots_for(node)):
                yield Param(
                    name=name,
                    kind=Param.KEYWORD_ONLY,
                    default=self.defaults.get(name, Param.empty),
                    annotation=self._dag_sig.annotations.get(name, Param.empty),
                )

        return Sig(sort_params(gen()))

    def inject_methods(self, obj=None):
        # TODO: Should be input_names of reversed_graph, but resulting "shadow" in
        #  the root nodes, along with their defaults (filtered by cache)
        non_root_var_nodes = list(filter(lambda x: x not in self.roots, self.var_nodes))
        if obj is None:
            from types import SimpleNamespace

            obj = SimpleNamespace(**{k: None for k in non_root_var_nodes})
        for var_node in non_root_var_nodes:
            sig = self._signature_for_node_method(var_node)
            f = sig(partial(self._call, var_node))
            setattr(obj, var_node, f)

        obj._cache = self.cache
        return obj


def cached_dag_test():
    """
    Covering issue https://github.com/i2mint/meshed/issues/34
    about "CachedDag.cache should be populated with inputs that it was called on"
    """
    from meshed.dag import DAG

    def f(a, x=1):
        return a + x

    def g(a, y=2):
        return a * y

    dag = DAG([f, g])

    c = CachedDag(dag)
    c("g", a=1)
    assert c.cache == {"g": 2, "a": 1}
    assert c("f" == 2)


def add(a, b=1):
    return a + b


def mult(x, y=2):
    return x * y


def exp(mult, n=3):
    return mult**n


def subtract(a, b=4):
    return a - b


# from meshed import code_to_dag
#
#
# @code_to_dag(func_src=locals())
# def dag(w, ww, www):
#     x = mult(w, ww)
#     y = add(x, www)
#     z = subtract(x, y)
#
#
# g = CachedDag(dag)
#
# assert g('z', {'w': 2, 'ww': 3, 'www': 4}) == -4 == dag(2, 3, 4)
```

## scrap/collapse_and_expand.py

```python
"""Ideas on collapsing and expanding nodes
See "Collapse and expand nodes" discussion:
https://github.com/i2mint/meshed/discussions/54

"""

import re
from typing import Union, Optional
from collections.abc import Iterable, Callable
from meshed.dag import DAG
from meshed.makers import code_to_dag


def remove_decorator_code(
    src: str, decorator_names: str | Iterable[str] | None = None
) -> str:
    """
    Remove the code corresponding to decorators from a source code string.
    If decorator_names is None, will remove all decorators.
    If decorator_names is an iterable of strings, will remove the decorators with those names.

    Examples:
    >>> src = '''
    ... @decorator
    ... def func():
    ...     pass
    ... '''
    >>> print(remove_decorator_code(src))
    def func():
        pass

    >>> src = '''
    ... @decorator1
    ... @decorator2
    ... def func():
    ...     pass
    ... '''
    >>> print(remove_decorator_code(src, "decorator1"))
    @decorator2
    def func():
        pass
    """
    import ast

    if isinstance(decorator_names, str):
        decorator_names = [decorator_names]

    class DecoratorRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.decorator_list:
                if decorator_names is None:
                    node.decorator_list = []  # Remove all decorators
                else:
                    node.decorator_list = [
                        d
                        for d in node.decorator_list
                        if not (isinstance(d, ast.Name) and d.id in decorator_names)
                    ]
            return node

        def visit_ClassDef(self, node):
            if node.decorator_list:
                if decorator_names is None:
                    node.decorator_list = []  # Remove all decorators
                else:
                    node.decorator_list = [
                        d
                        for d in node.decorator_list
                        if not (isinstance(d, ast.Name) and d.id in decorator_names)
                    ]
            return node

    tree = ast.parse(src)
    new_tree = DecoratorRemover().visit(tree)
    ast.fix_missing_locations(new_tree)

    return ast.unparse(new_tree)


def get_src_string(src: str | DAG) -> str:
    if isinstance(src, str):
        return src
    elif hasattr(src, "_code_to_dag_src"):
        return get_src_string(src._code_to_dag_src)
    elif callable(src):
        import inspect

        return inspect.getsource(src)
    else:
        raise ValueError(
            f"src should be a string or have a _code_to_dag_src (meaning the src was "
            f"made with code_to_dag), not a {type(src)}"
        )


# TODO: Generalize to src that is any DAG
def collapse_function_calls(
    src: str | DAG,
    call_func_name="call",
    *,
    rm_decorator="code_to_dag",
    include: Iterable[str] | Callable[[str], bool] | None = None,
):
    """
    Contract function calls in a source code string.

    That is, in source code, or a dag made from code_to_dag, replace calls of the form
    `call(func, arg)` with `func(arg)`.

    Note: Doesn't work with arbitrary DAG src, only those made from code_to_dag.
    """
    src_string = get_src_string(src)

    def should_include(func_name):
        if include is None:
            return True
        if isinstance(include, Iterable):
            return func_name in include
        if callable(include):
            return include(func_name)
        return False

    pattern = call_func_name + r"\(([^,]+),\s*([^)]+)\)"

    def replace(match):
        func_name, args = match.groups()
        if should_include(func_name):
            return f"{func_name}({args})"
        return match.group(0)

    new_src = re.sub(pattern, replace, src_string)

    if rm_decorator:
        new_src = remove_decorator_code(new_src, decorator_names=rm_decorator)

    return new_src if isinstance(src, str) else code_to_dag(new_src)


def expand_function_calls(
    src: Union[str, "DAG"],
    call_func_name="call",
    *,
    include: Iterable[str] | Callable[[str], bool] | None = None,
) -> str:
    """
    Inverse of collapse_function_calls.
    It replaces calls of the form `func(arg)` with `call(func, arg)`,
    except when the function call is part of a function definition header.
    If include is None, it expands all function calls.
    If include is a list of function names, only those functions are expanded.
    If include is a callable, it's used as a filter function.
    """
    src_string = get_src_string(src)

    def should_include(func_name):
        if include is None:
            return True
        if isinstance(include, Iterable):
            return func_name in include
        if callable(include):
            return include(func_name)
        return False

    pattern = r"(\b[a-zA-Z_]\w*)\(([^)]*)\)"

    def replace(match):
        # Get the start index of the match
        index = match.start()
        # Find the beginning of the current line
        line_start = src_string.rfind("\n", 0, index) + 1
        # Extract the text from the start of the line up to the match
        current_line = src_string[line_start:index]
        # If the current line starts with a function definition, skip expansion.
        if re.match(r"^\s*def\s", current_line):
            return match.group(0)
        func_name, args = match.groups()
        if should_include(func_name):
            return f"{call_func_name}({func_name}, {args})"
        return match.group(0)

    new_src = re.sub(pattern, replace, src_string)

    return new_src if isinstance(src, str) else code_to_dag(new_src)


# ------------------------------------------------------------------------------
# Older code

from dataclasses import dataclass
from i2 import Sig
from meshed.dag import DAG


@dataclass
class CollapsedDAG:
    """To collapse a DAG into a single function

    This is useful for when you want to use a DAG as a function,
    but you don't want to see all the arguments.

    """

    dag: DAG

    def __post_init__(self):
        Sig(self.dag)(self)  # so that __call__ gets dag's signature
        self.__name__ = self.dag.name

    def __call__(self, *args, **kwargs):
        return self.dag(*args, **kwargs)

    def expand(self):
        return self.dag


# TODO: Finish this
def expand_nodes(
    dag,
    nodes=None,
    *,
    is_node=lambda fnode, node: fnode.name == node or fnode.out == node,
    expansion_record_store=None,  # TODO: Implement this to keep track of what was expanded
):
    if nodes is None:
        nodes = ...  # find all func_nodes that have isinstance(fn.func, CollapsedDAG)

    def change_node_or_not(node):
        if is_node(node):
            return CollapsedDAG(node.func)
        else:
            return node

    return DAG(list(map(change_node_or_not, dag.func_nodes)))
```

## scrap/conversion.py

```python
"""Utils to convert graphs from one specification to another"""

import os

DFLT_PROG = "neato"
graph_template = 'strict graph "" {{\n{dot_str}\n}}'
digraph_template = 'strict digraph "" {{\n{dot_str}\n}}'


def ensure_dot_code(x: str):
    if not x.startswith("strict"):
        if "--" in x:
            print("asdfdf")
            x = graph_template.format(dot_str=x)
        else:  # if '->' in dot_str
            x = digraph_template.format(dot_str=x)
    return x


def dot_to_nx(dot_src):
    from pygraphviz import AGraph

    dot_src = ensure_dot_code(dot_src)
    return AGraph(string=dot_src)


def dot_to_ipython_image(dot_src, *, prog=DFLT_PROG, tmp_file="__tmp_file.png"):
    from IPython.display import Image

    dot_src = ensure_dot_code(dot_src)
    g = dot_to_nx(dot_src)
    g.draw(tmp_file, prog=prog)
    ipython_obj = Image(tmp_file)

    if os.path.isfile(tmp_file):
        os.remove(tmp_file)

    return ipython_obj


def dot_to_pydot(dot_src):
    import pydot

    dot_src = ensure_dot_code(dot_src)
    return pydot.graph_from_dot_data(dot_src)
```

## scrap/dask_graph_language.py

```python
"""How to make dags from the dask specification

See https://docs.dask.org/en/latest/graphs.html#example for the specification
"""

from meshed import FuncNode, DAG
from i2 import Sig


def node_funcs_from_dask_graph_dict(dask_graph_dict):
    for out_key, val in dask_graph_dict.items():
        if isinstance(val, tuple):
            func, *args = val
            bind_args = dict(zip(Sig(func).names, args))
            yield FuncNode(func=func, bind=bind_args, out=out_key)


def inc(i):
    return i + 1


def add(a, b):
    return a + b


d = {"y": (inc, "x"), "z": (add, "y", "a")}

dag = DAG(node_funcs_from_dask_graph_dict(d))

from contextlib import suppress

with suppress(ModuleNotFoundError, ImportError):
    dag.dot_digraph()


# For being able to handle non-tuple vals (like the 1 of 'x':1 and non string args
# (like the 10)), need more work.
# Can solve ambiguity between string INPUT and string denoting scope argument name with
# a AsString literal class
# d = {'x': 1,
#      'y': (inc, 'x'),
#      'z': (add, 'y', 10)}
```

## scrap/flow_control_script.py

```python
from meshed import DAG
from creek.automatas import BasicAutomata, mapping_to_transition_func
from typing import Any, Literal
from collections.abc import Callable, MutableMapping, Mapping
from dataclasses import dataclass
from i2 import ch_names


Case = Any
Cases = Mapping[Case, Callable]


RecordingCommands = Literal["start", "resume", "stop"]


def mk_test_objects():
    # from slang import fixed_step_chunker

    audio = range(100)
    audio_chk_size = 5
    # audio_chks = list(fixed_step_chunker(audio, chk_size=audio_chk_size))
    audio_chks = [
        audio[i : i + audio_chk_size] for i in range(0, len(audio), audio_chk_size)
    ]
    plc_values = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0]

    return audio_chks, plc_values


@dataclass
class RecordingSwitchBoard:
    store: MutableMapping = None
    _current_key = None

    def start(self, key, chk):
        self._current_key = key
        self.store[key] = []
        self._append(chk)

    def resume(self, key, chk):
        print(f"resume called")
        self._append(chk)

    def stop(self, key, chk):
        self._append(chk)
        self._current_key = None

    def _append(self, chk):
        if self._current_key is None:
            raise ValueError("Cannot append without first starting recording.")
        self.store[self._current_key].extend(chk)

    @property
    def is_recording(self):
        return self._current_key is not None


@dataclass
class SimpleSwitchCase:
    """A functional implementation of thw switch-case control flow.
    Makes a callable that takes two arguments, a case and an input.

    >>> f = SimpleSwitchCase({'plus_one': lambda x: x + 1, 'times_two': lambda x: x * 2})
    >>> f('plus_one', 2)
    3
    >>> f('times_two', 2)
    4
    """

    cases: Mapping[Case, Callable]

    def __call__(self, case, input):
        func = self.cases.get(case, None)
        if func is None:
            raise ValueError(f"Case {case} not found.")
        return func(input)


def mk_simple_switch_case(
    cases: Cases, *, name: str = None, case_name: str = None, input_name: str = None
):
    """
    Makes a simple switch-case function, with optional naming control.
    """
    switch_case_func = SimpleSwitchCase(cases)
    switch_case_func = ch_names(
        switch_case_func, **dict(case=case_name, input=input_name)
    )
    if name is not None:
        switch_case_func.__name__ = name
    return switch_case_func


def mk_recorder_switch(
    store, *, mk_recorder: Callable[[MutableMapping], Any] = RecordingSwitchBoard
):
    recorder = mk_recorder(store)
    return mk_simple_switch_case(
        {
            "start": lambda key_and_chk: recorder.start(*key_and_chk),
            "resume": lambda key_and_chk: recorder.resume(*key_and_chk),
            "stop": lambda key_and_chk: recorder.stop(*key_and_chk),
            "waiting": lambda x: None,
        },
        name="recorder_switch",
        case_name="state",
        input_name="key_and_chk",
    )


def mk_transition_func(
    trans_func_mapping,
    initial_state,  # symbol_var_name: str,
):
    recording_state_transition_func = mapping_to_transition_func(
        trans_func_mapping,
        strict=False,
    )
    transitioner = BasicAutomata(
        transition_func=recording_state_transition_func,
        state=initial_state,
    )

    # @i2.ch_names(symbol=symbol_var_name)
    def transition(symbol):
        return transitioner.transition(symbol)

    # transition = transitioner.reset().transition

    return transition


# store = mk_recorder_switch(store)
trans_func_mapping = {
    ("waiting", 1): "start",
    ("start", 0): "resume",
    ("start", 1): "stop",
    ("resume", 1): "stop",
    ("stop", 0): "waiting",
    ("stop", 1): "start",
}

# debugging tools
logger = {
    "symbol": [],
    "state": [],
    "state_func": [],
    "transition_func": [],
    "recorder": [],
}

# TFunc = mk_transition_func(trans_func_mapping, "waiting")
dag = DAG.from_funcs(
    recorder_switch=lambda store: mk_recorder_switch(store),
    recorder_logger=lambda recorder_switch: logger["recorder"].append(
        id(recorder_switch)
    ),
    # debug = lambda recorder_switch: print(id(recorder_switch)),
    transition_func=lambda trans_func_mapping: mk_transition_func(
        trans_func_mapping, "waiting"
    ),
    transition_logger=lambda transition_func: logger["transition_func"].append(
        transition_func
    ),
    symbol=lambda plc: plc,
    symbol_logger=lambda symbol: logger["symbol"].append(symbol),
    state=lambda transition_func, symbol: transition_func(symbol),
    # tFunc=lambda: TFunc,
    # state=lambda tFunc, symbol: tFunc(symbol),
    state_logger=lambda state: logger["state"].append(state),
    key_and_chk=lambda key, chk: (key, chk),
    # key_and_chk_logger=lambda key_and_chk: logger['key_and_chk'].append(key_and_chk),
    state_func=lambda recorder_switch, state, key_and_chk: recorder_switch(
        state, key_and_chk
    ),
    state_func_logger=lambda state_func: logger["state_func"].append(state_func),
    output=lambda state_func, key_and_chk: (
        state_func(*key_and_chk) if state_func is not None else None
    ),
    result=lambda recorder_switch, transition_func, symbol, state, key_and_chk, state_func, output: dict(
        recorder_switch=recorder_switch,
        transition_func=transition_func,
        symbol=symbol,
        state=state,
        key_and_chk=key_and_chk,
        state_func=state_func,
        output=output,
    ),
)


if __name__ == "__main__":
    store = dict()

    my_dag = dag.partial(store=store, trans_func_mapping=trans_func_mapping)
    # my_dag.dot_digraph()
    # print(i2.Sig(my_dag))

    audio_chks, plc_values = mk_test_objects()
    keys = range(max(len(audio_chks), len(plc_values)))  # need some source of keys now!

    for chk, plc, key in zip(audio_chks, plc_values, keys):
        # print(f"{store =}{chk=} {plc=} {key=} ")
        # print(f'{store=}')
        # print(f"{my_dag.last_scope=}")

        # print(f"{my_dag[:'state_func'](chk=chk, plc=plc, key=key)}")
        # res = my_dag(chk=chk, plc=plc, key=key)
        res = dag(
            store=store,
            trans_func_mapping=trans_func_mapping,
            chk=chk,
            plc=plc,
            key=key,
        )

        # print(store)  # Careful: use keyword
    print(logger["symbol"])

    print(logger["state"])
    print(logger["recorder"])

    # transitioner = logger['transition_func'][-1]
    transitioner = TFunc
    state_sequence = list(map(transitioner, plc_values))
    print(state_sequence)
    # print(transitioner)
```

## scrap/gk_with_networkx.py

```python
"""
seriously modified version of yahoo/graphkit
"""

# ---------- base --------------------------------------------------------------


class Data:
    """
    This wraps any data that is consumed or produced
    by a Operation. This data should also know how to serialize
    itself appropriately.
    This class an "abstract" class that should be extended by
    any class working with data in the HiC framework.
    """

    def __init__(self, **kwargs):
        pass

    def get_data(self):
        raise NotImplementedError

    def set_data(self, data):
        raise NotImplementedError


from dataclasses import dataclass, field


@dataclass
class Operation:
    """
    This is an abstract class representing a data transformation. To use this,
    please inherit from this class and customize the ``.compute`` method to your
    specific application.

    Names may be given to this layer and its inputs and outputs. This is
    important when connecting layers and data in a Network object, as the
    names are used to construct the graph.
    :param str name: The name the operation (e.g. conv1, conv2, etc..)
    :param list needs: Names of input data objects this layer requires.
    :param list provides: Names of output data objects this provides.
    :param dict params: A dict of key/value pairs representing parameters
                        associated with your operation. These values will be
                        accessible using the ``.params`` attribute of your object.
                        NOTE: It's important that any values stored in this
                        argument must be pickelable.
    """

    name: str = field(default="None")
    needs: list = field(default=None)
    provides: list = field(default=None)
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        This method is a hook for you to override. It gets called after this
        object has been initialized with its ``needs``, ``provides``, ``name``,
        and ``params`` attributes. People often override this method to implement
        custom loading logic required for objects that do not pickle easily, and
        for initialization of c++ dependencies.
        """
        pass

    def __eq__(self, other):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return bool(self.name is not None and self.name == getattr(other, "name", None))

    def __hash__(self):
        """
        Operation equality is based on name of layer.
        (__eq__ and __hash__ must be overridden together)
        """
        return hash(self.name)

    def compute(self, inputs):
        """
        This method must be implemented to perform this layer's feed-forward
        computation on a given set of inputs.
        :param list inputs:
            A list of :class:`Data` objects on which to run the layer's
            feed-forward computation.
        :returns list:
            Should return a list of :class:`Data` objects representing
            the results of running the feed-forward computation on
            ``inputs``.
        """

        raise NotImplementedError

    def _compute(self, named_inputs, outputs=None):
        inputs = [named_inputs[d] for d in self.needs]
        results = self.compute(inputs)

        results = zip(self.provides, results)
        if outputs:
            outputs = set(outputs)
            results = filter(lambda x: x[0] in outputs, results)

        return dict(results)

    def __getstate__(self):
        """
        This allows your operation to be pickled.
        Everything needed to instantiate your operation should be defined by the
        following attributes: params, needs, provides, and name
        No other piece of state should leak outside of these 4 variables
        """

        result = {}
        # this check should get deprecated soon. its for downward compatibility
        # with earlier pickled operation objects
        if hasattr(self, "params"):
            result["params"] = self.__dict__["params"]
        result["needs"] = self.__dict__["needs"]
        result["provides"] = self.__dict__["provides"]
        result["name"] = self.__dict__["name"]

        return result

    def __setstate__(self, state):
        """
        load from pickle and instantiate the detector
        """
        for k in iter(state):
            self.__setattr__(k, state[k])
        self.__postinit__()

    def __repr__(self):
        """
        Display more informative names for the Operation class
        """
        return "{}(name='{}', needs={}, provides={})".format(
            self.__class__.__name__,
            self.name,
            self.needs,
            self.provides,
        )


class NetworkOperation(Operation):
    def __init__(self, **kwargs):
        self.net = kwargs.pop("net")
        Operation.__init__(self, **kwargs)

        # set execution mode to single-threaded sequential by default
        self._execution_method = "sequential"

    def _compute(self, named_inputs, outputs=None):
        return self.net.compute(outputs, named_inputs, method=self._execution_method)

    def __call__(self, *args, **kwargs):
        return self._compute(*args, **kwargs)

    def set_execution_method(self, method):
        """
        Determine how the network will be executed.
        Args:
            method: str
                If "parallel", execute graph operations concurrently
                using a threadpool.
        """
        options = ["parallel", "sequential"]
        assert method in options
        self._execution_method = method

    def plot(self, filename=None, show=False):
        self.net.plot(filename=filename, show=show)

    def __getstate__(self):
        state = Operation.__getstate__(self)
        state["net"] = self.__dict__["net"]
        return state


# ------------ modifiers -------------------------------------------------------

"""
This sub-module contains input/output modifiers that can be applied to
arguments to ``needs`` and ``provides`` to let GraphKit know it should treat
them differently.

Copyright 2016, Yahoo Inc.
Licensed under the terms of the Apache License, Version 2.0. See the LICENSE
file associated with the project for terms.
"""


class optional(str):
    """
    Input values in ``needs`` may be designated as optional using this modifier.
    If this modifier is applied to an input value, that value will be input to
    the ``operation`` if it is available.  The function underlying the
    ``operation`` should have a parameter with the same name as the input value
    in ``needs``, and the input value will be passed as a keyword argument if
    it is available.

    Here is an example of an operation that uses an optional argument::

        from graphkit import operation, compose
        from graphkit.modifiers import optional

        # Function that adds either two or three numbers.
        def myadd(a, b, c=0):
            return a + b + c

        # Designate c as an optional argument.
        graph = compose('mygraph')(
            operator(name='myadd', needs=['a', 'b', optional('c')], provides='sum')(myadd)
        )

        # The graph works with and without 'c' provided as input.
        assert graph({'a': 5, 'b': 2, 'c': 4})['sum'] == 11
        assert graph({'a': 5, 'b': 2})['sum'] == 7

    """

    pass


# ------------ network ------------------------------------------------------

# Copyright 2016, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.


from contextlib import suppress

with suppress(ModuleNotFoundError, ImportError):
    import time
    import os
    import networkx as nx

    from io import StringIO

    # uses base.Operation

    class DataPlaceholderNode(str):
        """
        A node for the Network graph that describes the name of a Data instance
        produced or required by a layer.
        """

        def __repr__(self):
            return 'DataPlaceholderNode("%s")' % self

    class DeleteInstruction(str):
        """
        An instruction for the compiled list of evaluation steps to free or delete
        a Data instance from the Network's cache after it is no longer needed.
        """

        def __repr__(self):
            return 'DeleteInstruction("%s")' % self

    class Network:
        """
        This is the main network implementation. The class contains all of the
        code necessary to weave together operations into a directed-acyclic-graph (DAG)
        and pass data through.
        """

        def __init__(self, **kwargs):
            """ """

            # directed graph of layer instances and data-names defining the net.
            self.graph = nx.DiGraph()
            self._debug = kwargs.get("debug", False)

            # this holds the timing information for eache layer
            self.times = {}

            # a compiled list of steps to evaluate layers *in order* and free mem.
            self.steps = []

            # This holds a cache of results for the _find_necessary_steps
            # function, this helps speed up the compute call as well avoid
            # a multithreading issue that is occuring when accessing the
            # graph in networkx
            self._necessary_steps_cache = {}

        def add_op(self, operation):
            """
            Adds the given operation and its data requirements to the network graph
            based on the name of the operation, the names of the operation's needs, and
            the names of the data it provides.

            :param Operation operation: Operation object to add.
            """

            # assert layer and its data requirements are named.
            assert operation.name, "Operation must be named"
            assert operation.needs is not None, "Operation's 'needs' must be named"
            assert (
                operation.provides is not None
            ), "Operation's 'provides' must be named"

            # assert layer is only added once to graph
            assert (
                operation not in self.graph.nodes()
            ), "Operation may only be added once"

            # add nodes and edges to graph describing the data needs for this layer
            for n in operation.needs:
                self.graph.add_edge(DataPlaceholderNode(n), operation)

            # add nodes and edges to graph describing what this layer provides
            for p in operation.provides:
                self.graph.add_edge(operation, DataPlaceholderNode(p))

            # clear compiled steps (must recompile after adding new layers)
            self.steps = []

        def list_layers(self):
            assert self.steps, "network must be compiled before listing layers."
            return [(s.name, s) for s in self.steps if isinstance(s, Operation)]

        def show_layers(self):
            """Shows info (name, needs, and provides) about all layers in this network."""
            for name, step in self.list_layers():
                print("layer_name: ", name)
                print("\t", "needs: ", step.needs)
                print("\t", "provides: ", step.provides)
                print("")

        def compile(self):
            """Create a set of steps for evaluating layers
            and freeing memory as necessary"""

            # clear compiled steps
            self.steps = []

            # create an execution order such that each layer's needs are provided.
            ordered_nodes = list(nx.dag.topological_sort(self.graph))

            # add Operations evaluation steps, and instructions to free data.
            for i, node in enumerate(ordered_nodes):

                if isinstance(node, DataPlaceholderNode):
                    continue

                elif isinstance(node, Operation):

                    # add layer to list of steps
                    self.steps.append(node)

                    # Add instructions to delete predecessors as possible.  A
                    # predecessor may be deleted if it is a data placeholder that
                    # is no longer needed by future Operations.
                    for predecessor in self.graph.predecessors(node):
                        if self._debug:
                            print("checking if node %s can be deleted" % predecessor)
                        predecessor_still_needed = False
                        for future_node in ordered_nodes[i + 1 :]:
                            if isinstance(future_node, Operation):
                                if predecessor in future_node.needs:
                                    predecessor_still_needed = True
                                    break
                        if not predecessor_still_needed:
                            if self._debug:
                                print(
                                    "  adding delete instruction for %s" % predecessor
                                )
                            self.steps.append(DeleteInstruction(predecessor))

                else:
                    raise TypeError("Unrecognized network graph node")

        def _find_necessary_steps(self, outputs, inputs):
            """
            Determines what graph steps need to pe run to get to the requested
            outputs from the provided inputs.  Eliminates steps that come before
            (in topological order) any inputs that have been provided.  Also
            eliminates steps that are not on a path from he provided inputs to
            the requested outputs.

            :param list outputs:
                A list of desired output names.  This can also be ``None``, in which
                case the necessary steps are all graph nodes that are reachable
                from one of the provided inputs.

            :param dict inputs:
                A dictionary mapping names to values for all provided inputs.

            :returns:
                Returns a list of all the steps that need to be run for the
                provided inputs and requested outputs.
            """

            # return steps if it has already been computed before for this set of inputs and outputs
            outputs = (
                tuple(sorted(outputs)) if isinstance(outputs, (list, set)) else outputs
            )
            inputs_keys = tuple(sorted(inputs.keys()))
            cache_key = (inputs_keys, outputs)
            if cache_key in self._necessary_steps_cache:
                return self._necessary_steps_cache[cache_key]

            graph = self.graph
            if not outputs:

                # If caller requested all outputs, the necessary nodes are all
                # nodes that are reachable from one of the inputs.  Ignore input
                # names that aren't in the graph.
                necessary_nodes = set()
                for input_name in iter(inputs):
                    if graph.has_node(input_name):
                        necessary_nodes |= nx.descendants(graph, input_name)

            else:

                # If the caller requested a subset of outputs, find any nodes that
                # are made unecessary because we were provided with an input that's
                # deeper into the network graph.  Ignore input names that aren't
                # in the graph.
                unnecessary_nodes = set()
                for input_name in iter(inputs):
                    if graph.has_node(input_name):
                        unnecessary_nodes |= nx.ancestors(graph, input_name)

                # Find the nodes we need to be able to compute the requested
                # outputs.  Raise an exception if a requested output doesn't
                # exist in the graph.
                necessary_nodes = set()
                for output_name in outputs:
                    if not graph.has_node(output_name):
                        raise ValueError(
                            "graphkit graph does not have an output "
                            "node named %s" % output_name
                        )
                    necessary_nodes |= nx.ancestors(graph, output_name)

                # Get rid of the unnecessary nodes from the set of necessary ones.
                necessary_nodes -= unnecessary_nodes

            necessary_steps = [step for step in self.steps if step in necessary_nodes]

            # save this result in a precomputed cache for future lookup
            self._necessary_steps_cache[cache_key] = necessary_steps

            # Return an ordered list of the needed steps.
            return necessary_steps

        def compute(self, outputs, named_inputs, method=None):
            """
            Run the graph. Any inputs to the network must be passed in by name.

            :param list output: The names of the data node you'd like to have returned
                                once all necessary computations are complete.
                                If you set this variable to ``None``, all
                                data nodes will be kept and returned at runtime.

            :param dict named_inputs: A dict of key/value pairs where the keys
                                      represent the data nodes you want to populate,
                                      and the values are the concrete values you
                                      want to set for the data node.


            :returns: a dictionary of output data objects, keyed by name.
            """

            # assert that network has been compiled
            assert self.steps, "network must be compiled before calling compute."
            assert (
                isinstance(outputs, (list, tuple)) or outputs is None
            ), "The outputs argument must be a list"

            # choose a method of execution
            if method == "parallel":
                return self._compute_thread_pool_barrier_method(named_inputs, outputs)
            else:
                return self._compute_sequential_method(named_inputs, outputs)

        def _compute_thread_pool_barrier_method(
            self, named_inputs, outputs, thread_pool_size=10
        ):
            """
            This method runs the graph using a parallel pool of thread executors.
            You may achieve lower total latency if your graph is sufficiently
            sub divided into operations using this method.
            """
            from multiprocessing.dummy import Pool

            # if we have not already created a thread_pool, create one
            if not hasattr(self, "_thread_pool"):
                self._thread_pool = Pool(thread_pool_size)
            pool = self._thread_pool

            cache = {}
            cache.update(named_inputs)
            necessary_nodes = self._find_necessary_steps(outputs, named_inputs)

            # this keeps track of all nodes that have already executed
            has_executed = set()

            # with each loop iteration, we determine a set of operations that can be
            # scheduled, then schedule them onto a thread pool, then collect their
            # results onto a memory cache for use upon the next iteration.
            while True:

                # the upnext list contains a list of operations for scheduling
                # in the current round of scheduling
                upnext = []
                for node in necessary_nodes:
                    # only delete if all successors for the data node have been executed
                    if isinstance(node, DeleteInstruction):
                        if ready_to_delete_data_node(node, has_executed, self.graph):
                            if node in cache:
                                cache.pop(node)

                    # continue if this node is anything but an operation node
                    if not isinstance(node, Operation):
                        continue

                    if (
                        ready_to_schedule_operation(node, has_executed, self.graph)
                        and node not in has_executed
                    ):
                        upnext.append(node)

                # stop if no nodes left to schedule, exit out of the loop
                if len(upnext) == 0:
                    break

                done_iterator = pool.imap_unordered(
                    lambda op: (op, op._compute(cache)), upnext
                )
                for op, result in done_iterator:
                    cache.update(result)
                    has_executed.add(op)

            if not outputs:
                return cache
            else:
                return {k: cache[k] for k in iter(cache) if k in outputs}

        def _compute_sequential_method(self, named_inputs, outputs):
            """
            This method runs the graph one operation at a time in a single thread
            """
            # start with fresh data cache
            cache = {}

            # add inputs to data cache
            cache.update(named_inputs)

            # Find the subset of steps we need to run to get to the requested
            # outputs from the provided inputs.
            all_steps = self._find_necessary_steps(outputs, named_inputs)

            self.times = {}
            for step in all_steps:

                if isinstance(step, Operation):

                    if self._debug:
                        print("-" * 32)
                        print("executing step: %s" % step.name)

                    # time execution...
                    t0 = time.time()

                    # compute layer outputs
                    layer_outputs = step._compute(cache)

                    # add outputs to cache
                    cache.update(layer_outputs)

                    # record execution time
                    t_complete = round(time.time() - t0, 5)
                    self.times[step.name] = t_complete
                    if self._debug:
                        print("step completion time: %s" % t_complete)

                # Process DeleteInstructions by deleting the corresponding data
                # if possible.
                elif isinstance(step, DeleteInstruction):

                    if outputs and step not in outputs:
                        # Some DeleteInstruction steps may not exist in the cache
                        # if they come from optional() needs that are not privoded
                        # as inputs.  Make sure the step exists before deleting.
                        if step in cache:
                            if self._debug:
                                print("removing data '%s' from cache." % step)
                            cache.pop(step)

                else:
                    raise TypeError("Unrecognized instruction.")

            if not outputs:
                # Return the whole cache as output, including input and
                # intermediate data nodes.
                return cache

            else:
                # Filter outputs to just return what's needed.
                # Note: list comprehensions exist in python 2.7+
                return {k: cache[k] for k in iter(cache) if k in outputs}

        def plot(self, filename=None, show=False):
            """
            Plot the graph.

            params:
            :param str filename:
                Write the output to a png, pdf, or graphviz dot file. The extension
                controls the output format.

            :param boolean show:
                If this is set to True, use matplotlib to show the graph diagram
                (Default: False)

            :returns:
                An instance of the pydot graph

            """
            from contextlib import suppress

            with suppress(ModuleNotFoundError, ImportError):
                import pydot
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg

                assert self.graph is not None

                def get_node_name(a):
                    if isinstance(a, DataPlaceholderNode):
                        return a
                    return a.name

                g = pydot.Dot(graph_type="digraph")

                # draw nodes
                for nx_node in self.graph.nodes():
                    if isinstance(nx_node, DataPlaceholderNode):
                        node = pydot.Node(name=nx_node, shape="rect")
                    else:
                        node = pydot.Node(name=nx_node.name, shape="circle")
                    g.add_node(node)

                # draw edges
                for src, dst in self.graph.edges():
                    src_name = get_node_name(src)
                    dst_name = get_node_name(dst)
                    edge = pydot.Edge(src=src_name, dst=dst_name)
                    g.add_edge(edge)

                # save plot
                if filename:
                    basename, ext = os.path.splitext(filename)
                    with open(filename, "w") as fh:
                        if ext.lower() == ".png":
                            fh.write(g.create_png())
                        elif ext.lower() == ".dot":
                            fh.write(g.to_string())
                        elif ext.lower() in [".jpg", ".jpeg"]:
                            fh.write(g.create_jpeg())
                        elif ext.lower() == ".pdf":
                            fh.write(g.create_pdf())
                        elif ext.lower() == ".svg":
                            fh.write(g.create_svg())
                        else:
                            raise Exception(
                                "Unknown file format for saving graph: %s" % ext
                            )

                # display graph via matplotlib
                if show:
                    png = g.create_png()
                    sio = StringIO(png)
                    img = mpimg.imread(sio)
                    plt.imshow(img, aspect="equal")
                    plt.show()

                return g

    def ready_to_schedule_operation(op, has_executed, graph):
        """
        Determines if a Operation is ready to be scheduled for execution based on
        what has already been executed.

        Args:
            op:
                The Operation object to check
            has_executed: set
                A set containing all operations that have been executed so far
            graph:
                The networkx graph containing the operations and data nodes
        Returns:
            A boolean indicating whether the operation may be scheduled for
            execution based on what has already been executed.
        """
        dependencies = set(
            filter(lambda v: isinstance(v, Operation), nx.ancestors(graph, op))
        )
        return dependencies.issubset(has_executed)

    def ready_to_delete_data_node(name, has_executed, graph):
        """
        Determines if a DataPlaceholderNode is ready to be deleted from the
        cache.

        Args:
            name:
                The name of the data node to check
            has_executed: set
                A set containing all operations that have been executed so far
            graph:
                The networkx graph containing the operations and data nodes
        Returns:
            A boolean indicating whether the data node can be deleted or not.
        """
        data_node = get_data_node(name, graph)
        return set(graph.successors(data_node)).issubset(has_executed)

    def get_data_node(name, graph):
        """
        Gets a data node from a graph using its name
        """
        for node in graph.nodes():
            if node == name and isinstance(node, DataPlaceholderNode):
                return node
        return None

    # ------------ functional ------------------------------------------------------

    # Copyright 2016, Yahoo Inc.
    # Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.

    from itertools import chain

    # uses Operation, NetworkOperation from base
    # uses Network from network

    class FunctionalOperation(Operation):
        def __init__(self, **kwargs):
            self.fn = kwargs.pop("fn")
            Operation.__init__(self, **kwargs)

        def _compute(self, named_inputs, outputs=None):
            inputs = [
                named_inputs[d] for d in self.needs if not isinstance(d, optional)
            ]

            # Find any optional inputs in named_inputs.  Get only the ones that
            # are present there, no extra `None`s.
            optionals = {
                n: named_inputs[n]
                for n in self.needs
                if isinstance(n, optional) and n in named_inputs
            }

            # Combine params and optionals into one big glob of keyword arguments.
            kwargs = {k: v for d in (self.params, optionals) for k, v in d.items()}
            result = self.fn(*inputs, **kwargs)
            if len(self.provides) == 1:
                result = [result]

            result = zip(self.provides, result)
            if outputs:
                outputs = set(outputs)
                result = filter(lambda x: x[0] in outputs, result)

            return dict(result)

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def __getstate__(self):
            state = Operation.__getstate__(self)
            state["fn"] = self.__dict__["fn"]
            return state

    class operation(Operation):
        """
        This object represents an operation in a computation graph.  Its
        relationship to other operations in the graph is specified via its
        ``needs`` and ``provides`` arguments.

        :param function fn:
            The function used by this operation.  This does not need to be
            specified when the operation object is instantiated and can instead
            be set via ``__call__`` later.

        :param str name:
            The name of the operation in the computation graph.

        :param list needs:
            Names of input data objects this operation requires.  These should
            correspond to the ``args`` of ``fn``.

        :param list provides:
            Names of output data objects this operation provides.

        :param dict params:
            A dict of key/value pairs representing constant parameters
            associated with your operation.  These can correspond to either
            ``args`` or ``kwargs`` of ``fn`.
        """

        def __init__(self, fn=None, **kwargs):
            self.fn = fn
            Operation.__init__(self, **kwargs)

        def _normalize_kwargs(self, kwargs):

            # Allow single value for needs parameter
            if "needs" in kwargs and type(kwargs["needs"]) == str:
                assert kwargs["needs"], "empty string provided for `needs` parameters"
                kwargs["needs"] = [kwargs["needs"]]

            # Allow single value for provides parameter
            if "provides" in kwargs and type(kwargs["provides"]) == str:
                assert kwargs[
                    "provides"
                ], "empty string provided for `needs` parameters"
                kwargs["provides"] = [kwargs["provides"]]

            assert kwargs["name"], "operation needs a name"
            assert type(kwargs["needs"]) == list, "no `needs` parameter provided"
            assert type(kwargs["provides"]) == list, "no `provides` parameter provided"
            assert hasattr(
                kwargs["fn"], "__call__"
            ), "operation was not provided with a callable"

            if type(kwargs["params"]) is not dict:
                kwargs["params"] = {}

            return kwargs

        def __call__(self, fn=None, **kwargs):
            """
            This enables ``operation`` to act as a decorator or as a functional
            operation, for example::

                @operator(name='myadd1', needs=['a', 'b'], provides=['c'])
                def myadd(a, b):
                    return a + b

            or::

                def myadd(a, b):
                    return a + b
                operator(name='myadd1', needs=['a', 'b'], provides=['c'])(myadd)

            :param function fn:
                The function to be used by this ``operation``.

            :return:
                Returns an operation class that can be called as a function or
                composed into a computation graph.
            """

            if fn is not None:
                self.fn = fn

            total_kwargs = {}
            total_kwargs.update(vars(self))
            total_kwargs.update(kwargs)
            total_kwargs = self._normalize_kwargs(total_kwargs)

            return FunctionalOperation(**total_kwargs)

        def __repr__(self):
            """
            Display more informative names for the Operation class
            """
            return "{}(name='{}', needs={}, provides={}, fn={})".format(
                self.__class__.__name__,
                self.name,
                self.needs,
                self.provides,
                self.fn.__name__,
            )

    class compose:
        """
        This is a simple class that's used to compose ``operation`` instances into
        a computation graph.

        :param str name:
            A name for the graph being composed by this object.

        :param bool merge:
            If ``True``, this compose object will attempt to merge together
            ``operation`` instances that represent entire computation graphs.
            Specifically, if one of the ``operation`` instances passed to this
            ``compose`` object is itself a graph operation created by an
            earlier use of ``compose`` the sub-operations in that graph are
            compared against other operations passed to this ``compose``
            instance (as well as the sub-operations of other graphs passed to
            this ``compose`` instance).  If any two operations are the same
            (based on name), then that operation is computed only once, instead
            of multiple times (one for each time the operation appears).
        """

        def __init__(self, name=None, merge=False):
            assert name, "compose needs a name"
            self.name = name
            self.merge = merge

        def __call__(self, *operations):
            """
            Composes a collection of operations into a single computation graph,
            obeying the ``merge`` property, if set in the constructor.

            :param operations:
                Each argument should be an operation instance created using
                ``operation``.

            :return:
                Returns a special type of operation class, which represents an
                entire computation graph as a single operation.
            """
            assert len(operations), "no operations provided to compose"

            # If merge is desired, deduplicate operations before building network
            if self.merge:
                merge_set = set()
                for op in operations:
                    if isinstance(op, NetworkOperation):
                        net_ops = filter(
                            lambda x: isinstance(x, Operation), op.net.steps
                        )
                        merge_set.update(net_ops)
                    else:
                        merge_set.add(op)
                operations = list(merge_set)

            def order_preserving_uniquifier(seq, seen=None):
                seen = seen if seen else set()
                seen_add = seen.add
                return [x for x in seq if not (x in seen or seen_add(x))]

            provides = order_preserving_uniquifier(
                chain(*[op.provides for op in operations])
            )
            needs = order_preserving_uniquifier(
                chain(*[op.needs for op in operations]), set(provides)
            )

            # compile network
            net = Network()
            for op in operations:
                net.add_op(op)
            net.compile()

            return NetworkOperation(
                name=self.name, needs=needs, provides=provides, params={}, net=net
            )
```

## scrap/gui_interaction.py

```python
"""
This module contains some ideas around making a two-way interaction between meshed
and a GUI that will enable the construction of meshes as well as rendering them,
and possibly running them.
"""

from warnings import warn

warn("Deprecated: Moved to meshed.makers")

from meshed.makers import *
```

## scrap/misc_utils.py

```python
"""Misc utils"""

from collections.abc import Mapping
from meshed.util import ModuleNotFoundIgnore
from collections import deque, defaultdict

with ModuleNotFoundIgnore():
    import networkx as nx

    topological_sort_2 = nx.dag.topological_sort


from collections.abc import Iterable


def mermaid_pack_nodes(
    mermaid_code: str,
    nodes: Iterable[str],
    packed_node_name: str = None,
    *,
    arrow: str = "-->",
) -> str:
    """
    Output mermaid code with nodes packed into a single node.

    >>> mermaid_code = '''
    ... graph TD
    ...   A --> B
    ...   B --> C
    ...   A --> D
    ...   D --> E
    ...   E --> C
    ... '''
    >>>
    >>>
    >>> print(mermaid_pack_nodes(mermaid_code, ['B', 'C', 'E'], 'BCE'))  # doctest: +NORMALIZE_WHITESPACE
    graph TD
    A -->BCE
    A --> D
    D -->BCE
    """
    if packed_node_name is None:
        packed_node_name = "__".join(nodes)

    def gen_lines():
        for line in mermaid_code.strip().split("\n"):
            source, _arrow, target = line.partition(arrow)
            if source.strip() in nodes:
                source = packed_node_name
            if target.strip() in nodes:
                target = packed_node_name
            if (source != target) and source != packed_node_name:
                yield f"{_arrow}".join([source, target])
            else:
                # If there are any loops within the nodes to be packed,
                # they'll be represented as a loop for the packed node, which we skip.
                continue

    return "\n".join(gen_lines())


from typing import Any
from collections.abc import Mapping, Sized, MutableMapping, Iterable
from meshed.itools import children, parents


def coparents_sets(g: Mapping, source: Iterable):
    res = []
    for node in source:
        for kid in children(g, [node]):
            res.append(frozenset(parents(g, [kid])))
    return set(res)


def known_parents(g: Mapping, kid, source):
    return parents(g, [kid]).issubset(set(source))


def list_coparents(g: Mapping, coparent):
    all_kids = children(g, [coparent])
    result = [parents(g, [kid]) for kid in all_kids]

    return result


def kids_of_united_family(g: Mapping, source: Iterable):
    res = set()
    for coparent in source:
        for kid in children(g, [coparent]):
            if known_parents(g, kid, source):
                res.add(kid)
    return res


def extended_family(g: Mapping, source: Iterable):
    res = set(source)
    while True:
        allowed_kids = kids_of_united_family(g, res)
        if allowed_kids.issubset(res):
            return res
        res = res.union(allowed_kids)


from meshed.dag import DAG
from meshed.base import FuncNode


def funcnode_only(source: Iterable):
    return [item for item in source if isinstance(item, FuncNode)]


def dag_from_funcnodes(dag, input_names):
    kids = extended_family(g=dag.graph, source=input_names)
    fnodes = funcnode_only(kids)

    return DAG(fnodes)
```

## scrap/reactive_scope.py

```python
"""
Ideas towards a reactive-programming interpretation of meshes.
A scope (MutableMapping --think dict-like) that reacts to writes by computing
associated functions, themselves writing in the scope, creating a chain reaction that
propagates information through the scope.
"""

from functools import cached_property

from meshed import DAG, FuncNode

# TODO: Should ReactiveFuncNode exist? Could put the logic in a function and used in
#  ReactiveScope instead.


class ReactiveFuncNode(FuncNode):
    """A ``FuncNode`` that computes on a scope only if the scope has what it takes"""

    @cached_property
    def _dependencies(self):
        """The keys the scope needs to have so that the FuncNode is callable"""
        return set(self.bind.values())

    def call_on_scope(self, scope, write_output_into_scope=True):
        if self._dependencies.issubset(scope):
            return super().call_on_scope(scope, write_output_into_scope)


from collections.abc import MutableMapping


# TODO: Don't seem to need the relations to be acyclic. Try it out, and make it work.
# TODO: If we allow multiple writes/deletes, to a key or even to different keys,
#  we open ourselves to a lot more complexity. We need to be able to detect when
#  existing values are not valid anymore, given the relations exhibited by the func
#  nodes. This is a lot of work, and I'm not sure it's worth it. Might be better to
#  keep the scope as a simple mapping, and protect it from having actions taken on it
#  that might bring it into an invalid state.
class ReactiveScope(MutableMapping):
    """
    A scope that reacts to writes by computing associated functions, themselves writing
    in the scope, creating a chain reaction that propagates information through the
    scope.

    Parameters
    ----------
    func_nodes : Iterable[ReactiveFuncNode]
        The functions that will be called when the scope is written to.
    scope_factory : Callable[[], MutableMapping]
        A factory that returns a new scope. The scope will be cleared by calling this
        factory at each call to `.clear()`.

    Examples
    --------

    First, we need some func nodes to define the reaction relationships.
    We'll stuff these func nodes in a DAG, for ease of use, but it's not necessary.

    >>> from meshed import FuncNode, DAG
    >>>
    >>> def f(a, b):
    ...     return a + b
    >>> def g(a_plus_b, d):
    ...     return a_plus_b * d
    >>> f_node = FuncNode(func=f, out='a_plus_b')
    >>> g_node = FuncNode(func=g, bind={'d': 'b'})
    >>> d = DAG((f_node, g_node))
    >>>
    >>> print(d.dot_digraph_ascii())
    <BLANKLINE>
                  a
    <BLANKLINE>
                │
                │
                ▼
              ┌────────┐
      b   ──▶ │   f    │
              └────────┘
      │         │
      │         │
      │         ▼
      │
      │        a_plus_b
      │
      │         │
      │         │
      │         ▼
      │       ┌────────┐
      └─────▶ │   g_   │
              └────────┘
                │
                │
                ▼
    <BLANKLINE>
                  g
    <BLANKLINE>

    Now we make a scope with these func nodes.

    >>> s = ReactiveScope(d)

    The scope starts empty (by default).

    >>> s
    <ReactiveScope with .scope: {}>

    So if we try to access any key, we'll get a KeyError.

    >>> s['g']  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    KeyError: 'g'

    That's because we didn't put write anything in the scope yet.

    But, if you give ``g_`` enough data to be able to compute ``g`` (namely, if you
    write values of ``b`` and ``a_plus_b``), then ``g`` will automatically be computed.

    >>> s['b'] = 3
    >>> s['a_plus_b'] = 5
    >>> s
    <ReactiveScope with .scope: {'b': 3, 'a_plus_b': 5, 'g': 15}>

    So now we can access ``g``.

    >>> s['g']
    15

    Note though, that we first showed that ``g`` appeared in the scope before we
    explicitly asked for it. This was to show that ``g`` was computed as a
    side-effect of writing to the scope, not because we asked for it, triggering the
    computation

    Let's clear the scope and show that by specifying ``a`` and ``b``, we get all the
    other values of the network.

    >>> s.clear()
    >>> s
    <ReactiveScope with .scope: {}>
    >>> s['a'] = 3
    >>> s['b'] = 4
    >>> s
    <ReactiveScope with .scope: {'a': 3, 'b': 4, 'a_plus_b': 7, 'g': 28}>
    >>> s['g']  # (3 + 4) * 4 == 7 * 4 == 28
    28
    """

    def __init__(self, func_nodes=(), scope_factory=dict):
        # Note: scope_factory could be made to return a pre-filled dict too
        if isinstance(func_nodes, DAG):
            dag = func_nodes
            func_nodes = dag.func_nodes
        func_nodes = [ReactiveFuncNode.from_dict(fn.to_dict()) for fn in func_nodes]
        self.dag = DAG(func_nodes)
        self.func_nodes_for_var_node = {
            k: v for k, v in self.dag.graph.items() if k in self.dag.var_nodes
        }
        self.scope_factory = scope_factory
        self.clear()

    def clear(self):
        """Note: This actually doesn't clear the mapping, but rather, resets it to it's original state,
        as defined by the `.scope_factory`"""
        self.scope = self.scope_factory()

    def __getitem__(self, k):
        # TODO: try/catch and give the user a bit more info (e.g. what dependencies are missing?)
        return self.scope[k]

    def __setitem__(self, k, v):
        # write the value under the key
        self.scope[k] = v
        # TODO: Need to make sure the func_node are in topological order
        # TODO: The .get(k, ()): prefill with missing keys at init time instead?
        for func_node in self.func_nodes_for_var_node.get(k, ()):
            # "try" calling the func_node on the scope (if scope doesn't have enough
            #
            func_node.call_on_scope(self.scope)

    def __len__(self):
        return len(self.scope)

    def __contains__(self, k):
        return k in self.scope

    def __iter__(self):
        return iter(self.scope)

    def __delitem__(self, k):
        # TODO: Could use the same mechanism as setitem to propagate the deletion through the network
        raise NotImplementedError(
            "deletion of keys are not implemented, since cache invalidation hasn't. "
            "You can clear the whole scope with the `.clear()` method. "
            "(Note: This actually doesn't clear the mapping, but rather, resets it to it's original state.)"
        )

    def __repr__(self):
        return f"<{type(self).__qualname__} with .scope: {repr(self.scope)}>"
```

## scrap/wrapping_dags.py

```python
"""Wrapping dags"""

from meshed import DAG


class DDag(DAG):
    wrappers = ()

    def _call(self, *args, **kwargs):
        if not self.wrappers:
            return super()._call(*args, **kwargs)
        else:
            decorator = Line(*self.wrappers)
            decorated_dag_call = decorator(super()._call)
            return decorated_dag_call(*args, **kwargs)


def test_ddag():
    def f(a, b=2):
        return a + b

    def g(f, c=3):
        return f * c

    # d = DDag([f, g])
    d = DDag([f, g])

    d.dot_digraph()

    assert d(1, 2, 3) == 9  # can call
    from i2 import Sig

    assert str(Sig(d)) == "(a, b=2, c=3)"  # has correct signature

    def dec(func):
        def _dec(*args, **kwargs):
            print(func.__name__, args, kwargs)
            return func(*args, **kwargs)

        return _dec

    def rev(func):
        def _rev(*args, **kwargs):
            assert not kwargs, "Can't have keyword arguments with rev"
            return func(*args[::-1])

        return _rev

    d.wrappers = (dec,)

    assert d(1, 2, 3) == 9
    # prints: _call (1, 2, 3) {}
    assert d(1, 2, c=3) == 9
    # prints: _call (1, 2) {'c': 3}

    d.wrappers = (dec, rev)
    assert d(1, 2, 3) == 5
    # prints: _call (3, 2, 1) {}
```

## slabs.py

```python
"""Tools to generate slabs.

A slab is a dict that holds data generated by a stream for a given interval of time.

The main object of this module is `Slabs`, and object that defines how to
generate multiple streams, in the form of slabs.
More precisely, it defines how to source streams, operate on and combine these to create
further streams, and even push these streams to further processes, all through a single
simple interface: An (ordered) list of components that are called in sequence to either
pull data from some sources, compute a new stream based on previous ones, or push some
of the streams to further processes (such as visualization, or storage systems).

A slab is a collection of items of a same interval of time.
We represent a slab using a `dict` or mapping.
Typically, a slab will be the aggregation of multiple information streams that
happened around the same time.

`Slabs` is a tool that allows you to source multiple streams into a stream of
slabs that can contain the original data, or other datas computed from it, or both.

Note to developers, though the code below is a reduced form of the actual code,
it should be enough to understand the general idea.
For a discussion about the design of Slabs, see
https://github.com/i2mint/meshed/discussions/49.


>>> class Slabs:
...     def _call_on_scope(self, scope):
...         '''
...         Calls the components 1 by 1, sourcing inputs and writing outputs in scope
...         '''
...
...     def __next__(self):
...         '''Get the next slab by calling _call_on_scope on an new empty scope.
...         At least one of the components will have to be argument-less and provide
...         some data for other components to get their inputs from, if any are needed.
...         '''
...         return self._call_on_scope(scope={})
...
...     def __iter__(self):
...         '''Iterates over slabs until a handle exception is raised.'''
...         # Simplified code:
...         with self:  # enter all the contexts that need to be entered
...             while True:  # loop until you encounter a handled exception
...                 try:
...                     yield next(self)
...                 except self.handle_exceptions as exc_val:
...                     # use specific exceptions to signal that iteration should stop
...                     break

"""

from typing import (
    Union,
    Any,
    Protocol,
)
from collections.abc import Callable, Mapping, Iterable, MutableMapping
from i2 import Sig, ContextFanout
from meshed.base import FuncNode, ensure_func_nodes
from meshed.dag import DAG


class ExceptionalException(Exception):
    """Raised when an exception was supposed to be handled, but no matching handler
    was found.

    See the `_handle_exception` function, where it is raised.
    """


class IteratorExit(BaseException):
    """Raised when an iterator should quit being iterated on, signaling this event
    any process that cares to catch the signal.
    We chose to inherit directly from `BaseException` instead of `Exception`
    for the same reason that `GeneratorExit` does: Because it's not technically
    an error.

    See: https://docs.python.org/3/library/exceptions.html#GeneratorExit
    """


DFLT_INTERRUPT_EXCEPTIONS = (StopIteration, IteratorExit, KeyboardInterrupt)

DoNotBreak = type("DoNotBreak", (), {})
do_not_break = DoNotBreak()
do_not_break.__doc__ = (
    "Sentinel that should be used to signal Slabs iteration not to break. "
    "This sentinel should be returned by exception handlers if they want to tell "
    "the iteration not to stop (in all other cases, the iteration will stop)"
)

IgnoredOutput = Any
ExceptionHandlerOutput = Union[IgnoredOutput, DoNotBreak]


class ExceptionHandler(Protocol):
    """An exception handler is an argument-less callable that is called when a handled
    exception occurs during iteration. Most often, the handler does nothing,
    but could be used whose output will be ignored, unless it is do_not_break,
    which will signal that the iteration should continue."""

    def __call__(self) -> ExceptionHandlerOutput:
        pass


# TODO: Make HandledExceptionsMap into a NewType?
# doc: A map between exception types and exception handlers (callbacks)
ExceptionType = type(BaseException)
HandledExceptionsMap = Mapping[ExceptionType, ExceptionHandler]

# doc: If none of the exceptions need handlers, you can just specify a list of them
HandledExceptionsMapSpec = Union[
    HandledExceptionsMap,
    Iterable[BaseException],  # an iterable of exception types
    BaseException,  # or just one exception type
]


def do_nothing():
    pass


def log_and_return(msg, logger=print):
    logger(msg)
    return msg


# TODO: Could consider (topologically) ordering the exceptions to reduce the matching
#  possibilities (see _handle_exception)
def _get_handle_exceptions(
    handle_exceptions: HandledExceptionsMapSpec,
) -> HandledExceptionsMap:
    if isinstance(handle_exceptions, BaseException):
        # Only one? Ensure there's a tuple of exceptions:
        handle_exceptions = (handle_exceptions,)
    if not isinstance(handle_exceptions, Mapping):
        handle_exceptions = {exc_type: do_nothing for exc_type in handle_exceptions}
    return handle_exceptions


def _handle_exception(
    instance, exc_val: BaseException, handle_exceptions: HandledExceptionsMap
) -> ExceptionHandlerOutput:
    """Looks for an exception type matching exc_val and calls the corresponding
    handler with
    """
    inputs = dict(exc_val=exc_val, instance=instance)
    if type(exc_val) in handle_exceptions:  # try precise matching first
        exception_handler = handle_exceptions[type(exc_val)]
        return _call_from_dict(inputs, exception_handler, Sig(exception_handler))

    else:  # if not, find the first matching parent
        for exc_type, exception_handler in handle_exceptions.items():
            if isinstance(exc_val, exc_type):
                return _call_from_dict(
                    inputs, exception_handler, Sig(exception_handler)
                )
    # You never should get this far, but if you do, there's a problem, let's scream it:
    raise ExceptionalException(
        f"I couldn't find that exception in my handlers: {exc_val}"
    )


def _call_from_dict(kwargs: MutableMapping, func: Callable, sig: Sig):
    """A i2.call_forgivingly optimized for our purpose

    The sig argument needs to be the Sig(func) to work correctly.

    Two uses cases here:

    - using a scope dict as both the source of `Slabs` components, and as a place
    to temporarily store the outputs of these components.

    - exception handlers: We'd like the exception handlers to be easy to express.
    Maybe you need the object raising the exception to handle it,
    maybe you just want to log the event.
    In the first case, you the handler needs the said object to be passed to it,
    in the second, we don't need any arguments at all.
    With _call_from_dict, we don't have to choose, we just have to impose that
    the handler use specific keywords (namely `exc_val` and/or `instance`)
    when there are inputs.

    """
    args, kwargs = sig.mk_args_and_kwargs(
        kwargs,
        allow_excess=True,
        ignore_kind=True,
        allow_partial=False,
        apply_defaults=True,
    )
    return func(*args, **kwargs)


def _conditional_pluralization(n_items, singular_msg, plural_msg):
    """To route to the right message (or template) according to ``n_items``"""
    if n_items == 1:
        return singular_msg
    else:
        return plural_msg


def _validate_components(components):
    if not all(map(callable, components.values())):
        not_callable = [k for k, v in components.items() if not callable(v)]
        not_callable_keys = ", ".join(not_callable)
        # TODO: Analyze values of components further and enhance error message with
        #  further suggestions. For example, if there's an iterator component c,
        #  suggest that perhaps ``c.__next__`` was intended?
        #  These component-based suggestions should be placed as a default of an
        #  argument of _validate_components so that it can be parametrized.
        msg = _conditional_pluralization(
            len(not_callable),
            f"This component is not callable: {not_callable_keys}",
            f"These components are not callable: {not_callable_keys}",
        )
        raise TypeError(msg)


# TODO: Postelize (or add tooling for) the components specification and add validation.
class Slabs:
    """Object to source and manipulate multiple streams.

    A slab is a collection of items of a same interval of time.
    We represent a slab using a `dict` or mapping.
    Typically, a slab will be the aggregation of multiple information streams that
    happened around the same time.

    For example, say and edge device had a microphone, light, and movement sensor.
    An aggregate reading of these sensors could give you something like:

    >>> slab = {'audio': [1, 2, 4], 'light': 126, 'movement': None}

    `movement` is `None` because the sensor is off. If it were on, we'd have True or
    False as values.

    From this information, you'd like to compute a `turn_mov_on` value based on the
    formula.

    >>> from statistics import stdev
    >>> vol = stdev
    >>> should_turn_movement_sensor_on = lambda audio, light: vol(audio) * light > 50000

    The produce of the volume and the lumens gives you 192, so you now have...

    >>> slab = {
    ...     'audio': [1, 2, 4],
    ...     'light': 126,
    ...     'should_turn_movement_sensor_on': False,
    ...     'movement': None
    ... }

    The next slab that comes in is

    >>> slab = {'audio': [-96, 89, -92], 'light': 501, 'movement': None}

    which puts us over the threshold so

    >>> slab = {
    ...     'audio': [-96, 89, -92],
    ...     'light': 501,
    ...     'should_turn_movement_sensor_on': True,
    ...     'movement': None
    ... }

    and the movement sensor is turned on, the movement is detected, a `human_presence`
    signal is computed, and a notification sent if that metric is above a given theshold.

    The point here is that we incrementally compute various fields, enhancing our slab
    of information, and we do so iteratively over over slab that is streaming to us
    from our smart home device.

    `SlabsIter` is there to help you create such slabs, from source to enhanced.

    The situation above would look something along like this:

    >>> from statistics import stdev
    >>>
    >>> vol = stdev
    >>>
    >>> # Making a slabs iter object
    >>> def make_a_slabs_iter():
    ...
    ...     # Mocking the sensor readers
    ...     audio_sensor_read = iter([[1, 2, 3], [-96, 87, -92], [320, -96, 99]]).__next__
    ...     light_sensor_read = iter([126, 501, 523]).__next__
    ...     movement_sensor_read = iter([None, None, True]).__next__
    ...
    ...     return Slabs(
    ...         # The first three components get data from the sensors.
    ...         # The *_read objects are all callable, returning the next
    ...         # chunk of data for that sensor, if any.
    ...         audio=audio_sensor_read,
    ...         light=light_sensor_read,
    ...         movement=movement_sensor_read,
    ...         # The next
    ...         should_turn_movement_sensor_on = lambda audio, light: vol(audio) * light > 50000,
    ...         human_presence_score = lambda audio, light, movement: movement and sum([vol(audio), light]),
    ...         should_notify = lambda human_presence_score: human_presence_score and human_presence_score > 700,
    ...         notify = lambda should_notify: print('someone is there') if should_notify else None
    ...     )
    ...
    >>>
    >>> si = make_a_slabs_iter()
    >>> next(si)  # doctest: +NORMALIZE_WHITESPACE
    {'audio': [1, 2, 3],
     'light': 126,
     'movement': None,
     'should_turn_movement_sensor_on': False,
     'human_presence_score': None,
     'should_notify': None,
     'notify': None}
    >>> next(si)  # doctest: +NORMALIZE_WHITESPACE
    {'audio': [-96, 87, -92],
     'light': 501,
     'movement': None,
     'should_turn_movement_sensor_on': True,
     'human_presence_score': None,
     'should_notify': None,
     'notify': None}
    >>> next(si)  # doctest: +NORMALIZE_WHITESPACE
    someone is there
    {'audio': [320, -96, 99],
     'light': 523,
     'movement': True,
     'should_turn_movement_sensor_on': True,
     'human_presence_score': 731.1353726143957,
     'should_notify': True,
     'notify': None}

    If you ask for the next slab, you'll get a `StopIteration` (raised by the mocked
    sources since they reached the end of their iterators).

    >>> next(si)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
      ...
    StopIteration

    That said, if you iterate through a `SlabsIter` that handles the `StopIteration`
    exception (it does by default), you'll reach the end of you iteration gracefully.

    >>> si = make_a_slabs_iter()
    >>> for slab in si:
    ...     pass
    someone is there
    >>> si = make_a_slabs_iter()
    >>> slabs = list(si)  # gather all the slabs
    someone is there
    >>> len(slabs)
    3
    >>> slabs[-1]  # doctest: +NORMALIZE_WHITESPACE
    {'audio': [320, -96, 99],
     'light': 523,
     'movement': True,
     'should_turn_movement_sensor_on': True,
     'human_presence_score': 731.1353726143957,
     'should_notify': True,
     'notify': None}

    Note that ``Slabs`` uses a "scope" to store the intermediate results of the
    computation. This scope is a `dict` by default, but you can pass any
    ``MutableMapping`` to the ``scope_factory`` argument. This means that you can use
    other means to store intermediate results simply by wrapping them in a
    MutableMapping. For example, you could use message broker such as Redis to
    store the intermediate results, and have the components read and write to it.

    To help you with this, check out the `dol <https://pypi.org/project/dol/>`_
    and `py2store <https://pypi.org/project/py2store/>`_ libraries.

    """

    _output_of_context_enter = None

    def __init__(
        self,
        handle_exceptions: HandledExceptionsMapSpec = DFLT_INTERRUPT_EXCEPTIONS,
        scope_factory: Callable[[], MutableMapping] = dict,
        **components,
    ):
        _validate_components(components)
        self.components = components
        self.handle_exceptions = _get_handle_exceptions(handle_exceptions)
        self.scope_factory = scope_factory
        self._handled_exception_types = tuple(self.handle_exceptions)
        self.sigs = {
            name: Sig.sig_or_default(func) for name, func in self.components.items()
        }
        self.context = ContextFanout(**components)

    def _call_on_scope(self, scope: MutableMapping):
        """Calls the components 1 by 1, sourcing inputs and writing outputs in scope"""
        # for each component
        for name, component in self.components.items():
            # call the component using scope to source any arguments it needs
            # and write the result in scope, under the component's name.
            scope[name] = _call_from_dict(scope, component, self.sigs[name])
        return scope

    def __next__(self):
        """Get the next slab by calling _call_on_scope on an new empty scope.
        At least one of the components will have to be argument-less and provide
        some data for other components to get their inputs from, if any are needed.
        """
        return self._call_on_scope(scope=self.scope_factory())

    # TODO: Extend the flow control capabilities of execption handling
    #   (see https://github.com/i2mint/meshed/discussions/49)
    def __iter__(self):
        """Iterates over slabs until a handle exception is raised."""
        with self:  # enter all the contexts that need to be entered
            while True:  # loop until you encounter a handled exception
                try:
                    yield next(self)
                except self._handled_exception_types as exc_val:
                    handler_output = _handle_exception(
                        self, exc_val, self.handle_exceptions
                    )
                    # break, unless the handler tells us not to
                    if handler_output is not do_not_break:
                        self.exit_value = handler_output  # remember, in case useful
                        break

    def open(self):
        self._output_of_context_enter = self.context.__enter__()
        return self

    def close(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        return self._output_of_context_enter.__exit__(exc_type, exc_val, exc_tb)

    def run(self):
        for _ in self:
            pass

    __enter__ = open
    __exit__ = close
    __call__ = run

    @classmethod
    def from_func_nodes(
        cls,
        func_nodes: Iterable[FuncNode],
        *,
        handle_exceptions: HandledExceptionsMapSpec = DFLT_INTERRUPT_EXCEPTIONS,
        scope_factory: Callable[[], MutableMapping] = dict,
    ):
        """Make a Slabs object from a list of functions and/or FuncNodes, DAG, ..."""
        if hasattr(func_nodes, "func_nodes"):
            func_nodes = func_nodes.func_nodes
        else:
            func_nodes = list(ensure_func_nodes(func_nodes))
        assert all(
            list(fn.bind.keys()) == list(fn.bind.values()) for fn in func_nodes
        ), (
            "You can't use `from_func_nodes` (yet) your binds are not trivial. "
            "That is, if any of your functions' arguments have different names than "
            "the var nodes they're bound to"
        )
        # TODO: Make it work for non-trivial binds by using i2.wrapper
        components = {n.out: n.func for n in func_nodes}
        return cls(
            handle_exceptions=handle_exceptions,
            scope_factory=scope_factory,
            **components,
        )

    from_dag = from_func_nodes  # TODO: Have a vote if we want this alias.

    def to_func_nodes(self) -> Iterable[FuncNode]:
        for name, func in self.components.items():
            yield FuncNode(func, name=name, out=name)

    def to_dag(self) -> DAG:
        return DAG(list(self.to_func_nodes()))

    # TODO: Add @wraps(dot_digraph_body) to have DAG.dot_digraph signature
    def dot_digraph(self, *args, **kwargs):
        """
        Returns a dot_digraph of the DAG of the SlabsIter (see ``DAG.dot_digraph``)
        """
        dag = self.to_dag()
        return dag.dot_digraph()


SlabsIter = Slabs  # for backward compatibility


# --------------------------------------------------------------------------------------
# Slabs tools and helpers

from functools import wraps


def conditional_sentinel(
    condition_func: Callable[[tuple, dict], bool], sentinel: Any = None
):
    """Decorator that returns sentinel based on a user-defined condition.

    The condition function should take the arguments and keyword arguments of the
    decorated function as input and return a boolean value. If the condition is met,
    the sentinel value is returned instead of the decorated function's return value.

    Args:
        condition_func (Callable): A function that takes the arguments and keyword
            arguments of the decorated function as input and returns a boolean value.
        sentinel (Any): The value to return if the condition is met.

    >>> division_by_zero = lambda args, kwargs: (
    ...     (len(args) >= 2 and args[1] == 0) or
    ...     kwargs.get('y') == 0
    ... )
    >>>
    >>> @conditional_sentinel(division_by_zero, sentinel=0)
    ... def safe_division(x, y):
    ...     return x / y
    ...
    >>> safe_division(10, 2)
    5.0
    >>> safe_division(10, 0)
    0
    >>>

    See also `output_none_if_none_arguments`, made from `conditional_sentinel`:

    >>> @output_none_if_none_arguments
    ... def foo(x, y):
    ...     return x + y
    >>>
    >>> foo(1, 2)
    3
    >>> assert foo(None, None) is None

    """

    def decorator(func):
        """Inner decorator that takes the function and returns a wrapper."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function that checks the condition before calling func."""
            if condition_func(args, kwargs):
                return sentinel
            return func(*args, **kwargs)

        return wrapper

    return decorator


def all_arguments_are_none(args, kwargs):
    """Return True if all arguments are None."""
    return all(arg is None for arg in args) and all(
        val is None for val in kwargs.values()
    )


output_none_if_none_arguments = conditional_sentinel(
    all_arguments_are_none, sentinel=None
)

output_none_if_none_arguments.__doc__ = """
Decorator that returns None if all arguments are None.

>>> @output_none_if_none_arguments
... def foo(x, y):
...     return x + y
>>>
>>> foo(1, 2)
3
>>> assert foo(None, None) is None
"""
```

## tests/__init__.py

```python
"""Where test stuff is"""

from meshed.tests.objects_for_testing import (
    dag_plus_and_times,
    dag_plus_and_times_ext,
    dag_plus_times_minus,
    dag_plus_times_minus_partial,
)
```

## tests/objects_for_testing.py

```python
"""Objects for testing"""

from inspect import signature
from meshed.dag import DAG
from meshed import FuncNode


def f(a, b):
    return a + b


def g(a_plus_b, d):
    return a_plus_b * d


# here we specify that the output of f will be injected in g as an argument for the
# parameter a_plus_b
f_node = FuncNode(func=f, out="a_plus_b")
g_node = FuncNode(func=g)
dag_plus_and_times = DAG((f_node, g_node))
assert dag_plus_and_times(a=1, b=2, d=3) == 9


# we can do more complex renaming as well, for example here we specify that the value
# for b is also the value for d,
# resulting in the dag being now 2 variable dag
f_node = FuncNode(func=f, out="a_plus_b")
g_node = FuncNode(func=g, bind={"d": "b"})
dag_plus_and_times_ext = DAG((f_node, g_node))
assert dag_plus_and_times_ext(a=1, b=2) == 6


def f(a, b):
    return a + b


def g(c, d=4):
    return c * d


def h(ff, gg=42):
    return gg - ff


dag_plus_times_minus = DAG(
    [
        FuncNode(f, out="f_out", name="f"),
        FuncNode(g, out="g_out", name="g", func_label="The G Node"),
        FuncNode(h, bind={"ff": "f_out", "gg": "g_out"}),
    ]
)
dag_plus_times_minus.__doc__ = """
A three node DAG with a variety of artifacts 
(non-default out, bind, and func_label, as well as
a defaulted root node and a defaulted middle node)
"""

dag_plus_times_minus_partial = dag_plus_times_minus.partial(c=3, a=1)
assert dag_plus_times_minus_partial(b=5, d=6) == 12
assert str(signature(dag_plus_times_minus_partial)) == "(b, a=1, c=3, d=4)"
```

## tests/test_base.py

```python
from meshed.makers import code_to_dag


@code_to_dag
def dag_01():
    b = f(a)
    c = g(a)
    d = h(b, c)


@code_to_dag
def dag_02():
    b = f(a)
    c = g(x=a)
    d = h(y=b, c=c)


_string01 = """a -> f -> b
a -> g -> c
b,c -> h -> d"""

_string02 = """a -> f -> b
x=a -> g -> c
y=b,c -> h -> d"""


def test_synopsis_string():
    s11 = "\n".join([fn.synopsis_string() for fn in dag_01.func_nodes])
    s21 = "\n".join(
        [fn.synopsis_string(bind_info="hybrid") for fn in dag_01.func_nodes]
    )
    assert s11 == _string01 == s21
    s22 = "\n".join(
        [fn.synopsis_string(bind_info="hybrid") for fn in dag_02.func_nodes]
    )
    assert s22 == _string02

    from meshed.tests.objects_for_testing import dag_plus_times_minus

    last_fnode = dag_plus_times_minus.func_nodes[-1]
    assert last_fnode.synopsis_string(bind_info="hybrid") == (
        "ff=f_out,gg=g_out -> h_ -> h"
    )
    assert last_fnode.synopsis_string(bind_info="var_nodes") == (
        "f_out,g_out -> h_ -> h"
    )
    assert last_fnode.synopsis_string(bind_info="params") == ("ff,gg -> h_ -> h")
```

## tests/test_caching.py

```python
"""Tests for caching module"""


def test_lazy_props():
    # Note: LazyProps isn't used at the time of writing this (2022-05-20) so if
    # test fails can (maybe) remove.
    from meshed.caching import LazyProps
    from i2 import LiteralVal

    # TODO: Why doesn't it work with dataclasses?
    # from dataclasses import dataclass
    # @dataclass
    class Funnel(LazyProps):
        #     impressions: int = 1000,
        #     cost_per_impression: float = 0.001,
        #     click_per_impression: float = 0.02,  # aka click through rate
        #     sales_per_click: float = 0.05,
        #     revenue_per_sale: float = 100.00

        def __init__(
            self,
            impressions: int = 1000,
            cost_per_impression: float = 0.001,
            click_per_impression: float = 0.02,  # aka click through rate
            sales_per_click: float = 0.05,
            revenue_per_sale: float = 100.00,
        ):  # aka average basket value
            self.impressions = impressions
            self.cost_per_impression = cost_per_impression
            self.click_per_impression = click_per_impression
            self.sales_per_click = sales_per_click
            self.revenue_per_sale = revenue_per_sale

        def cost(self):
            return self.impressions * self.cost_per_impression

        def clicks(self):
            return self.impressions * self.click_per_impression

        def sales(self):
            return self.clicks * self.sales_per_click

        def revenue(self):
            return self.sales * self.revenue_per_sale

        def profit(self):
            return self.revenue - self.cost

        @LiteralVal  # Meaning "leave this attribute as is (i.e. don't make it a lazy prop)"
        def leave_this_alone(self, a, b):
            return a + b

    f = Funnel(impressions=100, sales_per_click=0.15)
    assert f.revenue == 30.0
    assert f.leave_this_alone(1, 2) == 3

    f = Funnel(click_per_impression=0.04)
    assert (f.revenue, f.cost, f.profit) == (200.0, 1.0, 199.0)
```

## tests/test_ch_funcs.py

```python
import meshed as ms
import pytest

import meshed.base
import meshed.util
from meshed.dag import ch_funcs, _validate_func_mapping
from meshed.tests.objects_for_testing import f, g
from meshed.base import compare_signatures
from i2 import Sig
from typing import NamedTuple


@pytest.fixture
def example_func_nodes():

    funcs = [f, g]
    result = meshed.base._mk_func_nodes(funcs)
    return result


@pytest.fixture
def example_func_mapping():
    mapping = {"f_": f, "g_": g}
    return mapping


def test_ch_funcs_no_change(example_func_nodes):
    funcs = [f, g]
    nodes = list(example_func_nodes)
    names = [node.name for node in nodes]

    dummy_mapping = dict(zip(names, funcs))

    new_dag = ch_funcs(
        func_nodes=nodes,
        func_mapping=dummy_mapping,
    )
    new_nodes = new_dag().func_nodes
    assert nodes == new_nodes


class FlagWithMessage(NamedTuple):
    flag: bool
    msg: str = ""


# This function is used to give a more detailed report on
# mismatched signatures
# the same can be done by tweaking ch_func_node_func
# and its "alternative" param
def validate_func_mapping_on_signatures(func_mapping, func_nodes):
    """
    This function is used to give a more detailed report on
    mismatched signatures
    The same can be done by tweaking ch_func_node_func
    and its "alternative" param
    """
    from meshed import DAG

    _validate_func_mapping(func_mapping, func_nodes)
    d = dict()
    dag = DAG(func_nodes)
    for key, func in func_mapping.items():
        if fnode := dag._func_node_for.get(key, None):
            old_func = fnode.func

            if compare_signatures(old_func, func):
                result = FlagWithMessage(flag=True)
            else:
                msg = f"Signatures disagree for key={key}"
                result = FlagWithMessage(flag=False, msg=msg)

        else:
            msg = f"No funcnode matching the key {key}"
            result = FlagWithMessage(flag=False, msg=msg)
        d[key] = result
    all_flags_true = all(item.flag for item in d.values())
    return all_flags_true, d


def test_validate_func_mapping_based_on_signatures(
    example_func_nodes, example_func_mapping
):
    nodes = list(example_func_nodes)
    # funcs = [f, g]
    func_mapping = example_func_mapping
    result = validate_func_mapping_on_signatures(func_mapping, nodes)
    expected = (
        True,
        {
            "f_": FlagWithMessage(flag=True, msg=""),
            "g_": FlagWithMessage(flag=True, msg=""),
        },
    )
    assert result == expected


def test_validate_bind_attributes():
    """
    in ch_func_node_func: validate compatibility (not equality of sigs)
    we cannot use call_compatibility
    rename everything
    https://github.com/i2mint/i2/issues/47
    """
    pass
```

## tests/test_components.py

```python
"""Tests for components.py"""

from meshed.components import Itemgetter, AttrGetter


def test_extractors():
    def data1():
        return {0: "start", 1: "stop", "bob": "alice"}

    first_extraction = Itemgetter([0, 1], name="first_extraction", input_name="data1")
    second_extraction = Itemgetter("bob", name="second_extraction", input_name="data1")

    from meshed import DAG

    dag1 = DAG([data1, first_extraction, second_extraction])

    dag1.synopsis_string() == """ -> data1_ -> data1
data -> first_extraction_ -> first_extraction
data -> second_extraction_ -> second_extraction"""

    assert dag1() == (("start", "stop"), "alice")

    def data2():
        from collections import namedtuple

        return namedtuple("data", ["one", "two", "bob"])("start", "stop", "alice")

    # Note: data produces a namedtuple looking like this
    d = data2()
    assert (d.one, d.two, d.bob) == ("start", "stop", "alice")

    extractor_1 = AttrGetter(["one", "two"], name="one_and_two", input_name="data2")
    extractor_2 = AttrGetter("bob", name="bob", input_name="data2")

    dag2 = DAG([data2, extractor_1, extractor_2])

    assert dag2() == (("start", "stop"), "alice")

    dag2.synopsis_string() == """ -> data2_ -> data2
data -> one_and_two_ -> one_and_two
data -> bob_ -> bob"""
```

## tests/test_composition.py

```python
"""Test composition"""

import pytest

from meshed.composition import line_with_dag


def test_line_with_dag():
    """
    Very simple test of a basic usage of line_with_dag
    """

    def f(x):
        return x + 1

    def g(y):
        return 2 * y

    d = line_with_dag(f, g)
    assert d(2) == 6
```

## tests/test_dag.py

```python
"""Test dags"""

import pytest

from meshed.makers import code_to_dag

# Note: This is just for the linter not to complain about the code_to_dag dag
mult, add, subtract, w, ww, www, x, y, z = map(lambda x: x, [None] * 9)


@code_to_dag()
def mult_add_subtract_dag():
    x = mult(w, ww)
    y = add(x, www)
    z = subtract(x, y)


def pass_on_tuple(a, b):
    return a, b


def add(x, y):
    return x + y


def _expand_and_sum_dag():
    x, y = pass_on_tuple(w, ww)
    result = add(x, y)


expand_and_sum_dag = code_to_dag(_expand_and_sum_dag, func_src=locals())


def test_code_to_dag_itemgetter():
    assert expand_and_sum_dag(2, 3) == 5


def test_dag_operations():
    # from meshed.makers import code_to_dag
    #
    # # Note: This is just for the linter not to complain about the code_to_dag dag
    # mult, add, subtract, w, ww, www, x, y, z = map(lambda x: x, [None] * 9)
    #
    # @code_to_dag()
    # def dag():
    #     x = mult(w, ww)
    #     y = add(x, www)
    #     z = subtract(x, y)

    dag = mult_add_subtract_dag

    from i2 import Sig

    assert str(Sig(dag)) == "(w, ww, www)"

    assert (
        dag(1, 2, 3) == "subtract(x=mult(w=1, ww=2), y=add(x=mult(w=1, ww=2), www=3))"
    )

    dag = dag.ch_funcs(
        mult=lambda w, ww: w * ww,
        add=lambda x, www: x + www,
        subtract=lambda x, y: x - y,
    )

    assert str(Sig(dag)) == "(w, ww, www)"
    assert dag(1, 2, 3) == -3


def test_funcnode_bind():
    """
    Test the renaming of arguments and output of functions using FuncNode and its
    effect on DAG
    """
    from meshed.dag import DAG
    from meshed import FuncNode

    def f(a, b):
        return a + b

    def g(a_plus_b, d):
        return a_plus_b * d

    # here we specify that the output of f will be injected in g as an argument for the parameter a_plus_b
    f_node = FuncNode(func=f, out="a_plus_b")
    g_node = FuncNode(func=g)
    dag = DAG((f_node, g_node))
    assert dag(a=1, b=2, d=3) == 9

    # we can do more complex renaming as well, for example here we specify that the value for b is also the value for d,
    # resulting in the dag being now 2 variable dag
    f_node = FuncNode(func=f, out="a_plus_b")
    g_node = FuncNode(func=g, bind={"d": "b"})
    dag = DAG((f_node, g_node))
    assert dag(a=1, b=2) == 6


def test_iterize_dag():
    def f(a, b=2):
        return a + b

    def g(f, c=3):
        return f * c

    from meshed import DAG

    d = DAG([f, g])
    # d.dot_digraph()  # smoke testing the digraph

    assert (  # if you needed to apply d to an iterator, you'd normally do this
        list(map(d, [1, 2, 3]))
    ) == ([9, 12, 15])

    # But if you need a function that "looks like" d, but is "vectorized" (really
    # iterized) version...
    from functools import partial
    from inspect import signature

    def iterize(func):
        _iterized_func = partial(map, func)
        _iterized_func.__signature__ = signature(func)
        return _iterized_func

    di = iterize(d)
    # di has the same signature as d:
    assert signature(di) == signature(d)
    assert (list(di([1, 2, 3]))) == ([9, 12, 15])  # But works with a being an iterator

    # Note that di will return an iterator that needs to be "consumed" (here with list)
    # That is, no matter what the (iterable) type of the input is.
    # If you wanted to systematically get your output as a list (or tuple, or set,
    # or numpy.array),
    # there's several choices...

    # You could use i2.Pipe

    from i2 import Pipe

    di_list = Pipe(di, list)
    assert di_list([1, 2, 3]) == [9, 12, 15]


def test_binding_to_a_root_node():
    """
    See: https://github.com/i2mint/meshed/issues/7
    """
    from meshed.dag import DAG
    from meshed.util import ValidationError
    from meshed import FuncNode

    def f(a, b):
        return a + b

    def g(a_plus_b, d):
        return a_plus_b * d

    # we bind d to b, and it works!
    f_node = FuncNode(func=f, out="a_plus_b")
    g_node = FuncNode(func=g, bind={"d": "b"})
    dag = DAG((f_node, g_node))
    assert dag(a=1, b=2) == 6

    # but if b and d are not aligned on all other parameter props besides name
    # (kind, default, annotation), then we get an error

    def gg(a_plus_b, d=4):
        return a_plus_b * d

    gg_node = FuncNode(func=gg, bind={"d": "b"})

    with pytest.raises(ValidationError) as e_info:
        _ = DAG((f_node, gg_node))

    assert "didn't have the same default" in e_info.value.args[0]

    # There's several solutions to this.
    # First, we can simply prepare the functions so that the defaults align.
    # The following shows how to do this in two different ways

    # 1: "Manually"
    def ff(a, b=4):
        return f(a, b)

    ff_node = FuncNode(func=ff, out="a_plus_b")
    dag = DAG((ff_node, gg_node))
    assert dag(a=1, b=2) == 6

    # 2: With i2.Sig
    from i2 import Sig

    give_default_to_b = lambda func: Sig(func).ch_defaults(b=4)(func)
    ff_node = FuncNode(func=give_default_to_b(f), out="a_plus_b")
    dag = DAG((ff_node, gg_node))
    assert dag(a=1, b=2) == 6
    # And if you don't specify b, it has that default you set!
    assert dag(a=1) == 20

    # Second, we could specify a different "merging policy" (the function that
    # determines how to resolve the issue of several params with the same name
    # (or binding) that conflict on some prop (kind, default and/or annotation)

    # Before we go there though, let's show that default is not the only problem.
    # If the annotation, or the kind are different, we also run in to the same problem
    # (and solution to it)
    def f(a, b):
        return a + b

    def ggg(a_plus_b, d: int):  # note that d has no default, but an annotation
        return a_plus_b * d

    ggg_node = FuncNode(func=ggg, bind={"d": "b"})
    with pytest.raises(ValidationError) as e_info:
        _ = DAG((f_node, ggg_node))
    assert "didn't have the same annotation" in e_info.value.args[0]

    # Solution (with i2.Sig)

    give_annotation_to_b = lambda func: Sig(func).ch_annotations(b=int)(func)
    ff_node = FuncNode(func=give_annotation_to_b(f), out="a_plus_b")
    dag = DAG((ff_node, ggg_node))
    assert dag(a=1, b=2) == 6

    # The other solution to the parameter property misalignment is to tell the DAG
    # constructor what we want it to do with conflicts. For example, just ignore them.
    # (Not a good general policy though!)

    from meshed.dag import conservative_parameter_merge
    from functools import partial

    first_wins_all_merger = partial(
        conservative_parameter_merge,
        same_kind=False,
        same_default=False,
        same_annotation=False,
    )

    def f(a, b: int, /):
        return a + b

    def g(a_plus_b, d: float = 4):
        return a_plus_b * d

    lenient_dag_maker = partial(DAG, parameter_merge=first_wins_all_merger)

    f_node = FuncNode(func=f, out="a_plus_b")
    g_node = FuncNode(func=g, bind={"d": "b"})
    dag = lenient_dag_maker([f_node, g_node])
    assert dag(1, 2) == 6
    # Note we can't do dag(a=1, b=2) since (like f) it's position-only.
    # Indeed the dag inherits its arguments' properties from the functions composing it, in this case f

    # Resolving conflicts this way isn't the best general policy (that's why it's not
    # the default).
    # In production, it's advised to implement a more careful merging policy, possibly
    # specifying (in the `parameter_merge` callable itself) explicitly what to do for
    # every situation that we encounter.


def test_dag_partialize():
    from functools import partial
    from i2 import Sig
    from meshed import DAG
    from inspect import signature

    def foo(a, b):
        return a - b

    f = DAG([foo])
    assert str(Sig(f)) == "(a, b)"

    # if we give ``b`` a default:
    ff = f.partial(b=9)
    assert str(Sig(ff)) == "(a, b=9)"
    # note that the Sig of the partial of foo is '(a, *, b=9)' though
    assert str(Sig(partial(foo, b=9))) == "(a, *, b=9)"
    assert ff(10) == ff(a=10) == 1

    # if we give ``a`` (the first arg) a default but not ``b`` (the second arg)
    fff = f.partial(a=4)  # fixing a, which is before b
    # note that this fixing a reorders the parameters (so we have a valid signature!)
    assert str(Sig(fff)) == "(b, a=4)"

    fn = fff.func_nodes[0]
    assert fn.call_on_scope(dict(b=3)) == 1

    def f(a, b):
        return a + b

    def g(c, d=4):
        return c * d

    def h(f, g):
        return g - f

    larger_dag = DAG([f, g, h])

    new_dag = larger_dag.partial(c=3, a=1)
    assert new_dag(b=5, d=6) == 12
    assert str(signature(new_dag)) == "(b, a=1, c=3, d=4)"
```

## tests/test_dag_2.py

```python
import meshed as ms
import pytest

import meshed.base
import meshed.util


@pytest.fixture
def example_func_nodes():
    def f(a=1, b=2):
        return 12

    def g(c=3):
        return 42

    func_nodes = [f, g]
    result = meshed.base._mk_func_nodes(func_nodes)
    return result


def test_find_first_free_name():
    prefix = "ab"
    exclude_names = ("cd", "lm", "ab", "ab__0", "ef")
    assert (
        meshed.util.find_first_free_name(
            prefix, exclude_names=exclude_names, start_at=0
        )
        == "ab__1"
    )


def test_mk_func_name():
    def myfunc1(a=1, b=3, c=1):
        return a + b * c

    assert meshed.util.mk_func_name(myfunc1, exclude_names=("myfunc1")) == "myfunc1__2"


def test_arg_names():
    def myfunc1(a=1, b=3, c=1):
        return a + b * c

    args_list = meshed.util.arg_names(myfunc1, "myfunc1", exclude_names=("a", "b"))
    assert args_list == ["myfunc1__a", "myfunc1__b", "c"]


def test_named_partial():
    f = meshed.util.named_partial(print, sep="\\n")
    assert f.__name__ == "print"
    g = meshed.util.named_partial(print, sep="\\n", __name__="now_partial_has_a_name")
    assert g.__name__ == "now_partial_has_a_name"


def test_hook_up():
    def formula1(w, /, x: float, y=1, *, z: int = 1):
        return ((w + x) * y) ** z

    d = {}
    f = ms.dag.hook_up(formula1, d)
    d.update(w=2, x=3, y=4)
    f()
    assert d == {"w": 2, "x": 3, "y": 4, "formula1": 20}
    d.clear()
    d.update(w=1, x=2, y=3)
    f()
    assert d["formula1"] == 9


def test_complete_dict_with_iterable_of_required_keys():
    d = {"a": "A", "c": "C"}
    meshed.base._complete_dict_with_iterable_of_required_keys(d, "abc")
    assert d == {"a": "A", "c": "C", "b": "b"}


def test_inverse_dict_asserting_losslessness():
    d = {"w": 2, "x": 3, "y": 4, "formula1": 20}
    d_inv = meshed.util.inverse_dict_asserting_losslessness(d)
    assert d_inv == {2: "w", 3: "x", 4: "y", 20: "formula1"}


def test_mapped_extraction():
    extracted = meshed.base._mapped_extraction(
        src={"A": 1, "B": 2, "C": 3}, to_extract={"a": "A", "c": "C", "d": "D"}
    )
    assert dict(extracted) == {"a": 1, "c": 3}


def test_underscore_func_node_names_maker():
    def func_1():
        pass

    name, out = meshed.base.underscore_func_node_names_maker(
        func_1, name="init_func", out="output_name"
    )
    assert name, out == ("init_func", "output_name")
    assert meshed.base.underscore_func_node_names_maker(func_1) == (
        "func_1_",
        "func_1",
    )
    assert meshed.base.underscore_func_node_names_maker(func_1, name="init_func") == (
        "init_func",
        "_init_func",
    )
    assert meshed.base.underscore_func_node_names_maker(func_1, out="output_name") == (
        "func_1",
        "output_name",
    )


def test_duplicates():
    assert meshed.base.duplicates("abbaaeccf") == ["a", "b", "c"]


def test_FuncNode():
    def multiply(x, y):
        return x * y

    item_price = 3.5
    num_of_items = 2
    func_node = meshed.base.FuncNode(
        func=multiply,
        bind={"x": "item_price", "y": "num_of_items"},
    )
    assert (
        str(func_node) == "FuncNode(x=item_price,y=num_of_items -> multiply_ -> "
        "multiply)"
    )
    scope = {"item_price": 3.5, "num_of_items": 2}
    assert func_node.call_on_scope(scope) == 7.0
    assert scope == {"item_price": 3.5, "num_of_items": 2, "multiply": 7.0}
    # Give a name to output
    assert (
        str(
            meshed.base.FuncNode(
                func=multiply,
                name="total_price",
                bind={"x": "item_price", "y": "num_of_items"},
            )
        )
        == "FuncNode(x=item_price,y=num_of_items -> total_price -> _total_price)"
    )
    # rename the function and the output
    assert (
        str(
            meshed.base.FuncNode(
                func=multiply,
                name="total_price",
                bind={"x": "item_price", "y": "num_of_items"},
                out="daily_expense",
            )
        )
        == "FuncNode(x=item_price,y=num_of_items -> total_price -> daily_expense)"
    )


def test_mk_func_nodes():
    def f(a=1, b=2):
        return 12

    def g(c=3):
        return 42

    func_nodes = [f, g]
    result = meshed.base._mk_func_nodes(func_nodes)
    assert str(list(result)) == "[FuncNode(a,b -> f_ -> f), FuncNode(c -> g_ -> g)]"


def test_func_nodes_to_graph_dict(example_func_nodes):
    fnodes = example_func_nodes
    result = meshed.base._func_nodes_to_graph_dict(fnodes)
    assert True
```

## tests/test_dag_defaults.py

```python
import pytest
from pytest import fixture
from meshed import DAG
from i2 import Sig


@fixture
def foo():
    def func(x, y=1):
        return x + y

    return func


def test_dag_with_defaults(foo):
    foo_dag = DAG([foo])
    assert foo_dag(0) == 1
    bar_dag = Sig(lambda x, y=2: None)(foo_dag)  # Bug: does not change dag.sig!!
    # bar_dag.sig = Sig(bar_dag)  # changed it manually
    assert str(Sig(bar_dag)) == "(x, y=2)"
    assert bar_dag(0) == 2  # Correct result after above change in dag._call
    assert True
```

## tests/test_dag_variadics.py

```python
import pytest
from pytest import fixture
from meshed import DAG
from i2 import Sig


@fixture
def foo():
    def func(x, y=1):
        return x + y

    return func


def test_addition_variadics():
    def foo(w, /, x: float, y="YY", *, z: str = "ZZ", **rest):
        pass

    sig = Sig(foo)
    # res = sig.map_arguments(
    #    (11, 22, "you"), dict(z="zoo", other="stuff"), post_process=True
    # )
    # assert res == "{'w': 11, 'x': 22, 'y': 'you', 'z': 'zoo', 'other': 'stuff'}"
    assert True
```

## tests/test_getitem.py

```python
import pytest


from collections import Counter
from meshed import FuncNode
from meshed.dag import DAG
from pytest import fixture


def X_test(train_test_split):
    return train_test_split[1]


def y_test(train_test_split):
    return train_test_split[3]


def truth(y_test):  # to link up truth and test_y
    return y_test


def confusion_count(prediction, truth):
    """Get a dict containing the counts of all combinations of predicction and corresponding truth values."""
    return Counter(zip(prediction, truth))


def prediction(predict_proba, threshold):
    """Get an array of predictions from thresholding the scores of predict_proba array."""
    return list(map(lambda x: x >= threshold, predict_proba))


def predict_proba(model, X_test):
    """Get the prediction_proba scores of a model given some test data"""
    return model.predict_proba(X_test)


def _aligned_items(a, b):
    """Yield (k, a_value, b_value) triples for all k that are both a key of a and of b"""
    # reason for casting to dict is to make sure things like pd.Series use the right keys.
    # could also use k in a.keys() etc. to solve this.
    a = dict(a)
    b = dict(b)
    for k in a:
        if k in b:
            yield k, a[k], b[k]


def dot_product(a, b):
    """
    >>> dot_product({'a': 1, 'b': 2, 'c': 3}, {'b': 4, 'c': -1, 'd': 'whatever'})
    5
    """
    return sum(ak * bk for _, ak, bk in _aligned_items(a, b))


def classifier_score(confusion_count, confusion_value):
    """Compute a score for a classifier that produced the `confusion_count`, based on the given `confusion_value`.
    Meant to be curried by fixing the confusion_value dict.

    The function is purposely general -- it is not specific to binary classifier outcomes, or even any classifier outcomes.
    It simply computes a normalized dot product, depending on the inputs keys to align values to multiply and
    considering a missing key as an expression of a null value.
    """
    return dot_product(confusion_count, confusion_value) / sum(confusion_count.values())


@fixture
def bigger_dag():
    bigger_dag = DAG(
        [
            classifier_score,
            confusion_count,
            prediction,
            predict_proba,
            X_test,
            y_test,
            truth,
        ]
    )
    return bigger_dag


def test_full_subgraph(bigger_dag):
    result = bigger_dag[["truth", "prediction"]:"confusion_count"]
    expected = "DAG(func_nodes=[FuncNode(prediction,truth -> confusion_count_ -> confusion_count)], name=None)"
    assert result.__repr__() == expected
```

## tests/test_itools.py

```python
import meshed as ms
import pytest


@pytest.fixture
def example_graph():
    return dict(a=["c"], b=["c", "c"], c=["a", "b", "d", "e"], d=["c"], e=["c", "z"])


@pytest.fixture
def graph_children():
    return {0: [1, 2], 1: [2, 3, 4], 2: [1, 4], 3: [4]}


@pytest.fixture
def digraph_children():
    return {0: [1, 2], 1: [2, 3, 4], 2: [4], 3: [4]}


@pytest.fixture
def graph_dict():
    return dict(a="c", b="ce", c="abde", d="c", e=["c", "z"], f={})


def test_add_edge(example_graph):
    g = example_graph
    ms.itools.add_edge(g, "d", "a")
    assert g == {
        "a": ["c"],
        "b": ["c", "c"],
        "c": ["a", "b", "d", "e"],
        "d": ["c", "a"],
        "e": ["c", "z"],
    }
    ms.itools.add_edge(g, "t", "y")
    assert g == {
        "a": ["c"],
        "b": ["c", "c"],
        "c": ["a", "b", "d", "e"],
        "d": ["c", "a"],
        "e": ["c", "z"],
        "t": ["y"],
    }


def test_edges(example_graph):
    g = example_graph
    assert sorted(ms.itools.edges(g)) == [
        ("a", "c"),
        ("b", "c"),
        ("b", "c"),
        ("c", "a"),
        ("c", "b"),
        ("c", "d"),
        ("c", "e"),
        ("d", "c"),
        ("e", "c"),
        ("e", "z"),
    ]


def test_nodes(example_graph):
    g = example_graph
    assert sorted(ms.itools.nodes(g)) == ["a", "b", "c", "d", "e", "z"]


def test_has_node():
    g = {0: [1, 2], 1: [2]}
    assert ms.itools.has_node(g, 0)
    assert ms.itools.has_node(g, 2)
    assert not ms.itools.has_node(g, 2, check_adjacencies=False)
    gg = {0: [1, 2], 1: [2], 2: []}
    assert ms.itools.has_node(gg, 2, check_adjacencies=False)


def test_successors(graph_children):
    g = graph_children
    assert set(ms.itools.successors(g, 1)) == {1, 2, 3, 4}
    assert set(ms.itools.successors(g, 3)) == {4}
    assert set(ms.itools.successors(g, 4)) == set()


def test_predecessors(graph_children):
    g = graph_children
    assert set(ms.itools.predecessors(g, 4)) == {0, 1, 2, 3}
    assert set(ms.itools.predecessors(g, 2)) == {0, 1, 2}
    assert set(ms.itools.predecessors(g, 0)) == set()


def test_children(graph_children):
    g = graph_children
    assert set(ms.itools.children(g, [2, 3])) == {1, 4}
    assert set(ms.itools.children(g, [4])) == set()


def test_parents(graph_children):
    g = graph_children
    assert set(ms.itools.parents(g, [2, 3])) == {0, 1}
    assert set(ms.itools.parents(g, [0])) == set()


def test_ancestors(digraph_children):
    g = digraph_children
    assert set(ms.itools.ancestors(g, [2, 3])) == {0, 1}
    assert set(ms.itools.ancestors(g, [0])) == set()


def test_descendants(digraph_children):
    g = digraph_children
    assert set(ms.itools.descendants(g, [2, 3])) == {4}
    assert set(ms.itools.descendants(g, [4])) == set()


def test_root_nodes(graph_dict):
    g = graph_dict
    assert sorted(ms.itools.root_nodes(g)) == ["f"]


def test_root_ancestors():
    graph = {
        "exposed": ["r_"],
        "infect_if_expose": ["r_"],
        "r_": ["r"],
        "r": ["infected_"],
        "vax": ["infected_", "die_"],
        "infection_vax_factor": ["infected_"],
        "infected_": ["infected"],
        "infected": ["die_"],
        "die_if_infected": ["die_"],
        "death_vax_factor": ["die_"],
        "die_": ["die"],
        "die": ["death_toll_"],
        "population": ["death_toll_"],
        "death_toll_": ["death_toll"],
    }

    assert ms.itools.root_ancestors(graph, "r") == {"exposed", "infect_if_expose"}
    assert ms.itools.root_ancestors(graph, "r_") == {"exposed", "infect_if_expose"}
    assert ms.itools.root_ancestors(graph, "infected") == {
        "exposed",
        "infect_if_expose",
        "infection_vax_factor",
        "vax",
    }
    assert ms.itools.root_ancestors(graph, "r die") == {
        "death_vax_factor",
        "die_if_infected",
        "exposed",
        "infect_if_expose",
        "infection_vax_factor",
        "vax",
    }


def test_leaf_nodes(graph_dict):
    g = graph_dict
    assert sorted(ms.itools.leaf_nodes(g)) == ["f", "z"]


def test_isolated_nodes(graph_dict):
    g = graph_dict
    assert set(ms.itools.isolated_nodes(g)) == {"f"}


def test_find_path(graph_dict):
    g = graph_dict
    assert ms.itools.find_path(g, "a", "c") == ["a", "c"]
    assert ms.itools.find_path(g, "a", "b") == ["a", "c", "b"]
    assert ms.itools.find_path(g, "a", "z") == ["a", "c", "b", "e", "z"]


def test_reverse_edges(example_graph):
    g = example_graph
    assert sorted(list(ms.itools.reverse_edges(g))) == [
        ("a", "c"),
        ("b", "c"),
        ("c", "a"),
        ("c", "b"),
        ("c", "b"),
        ("c", "d"),
        ("c", "e"),
        ("d", "c"),
        ("e", "c"),
        ("z", "e"),
    ]


def test_out_degrees(graph_dict):
    g = graph_dict
    assert dict(ms.itools.out_degrees(g)) == (
        {"a": 1, "b": 2, "c": 4, "d": 1, "e": 2, "f": 0}
    )


def test_in_degrees(graph_dict):
    g = graph_dict
    assert dict(ms.itools.in_degrees(g)) == (
        {"a": 1, "b": 1, "c": 4, "d": 1, "e": 2, "f": 0, "z": 1}
    )


def test_copy_of_g_with_some_keys_removed(example_graph):
    g = example_graph
    keys = ["c", "d"]
    gg = ms.itools.copy_of_g_with_some_keys_removed(g, keys)
    assert gg == {"a": ["c"], "b": ["c", "c"], "e": ["c", "z"]}


def test_topological_sort_helper(example_graph):
    g = example_graph
    v = "a"
    stack = ["b", "c"]
    visited = {"e"}
    ms.itools._topological_sort_helper(g, v, visited, stack)
    assert visited == {"a", "b", "c", "d", "e"}
    assert sorted(stack) == ["a", "b", "b", "c", "c", "d"]


def test_topological_sort():
    g = {0: [4, 2], 4: [3, 1], 2: [3], 3: [1]}
    assert list(ms.itools.topological_sort(g)) == [0, 4, 2, 3, 1]


def test_handle_exclude_nodes():
    def f(a=1, b=2, _exclude_nodes=["c"]):
        return f"_exclude_nodes is now a set:{_exclude_nodes}"

    new_f = ms.itools._handle_exclude_nodes(f)
    assert new_f(1) == "_exclude_nodes is now a set:{'c'}"


@pytest.fixture
def simple_graph():
    return dict(a="c", b="cd", c="abd", e="")


def test_edge_reversed_graph(simple_graph):
    g = simple_graph
    assert ms.itools.edge_reversed_graph(g) == {
        "c": ["a", "b"],
        "d": ["b", "c"],
        "a": ["c"],
        "b": ["c"],
        "e": [],
    }
    reverse_g_with_sets = ms.itools.edge_reversed_graph(g, set, set.add)
    assert reverse_g_with_sets == {
        "c": {"a", "b"},
        "d": {"b", "c"},
        "a": {"c"},
        "b": {"c"},
        "e": set(),
    }
    assert ms.itools.edge_reversed_graph(dict(e="", a="e")) == {
        "e": ["a"],
        "a": [],
    }
    assert ms.itools.edge_reversed_graph(dict(a="e", e="")) == {
        "e": ["a"],
        "a": [],
    }
```

## tests/test_makers.py

```python
import pytest

# just to shut the linter up about these
from i2 import Sig
from meshed.makers import code_to_dag


def user_story_01():
    # Sure, linter complains that names are not known, but all we want is valid code.
    # TODO: How to shut the linter up on this?

    # simple function calls
    data_source = get_data_source()  # no inputs
    wfs = make_wfs(data_source)  # one input
    chks = chunker(wfs, chk_size)  # two (positional) inputs

    # verify that we can handle multiple outputs (split)
    train_chks, test_chks = splitter(chks)

    # verify that we can handle k=v inputs (if v is a variable name):
    featurizer_obj = learn_featurizer(featurizer_learner, train_data=train_chks)


def test_user_story_01():
    dag = code_to_dag(user_story_01)
    assert (
        dag.synopsis_string()
        == """ -> get_data_source -> data_source
data_source -> make_wfs -> wfs
wfs,chk_size -> chunker -> chks
chks -> splitter -> train_chks__test_chks
train_chks__test_chks -> train_chks__0 -> train_chks
train_chks__test_chks -> test_chks__1 -> test_chks
featurizer_learner,train_chks -> learn_featurizer -> featurizer_obj"""
    )
    assert str(Sig(dag)) == "(chk_size, featurizer_learner)"


def test_smoke_code_to_dag(src=user_story_01):
    dag = code_to_dag(src)


call, src_to_wf, data_src, chunker, chain, featurizer, model = [object] * 7


def user_story_02():
    wfs = call(src_to_wf, data_src)
    chks_iter = map(chunker, wfs)
    chks = chain(chks_iter)
    fvs = map(featurizer, chks)
    model_outputs = map(model, fvs)
```

## tests/test_meshed_tools.py

```python
"""Test hybrid dag that uses a web service for some functions."""

from meshed import DAG
from meshed.examples import online_marketing_funcs as funcs
from meshed.tools import mk_hybrid_dag, launch_webservice


def test_hybrid_dag(
    dag_funcs=funcs,
    funcs_ids_to_cloudify=["cost", "revenue"],
    input_dict=dict(
        impressions=1000,
        cost_per_impression=0.02,
        click_per_impression=0.3,
        sales_per_click=0.05,
        revenue_per_sale=100,
    ),
):
    """Test hybrid dag that uses a web service for some functions.

    :param dag_funcs: list of dag functions, defaults to funcs
    :type dag_funcs: List[Callable], optional
    :param funcs_ids_to_cloudify: list of function ids to be cloudified, defaults to ['cost', 'revenue']
    :type funcs_ids_to_cloudify: list, optional
    :param input_dict: kwargs, defaults to dict( impressions=1000, cost_per_impression=0.02, click_per_impression=0.3, sales_per_click=0.05, revenue_per_sale=100 )
    :type input_dict: dict, optional
    """
    # The parameters
    dag = DAG(dag_funcs)

    print("Calling mk_hybrid_dag!")
    # Calling mk_hybrid_dag
    hybrid_dag = mk_hybrid_dag(dag, funcs_ids_to_cloudify)

    print("Starting web service!")
    with launch_webservice(hybrid_dag.funcs_to_cloudify) as ws:
        print("Web service started!")
        print("Calling dag and ws_dag!")
        dag_result = dag(**input_dict)
        print(f"dag_result: {dag_result}")
        ws_dag_result = hybrid_dag.ws_dag(**input_dict)
        print(f"ws_dag_result: {ws_dag_result}")
        assert dag_result == ws_dag_result, "Results are not equal!"
        print("Results are equal!")
        print("Done!")
```

## tests/test_util.py

```python
import meshed as ms


def test_provides_on_funcnode():
    from meshed import provides, DAG, FuncNode

    @provides("one")
    def get_one():
        return 1

    @provides("two")
    def make_two():
        return 2

    @provides("two")
    def another_number():
        return 2

    @provides("a_sum", "_")
    def add_one_and_two(one, two):
        return one + two

    dag = DAG([get_one, make_two, add_one_and_two])
    assert dag.synopsis_string() == (
        " -> get_one -> one\n -> make_two -> two\none,two -> add_one_and_two -> a_sum"
    )
    assert dag() == 3


def test_name_of_obj():
    assert ms.util.name_of_obj(map) == "map"
    assert ms.util.name_of_obj([1, 2, 3])
    "list"
    assert ms.util.name_of_obj(print) == "print"
    assert ms.util.name_of_obj(lambda x: x) == "<lambda>"
    from functools import partial

    assert ms.util.name_of_obj(partial(print, sep=",")) == "print"


def test_incremental_str_maker():
    lambda_name = ms.util.incremental_str_maker(str_format="lambda_{:03.0f}")
    assert lambda_name() == "lambda_001"
    assert lambda_name() == "lambda_002"


def test_func_name():
    def my_func(a=1, b=2):
        return a * b

    assert ms.util.func_name(my_func) == "my_func"
    assert ms.util.func_name(lambda x: x).startswith("lambda_")
```

## tests/utils_for_testing.py

```python
"""Make objects for testing fast"""

from meshed.util import mk_place_holder_func
from meshed.dag import *
from i2 import Sig


def parse_names(string):
    return list(map(str.strip, string.split(",")))


def string_to_func(dot_string):
    arg_names, func_name = map(parse_names, dot_string.split("->"))
    assert len(func_name) == 1
    func_name = func_name[0]
    return mk_place_holder_func(arg_names, func_name)


def string_to_func_node(dot_string):
    arg_names, func_name, output_name = map(parse_names, dot_string.split("->"))
    assert len(func_name) == 1

    func_name = func_name[0]
    assert len(output_name) == 1
    output_name = output_name[0]

    func = mk_place_holder_func(arg_names, func_name)
    return FuncNode(func, name=func_name, out=output_name)


def string_to_dag(dot_string):
    """
    >>> dot_string = '''
    ... a, b, c -> d -> e
    ... b, f -> g -> h
    ... a, e -> i -> j
    ... '''
    >>> dag = string_to_dag(dot_string)
    >>> print(dag.synopsis_string())
    a,b,c -> d -> e
    b,f -> g -> h
    a,e -> i -> j

    >>> Sig(dag)
    <Sig (a, b, c, f)>
    >>> sorted(dag(1,2,3,4))
    ['g(b=2, f=4)', 'i(a=1, e=d(a=1, b=2, c=3))']"""
    func_nodes = list(map(string_to_func_node, filter(bool, dot_string.split("\n"))))
    return DAG(func_nodes)
```

## tools.py

```python
"""Tools to work with meshed"""

from collections import namedtuple
from contextlib import contextmanager
from functools import cached_property, partial
import multiprocessing
import os
import time
from typing import List
from collections.abc import Callable
from urllib.parse import urljoin

import i2

from meshed.dag import DAG


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 3030))
API_URL = os.environ.get("API_URL", f"http://localhost:{PORT}")
SERVER = os.environ.get("SERVER", "wsgiref")
OPENAPI_URL = urljoin(API_URL, "openapi")


def find_funcs(dag, func_outs):
    return list(dag.find_funcs(lambda x: x.out in func_outs))


def mk_dag_with_ws_funcs(dag: DAG, ws_funcs: dict) -> DAG:
    """Creates a new DAG with the web service functions.

    :param dag: DAG to be hybridized
    :type dag: DAG
    :param ws_funcs: mapping of web service functions
    :type ws_funcs: dict
    :return: new DAG with the web service functions
    :rtype: DAG
    """
    return dag.ch_funcs(**ws_funcs)


def launch_funcs_webservice(funcs: list[Callable]):
    """Launches a web service application with the specified functions.

    :param funcs: functions to be hosted by the web service
    :type funcs: List[Callable]
    """
    from extrude import mk_api, run_api

    ws_app = mk_api(funcs, openapi=dict(base_url=API_URL))
    run_api(ws_app, host=HOST, port=PORT, server=SERVER)


@contextmanager
def launch_webservice(funcs_to_cloudify, wait_after_start_seconds=10):
    """Context manager to launch a web service application in a separate process."""
    ws = multiprocessing.Process(
        target=launch_funcs_webservice, args=(funcs_to_cloudify,)
    )
    ws.start()
    # TODO: I prefer using a timeout instead of a fixed wait time
    # TODO: Use strand tool for this: https://github.com/i2mint/strand/blob/7443631e9d2486358f0a34ed182e85b6ded5e50c/strand/taskrunning/utils.py#L54
    time.sleep(wait_after_start_seconds)
    yield ws

    ws.terminate()


class CloudFunctions:
    def __init__(self, funcs: list[Callable], openapi_url=OPENAPI_URL, logger=print):
        """Creates a Python dictionary-like object that maps the web service functions to Python functions.

        :param funcs: list of functions hosted by the web service
        :type funcs: List[Callable]
        :param openapi_url: url to get openapi spec, defaults to "http://localhost:{PORT}/openapi"
        :type openapi_url: str, optional
        :param logger: logger function, defaults to print
        :type logger: Callable, optional
        """
        self.funcs = funcs
        self.openapi_url = openapi_url
        self.logger = logger if callable(logger) else lambda x: None

    @cached_property
    def http_client(self):
        from http2py import HttpClient

        try:
            return HttpClient(url=self.openapi_url)
        except Exception:
            self.logger(
                f"Could not connect to {self.openapi_url}. Waiting 10 seconds and trying again."
            )
            time.sleep(10)
            return HttpClient(url=self.openapi_url)

    @cached_property
    def func_names(self):
        return frozenset(f.__name__ for f in self.funcs)

    def __getitem__(self, key):
        """Returns a Python function that calls the web service function.
        HttpClient is queried at the execution of the function.
        """

        @i2.Sig(next(f for f in self.funcs if key == f.__name__))
        def ws_func(*a, **kw):
            self.logger(f"Getting web service for: {key}")
            if (_wsf := getattr(self.http_client, key, None)) is not None:
                self.logger(f"Found web service for: {key}")
                return _wsf(*a, **kw)
            raise KeyError(key)

        return ws_func

    def __contains__(self, key):
        return key in self.func_names

    def __len__(self):
        return len(self.func_names)

    def keys(self):
        return self.func_names

    def values(self):
        return (self[k] for k in self.keys())

    def items(self):
        return ((k, self[k]) for k in self.keys())


def mk_hybrid_dag(dag: DAG, func_ids_to_cloudify: list):
    """Creates a hybrid DAG that uses the web service for the specified functions.

    :param dag: dag to be hybridized
    :type dag: DAG
    :param func_ids_to_cloudify: list of function ids to be cloudified
    :type func_ids_to_cloudify: list
    :return: namedtuple with funcs_to_cloudify, ws_dag and ws_funcs
    :rtype: namedtuple
    """
    funcs_to_cloudify = find_funcs(dag, func_ids_to_cloudify)
    ws_funcs = CloudFunctions(funcs_to_cloudify)
    ws_dag = mk_dag_with_ws_funcs(dag, ws_funcs)

    HybridDAG = namedtuple("HybridDAG", ["funcs_to_cloudify", "ws_dag", "ws_funcs"])
    return HybridDAG(funcs_to_cloudify, ws_dag, ws_funcs)
```

## util.py

```python
"""util functions"""

import re
from functools import partial, wraps
from inspect import Parameter, getmodule
from types import ModuleType
from typing import (
    Any,
    Union,
    Optional,
    TypeVar,
    Tuple,
    List,
)
from collections.abc import Callable, Iterator, Iterable, Mapping
from importlib import import_module
from operator import itemgetter

from i2 import Sig, name_of_obj, LiteralVal, FuncFanout, Pipe

T = TypeVar("T")


def objects_defined_in_module(
    module: str | ModuleType,
    *,
    name_filt: Callable | None = None,
    obj_filt: Callable | None = None,
):
    """
    Get a dictionary of objects defined in a Python module, optionally filtered by their names and values.

    Parameters
    ----------
    module: Union[str, ModuleType]
        The module to look up. Can either be
        - the module object itself,
        - a string specifying the module's fully qualified name (e.g., 'os.path'), or
        - a .py filepath to the module

    name_filt: Optional[Callable], default=None
        An optional function used to filter the names of objects in the module.
        This function should take a single argument (the object name as a string)
        and return a boolean. Only objects whose names pass the filter (i.e.,
        for which the function returns True) are included.
        If None, no name filtering is applied.

    obj_filt: Optional[Callable], default=None
        An optional function used to filter the objects in the module. This function should take a
        single argument (the object itself) and return a boolean. Only objects that pass the filter
        (i.e., for which the function returns True) are included.
        If None, no object filtering is applied.

    Returns
    -------
    dict
        A dictionary where keys are names of objects defined in the module (filtered by name_filt and obj_filt)
        and values are the corresponding objects.

    Examples
    --------
    >>> import os
    >>> all_os_objects = objects_defined_in_module(os)
    >>> 'removedirs' in all_os_objects
    True
    >>> all_os_objects['removedirs'] == os.removedirs
    True

    See that you can specify the module via a string too, and filter to get only
    callables that don't start with an underscore:

    >>> this_modules_funcs = objects_defined_in_module(
    ...     'meshed.util',
    ...     name_filt=lambda name: not name.startswith('_'),
    ...     obj_filt=callable,
    ... )
    >>> callable(this_modules_funcs['objects_defined_in_module'])
    True

    """
    if isinstance(module, str):
        if module.endswith(".py") and os.path.isfile(module):
            module_filepath = module
            with filepath_to_module(module_filepath) as module:
                return objects_defined_in_module(
                    module, name_filt=name_filt, obj_filt=obj_filt
                )
        else:
            module = import_module(module)

    # At this point we have a module object (ModuleType)
    name_filt = name_filt or (lambda x: True)
    obj_filt = obj_filt or (lambda x: True)
    module_objs = vars(module)
    # Note we only filter for names here, not objects, because we want to keep the
    # object filtering for after we've gotten the module objects
    name_and_module = {
        name: getmodule(obj)
        for name, obj in module_objs.items()
        if name_filt(name) and obj is not None
    }
    obj_names = [
        obj_name
        for obj_name, obj_module in name_and_module.items()
        if obj_module is not None and obj_module.__name__ == module.__name__
    ]
    return {k: module_objs[k] for k in obj_names if obj_filt(module_objs[k])}


import importlib.util
import sys
import os
from contextlib import contextmanager


@contextmanager
def filepath_to_module(file_path: str):
    """
    A context manager to import a Python file as a module.

    :param file_path: The file path of the Python file to import.
    :yield: The module object.
    """
    file_path = os.path.abspath(file_path)
    dir_path = os.path.dirname(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create a module spec from the file path
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if spec is None:
        raise ImportError(f"Module {file_path} could not be imported.")

    # Add the directory of the file to sys.path
    sys.path.insert(0, dir_path)

    try:
        # Create a module from the spec and execute its code
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module

    finally:
        # Clean up: remove the added directory from sys.path
        sys.path.remove(dir_path)


def provides(*var_names: str) -> Callable[[Callable], Callable]:
    """Decorator to assign ``var_names`` to a ``_provides`` attribute of function.

    This is meant to be used to indicate to a mesh what var nodes a function can source
    values for.

    >>> @provides('a', 'b')
    ... def f(x):
    ...     return x + 1
    >>> f._provides
    ('a', 'b')

    If no ``var_names`` are given, then the function name is used as the var name:

    >>> @provides()
    ... def g(x):
    ...     return x + 1
    >>> g._provides
    ('g',)

    If ``var_names`` contains ``'_'``, then the function name is used as the var name
    for that position:

    >>> @provides('b', '_')
    ... def h(x):
    ...     return x + 1
    >>> h._provides
    ('b', 'h')

    """

    def add_provides_attribute(func):
        if not var_names:
            var_names_ = (name_of_obj(func),)
        else:
            var_names_ = tuple(
                [x if x != "_" else name_of_obj(func) for x in var_names]
            )
        func._provides = var_names_
        return func

    return add_provides_attribute


def if_then_else(if_func, then_func, else_func, *args, **kwargs):
    """
    Tool to "functionalize" the if-then-else logic.

    >>> from functools import partial
    >>> f = partial(if_then_else, str.isnumeric, int, str)
    >>> f('a string')
    'a string'
    >>> f('42')
    42

    """
    if if_func(*args, **kwargs):
        return then_func(*args, **kwargs)
    else:
        return else_func(*args, **kwargs)


# TODO: Revise FuncFanout so it makes a generator of values, or items, instead of a dict
def funcs_conjunction(*funcs):
    """
    Makes a conjunction of functions. That is, ``func1(x) and func2(x) and ...``

    >>> f = funcs_conjunction(lambda x: isinstance(x, str), lambda x: len(x) >= 5)
    >>> f('app')  # because length is less than 5...
    False
    >>> f('apple')  # length at least 5 so...
    True

    Note that in:

    >>> f(42)
    False

    it is ``False`` because it is not a string.
    This shows that the second function is not applied to the input at all, since it
    doesn't need to, and if it were, we'd get an error (length of a number?!).

    """
    return Pipe(FuncFanout(*funcs), partial(map, itemgetter(1)), all)


def funcs_disjunction(*funcs):
    """
    Makes a disjunction of functions. That is, ``func1(x) or func2(x) or ...``

    >>> f = funcs_disjunction(lambda x: x > 10, lambda x: x < -5)
    >>> f(7)
    False
    >>> f(-7)
    True
    """
    return Pipe(FuncFanout(*funcs), partial(map, itemgetter(1)), any)


def extra_wraps(func, name=None, doc_prefix=""):
    func.__name__ = name or func_name(func)
    func.__doc__ = doc_prefix + getattr(func, "__name__", "")
    return func


def mywraps(func, name=None, doc_prefix=""):
    def wrapper(wrapped):
        return extra_wraps(wraps(func)(wrapped), name=name, doc_prefix=doc_prefix)

    return wrapper


def iterize(func, name=None):
    """From an Input->Ouput function, makes a Iterator[Input]->Itertor[Output]
    Some call this "vectorization", but it's not really a vector, but an
    iterable, thus the name.

    `iterize` is a partial of `map`.

    >>> f = lambda x: x * 10
    >>> f(2)
    20
    >>> iterized_f = iterize(f)
    >>> list(iterized_f(iter([1,2,3])))
    [10, 20, 30]

    Consider the following pipeline:

    >>> from i2 import Pipe
    >>> pipe = Pipe(lambda x: x * 2, lambda x: f"hello {x}")
    >>> pipe(1)
    'hello 2'

    But what if you wanted to use the pipeline on a "stream" of data. The
    following wouldn't work:

    >>> try:
    ...     pipe(iter([1,2,3]))
    ... except TypeError as e:
    ...     print(f"{type(e).__name__}: {e}")
    ...
    ...
    TypeError: unsupported operand type(s) for *: 'list_iterator' and 'int'

    Remember that error: You'll surely encounter it at some point.

    The solution to it is (often): ``iterize``,
    which transforms a function that is meant to be applied to a single object,
    into a function that is meant to be applied to an array, or any iterable
    of such objects.
    (You might be familiar (if you use `numpy` for example) with the related
    concept of "vectorization",
    or [array programming](https://en.wikipedia.org/wiki/Array_programming).)


    >>> from i2 import Pipe
    >>> from meshed.util import iterize
    >>> from typing import Iterable
    >>>
    >>> pipe = Pipe(
    ...     iterize(lambda x: x * 2),
    ...     iterize(lambda x: f"hello {x}")
    ... )
    >>> iterable = pipe([1, 2, 3])
    >>> # see that the result is an iterable
    >>> assert isinstance(iterable, Iterable)
    >>> list(iterable)  # consume the iterable and gather it's items
    ['hello 2', 'hello 4', 'hello 6']
    """
    # TODO: See if partialx can be used instead
    wrapper = mywraps(
        func, name=name, doc_prefix=f"generator version of {func_name(func)}:\n"
    )
    return wrapper(partial(map, func))


# from typing import Callable, Any
# from functools import wraps
# from i2 import Sig, name_of_obj


def my_isinstance(obj, class_or_tuple):
    """Same as builtin instance, but without position only constraint.
    Therefore, we can partialize class_or_tuple:

    Otherwise, couldn't do:

    >>> isinstance_of_str = partial(my_isinstance, class_or_tuple=str)
    >>> isinstance_of_str('asdf')
    True
    >>> isinstance_of_str(3)
    False

    """
    return isinstance(obj, class_or_tuple)


def instance_checker(class_or_tuple):
    """Makes a boolean function that checks the instance of an object

    >>> isinstance_of_str = instance_checker(str)
    >>> isinstance_of_str('asdf')
    True
    >>> isinstance_of_str(3)
    False

    """
    return partial(my_isinstance, class_or_tuple=class_or_tuple)


class ConditionalIterize:
    """A decorator that "iterizes" a function call if input satisfies a condition.
    That is, apply ``map(func, input)`` (iterize) or ``func(input)`` according to some
    conidition on ``input``.

    >>> def foo(x, y=2):
    ...     return x * y

    The function does this:

    >>> foo(3)
    6
    >>> foo('string')
    'stringstring'

    The iterized version of the function does this:

    >>> iterized_foo = iterize(foo)
    >>> list(iterized_foo([1, 2, 3]))
    [2, 4, 6]

    >>> from typing import Iterable
    >>> new_foo = ConditionalIterize(foo, Iterable)
    >>> new_foo(3)
    6
    >>> list(new_foo([1, 2, 3]))
    [2, 4, 6]

    See what happens if we do this:

    >>> list(new_foo('string'))
    ['ss', 'tt', 'rr', 'ii', 'nn', 'gg']

    Maybe you expected `'stringstring'` because you are thinking of `string` as a valid,
    single input. But the condition of iterization is to be an Iterable, which a
    string is, thus the (perhaps) unexpected result.

    In fact, this problem is a general one:
    If your base function doesn't process iterables, the ``isinstance(x, Iterable)``
    is good enough -- but if it is supposed to process an iterable in the first place,
    how can you distinguish whether to use the iterized version or not?
    The solution depends on the situation and the iterface you want. You choose.

    Since the situation where you'll want to iterize functions in the first place is when
    you're building streaming pipelines, a good fallback choice is to iterize if and
    only if the input is an iterator. This is condition will trigger the iterization
    when the input has a ``__next__`` -- so things like generators, but not lists,
    tuples, sets, etc.

    See in the following that ``ConditionalIterize`` also has a ``wrap`` class method
    that can be used to wrap a function at definition time.

    >>> @ConditionalIterize.wrap(Iterator)  # Iterator is the default, so no need here
    ... def foo(x, y=2):
    ...     return x * y
    >>> foo(3)
    6
    >>> foo('string')
    'stringstring'

    If you want to process a "stream" of numbers 1, 2, 3, don't do it this way:

    >>> foo([1, 2, 3])
    [1, 2, 3, 1, 2, 3]

    Instead, you should explicitly wrap that iterable in an iterator, to trigger the
    iterization:

    >>> list(foo(iter([1, 2, 3])))
    [2, 4, 6]

    So far, the only way we controlled the iterize condition is through a type.
    Really, the condition that is used behind the scenes is
    ``isinstance(obj, self.iterize_type)``.
    If you need more complex conditions though, you can specify it through the
    ``iterize_condition`` argument. The ``iterize_type`` is also used to
    annotate the resulting wrapped function if it's first argument is annotated.
    As a consequence, ``iterize_type`` needs to be a "generic" type.

    >>> @ConditionalIterize.wrap(Iterable, lambda x: isinstance(x, (list, tuple)))
    ... def foo(x: int, y=2):
    ...     return x * y
    >>> foo(3)
    6
    >>> list(foo([1, 2, 3]))
    [2, 4, 6]
    >>> from inspect import signature

    We annotated ``x`` as ``int``, so see now the annotation of the wrapped function:

    >>> str(signature(foo))
    '(x: Union[int, Iterable[int]], y=2)'

    """

    def __init__(
        self,
        func: Callable,
        iterize_type: type = Iterator,
        iterize_condition: Callable[[Any], bool] | None = None,
    ):
        """

        :param func:
        :param iterize_type: The generic type to use for the new annotation
        :param iterize_condition: The condition to use to check if we should use
            the iterized version or not. If not given, will use
            ``functools.partial(my_isinstance, iterize_type)``
        """
        self.func = func
        self.iterize_type = iterize_type
        if iterize_condition is None:
            iterize_condition = instance_checker(iterize_type)
        self.iterize_condition = iterize_condition
        self.iterized_func = iterize(self.func)
        self.sig = Sig(self.func)
        wraps(self.func)(self)
        self.__signature__ = self._new_sig()

    def __call__(self, *args, **kwargs):
        _kwargs = self.sig.map_arguments(
            args, kwargs, apply_defaults=True, allow_partial=True
        )
        first_arg = next(iter(_kwargs.values()))
        if self.iterize_condition(first_arg):
            return self.iterized_func(*args, **kwargs)
        else:
            return self.func(*args, **kwargs)

    def __repr__(self):
        return f"<ConditionalIterize {name_of_obj(self)}{Sig(self)}>"

    def _new_sig(self):
        if len(self.sig.names) == 0:
            raise TypeError(
                f"You can only apply conditional iterization on functions that have "
                f"at least one input. This one had none: {self.func}"
            )
        first_param = self.sig.names[0]
        new_sig = self.sig  # same sig by default
        if first_param in self.sig.annotations:
            obj_annot = self.sig.annotations[first_param]
            new_sig = self.sig.ch_annotations(
                **{first_param: Union[obj_annot, self.iterize_type[obj_annot]]}
            )
        return new_sig

    @classmethod
    def wrap(
        cls,
        iterize_type: type = Iterator,
        iterize_condition: Callable[[Any], bool] | None = None,
    ):
        return partial(
            cls, iterize_type=iterize_type, iterize_condition=iterize_condition
        )


class ModuleNotFoundIgnore:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is ModuleNotFoundError:
            pass
        return True


def incremental_str_maker(str_format="{:03.f}"):
    """Make a function that will produce a (incrementally) new string at every call."""
    i = 0

    def mk_next_str():
        nonlocal i
        i += 1
        return str_format.format(i)

    return mk_next_str


lambda_name = incremental_str_maker(str_format="lambda_{:03.0f}")
unnameable_func_name = incremental_str_maker(str_format="unnameable_func_{:03.0f}")

FunctionNamer = Callable[[Callable], str]

func_name: FunctionNamer


def func_name(func) -> str:
    """The func.__name__ of a callable func, or makes and returns one if that fails.
    To make one, it calls unamed_func_name which produces incremental names to reduce the chances of clashing
    """
    try:
        name = func.__name__
        if name == "<lambda>":
            return lambda_name()
        return name
    except AttributeError:
        return unnameable_func_name()


# ---------------------------------------------------------------------------------------
# Misc

from typing import Optional
from collections.abc import Iterable, Callable


def args_funcnames(
    funcs: Iterable[Callable], name_of_func: FunctionNamer | None = func_name
):
    """Generates (arg_name, func_id) pairs from the iterable of functions"""
    from inspect import signature, Parameter

    for func in funcs:
        sig = signature(func)
        for param in sig.parameters.values():
            arg_name = ""  # initialize
            if param.kind == Parameter.VAR_POSITIONAL:
                arg_name += "*"
            elif param.kind == Parameter.VAR_KEYWORD:
                arg_name += "**"
            arg_name += param.name  # append name of param
            yield arg_name, name_of_func(func)


def funcs_to_digraph(funcs, graph=None):
    from graphviz import Digraph

    graph = graph or Digraph()
    graph.edges(list(args_funcnames(funcs)))
    graph.body.extend([", ".join(func.__name__ for func in funcs) + " [shape=box]"])
    return graph


def dot_to_ascii(dot: str, fancy: bool = True):
    """Convert a dot string to an ascii rendering of the diagram.

    Needs a connection to the internet to work.


    >>> graph_dot = '''
    ...     graph {
    ...         rankdir=LR
    ...         0 -- {1 2}
    ...         1 -- {2}
    ...         2 -> {0 1 3}
    ...         3
    ...     }
    ... '''
    >>>
    >>> graph_ascii = dot_to_ascii(graph_dot)  # doctest: +SKIP
    >>>
    >>> print(graph_ascii)  # doctest: +SKIP
    <BLANKLINE>
                     ┌─────────┐
                     ▼         │
         ┌───┐     ┌───┐     ┌───┐     ┌───┐
      ┌▶ │ 0 │ ─── │ 1 │ ─── │   │ ──▶ │ 3 │
      │  └───┘     └───┘     │   │     └───┘
      │    │                 │   │
      │    └──────────────── │ 2 │
      │                      │   │
      │                      │   │
      └───────────────────── │   │
                             └───┘
    <BLANKLINE>

    """
    import requests

    url = "https://dot-to-ascii.ggerganov.com/dot-to-ascii.php"
    boxart = 0

    # use nice box drawing char instead of + , | , -
    if fancy:
        boxart = 1

    stripped_dot_str = dot.strip()
    if not (
        stripped_dot_str.startswith("graph") or stripped_dot_str.startswith("digraph")
    ):
        dot = "graph {\n" + dot + "\n}"

    params = {
        "boxart": boxart,
        "src": dot,
    }

    try:
        response = requests.get(url, params=params).text
    except requests.exceptions.ConnectionError:
        return "ConnectionError: You need the internet to convert dot into ascii!"

    if response == "":
        raise SyntaxError("DOT string is not formatted correctly")

    return response


def print_ascii_graph(funcs):
    digraph = funcs_to_digraph(funcs)
    dot_str = "\n".join(map(lambda x: x[1:], digraph.body[:-1]))
    print(dot_to_ascii(dot_str))


class ValidationError(ValueError):
    """Error that is raised when an object's validation failed"""


class NotUniqueError(ValidationError):
    """Error to be raised when unicity is expected, but violated"""


class NotFound(ValidationError):
    """To be raised when something is expected to exist, but doesn't"""


class NameValidationError(ValueError):
    """Use to indicate that there's a problem with a name or generating a valid name"""


def find_first_free_name(prefix, exclude_names=(), start_at=2):
    if prefix not in exclude_names:
        return prefix
    else:
        i = start_at
        while True:
            name = f"{prefix}__{i}"
            if name not in exclude_names:
                return name
            i += 1


def mk_func_name(func, exclude_names=()):
    """Makes a function name that doesn't clash with the exclude_names iterable.
    Tries it's best to not be lazy, but instead extract a name from the function
    itself."""
    name = name_of_obj(func) or "func"
    if name == "<lambda>":
        name = lambda_name()  # make a lambda name that is a unique identifier
    return find_first_free_name(name, exclude_names)


def arg_names(func, func_name, exclude_names=()):
    names = Sig(func).names

    def gen():
        _exclude_names = exclude_names
        for name in names:
            if name not in _exclude_names:
                yield name
            else:
                found_name = find_first_free_name(
                    f"{func_name}__{name}", _exclude_names
                )
                yield found_name
                _exclude_names = _exclude_names + (found_name,)

    return list(gen())


def named_partial(func, *args, __name__=None, **keywords):
    """functools.partial, but with a __name__

    >>> f = named_partial(print, sep='\\n')
    >>> f.__name__
    'print'

    >>> f = named_partial(print, sep='\\n', __name__='now_partial_has_a_name')
    >>> f.__name__
    'now_partial_has_a_name'
    """
    f = partial(func, *args, **keywords)
    f.__name__ = __name__ or func.__name__
    return f


def _place_holder_func(*args, _sig=None, **kwargs):
    _kwargs = _sig.map_arguments(args, kwargs)
    _kwargs_str = ", ".join(f"{k}={v}" for k, v in _kwargs.items())
    return f"{_sig.name}({_kwargs_str})"


def mk_place_holder_func(arg_names_or_sig, name=None, defaults=(), annotations=()):
    """Make (working and picklable) function with a specific signature.

    This is useful for testing as well as injecting compliant functions in DAG templates.

    :param arg_names_or_sig: Anything that i2.Sig can accept as it's first input.
        (Such as a string of argument(s), function, signature, etc.)
    :param name: The ``__name__`` to give the function.
    :param defaults: If you want to add/change defaults
    :param annotations: If you want to add/change annotations
    :return: A (working and picklable) function with a specific signature


    >>> f = mk_place_holder_func('a b', 'my_func')
    >>> f(1,2)
    'my_func(a=1, b=2)'

    The first argument can be any expression of a signature that ``i2.Sig`` can
    understand. For instance, it could be a function itself.
    See how the function takes on ``mk_place_holder_func``'s signature and name in the
    following example:

    >>> g = mk_place_holder_func(mk_place_holder_func)
    >>> from inspect import signature
    >>> str(signature(g))  # should give the same signature as mk_place_holder_func
    '(arg_names_or_sig, name=None, defaults=(), annotations=())'
    >>> g(1,2,defaults=3, annotations=4)
    'mk_place_holder_func(arg_names_or_sig=1, name=2, defaults=3, annotations=4)'

    """
    defaults = dict(defaults)
    sig = Sig(arg_names_or_sig)
    sig = sig.ch_defaults(**dict(defaults))
    sig = sig.ch_annotations(**dict(annotations))

    sig.name = name or sig.name or "place_holder_func"

    func = sig(partial(_place_holder_func, _sig=sig))
    func.__name__ = sig.name

    return func


# TODO: Probably can improve efficiency and reusability using generators?
def ordered_set_operations(a: Iterable, b: Iterable) -> tuple[list, list, list]:
    """
    Returns a triple (a-b, a&b, b-a) for two iterables a and b.
    The operations are performed as if a and b were sets, but the order in a is conserved.

    >>> ordered_set_operations([1, 2, 3, 4], [3, 4, 5, 6])
    ([1, 2], [3, 4], [5, 6])

    >>> ordered_set_operations("abcde", "cdefg")
    (['a', 'b'], ['c', 'd', 'e'], ['f', 'g'])

    >>> ordered_set_operations([1, 2, 2, 3], [2, 3, 3, 4])
    ([1], [2, 3], [4])
    """
    set_b = set(b)
    a = tuple(a)  # because a traversed three times (consider a one-pass algo)
    a_minus_b = [x for x in a if x not in set_b]
    a_intersect_b = [x for x in a if x in set_b and not set_b.remove(x)]
    b_minus_a = [x for x in b if x not in set(a)]

    return a_minus_b, a_intersect_b, b_minus_a


# utils to reorder funcnodes


def pairs(xs):
    if len(xs) <= 1:
        return xs
    else:
        pairs = list(zip(xs, xs[1:]))
    return pairs


def curry(func):
    def res(*args):
        return func(tuple(args))

    return res


def uncurry(func):
    def res(tup):
        return func(*tup)

    return res


Renamer = Union[Callable[[str], str], str, Mapping[str, str]]


def _if_none_return_input(func):
    """Wraps a function so that when the original func outputs None, the wrapped will
    return the original input instead.

    >>> def func(x):
    ...     if x % 2 == 0:
    ...         return None
    ...     else:
    ...         return x * 10
    >>> wfunc = _if_none_return_input(func)
    >>> func(3)
    30
    >>> wfunc(3)
    30
    >>> assert func(4) is None
    >>> wfunc(4)
    4
    """

    def _func(input_val):
        if (output_val := func(input_val)) is not None:
            return output_val
        else:
            return input_val

    return _func


def numbered_suffix_renamer(name, sep="_"):
    """
    >>> numbered_suffix_renamer('item')
    'item_1'
    >>> numbered_suffix_renamer('item_1')
    'item_2'
    """
    p = re.compile(sep + r"(\d+)$")
    m = p.search(name)
    if m is None:
        return f"{name}{sep}1"
    else:
        num = int(m.group(1)) + 1
        return p.sub(f"{sep}{num}", name)


class InvalidFunctionParameters(ValueError):
    """To be used when a function's parameters are not compliant with some rule about
    them."""


def _suffix(start=0):
    i = start
    while True:
        yield f"_{i}"
        i += 1


def _add_suffix(x, suffix):
    return f"{x}{suffix}"


incremental_suffixes = _suffix()
_renamers = (lambda x: f"{x}{suffix}" for suffix in incremental_suffixes)


def _return_val(first_arg, val):
    return val


def _equality_checker(x, val):
    return x == val


def _not_callable(obj):
    return not callable(obj)


# Pattern: routing
# TODO: Replace
def conditional_trans(
    obj: T, condition: Callable[[T], bool], trans: Callable[[T], Any]
):
    """Conditionally transform an object unless it is marked as a literal.

    >>> from functools import partial
    >>> trans = partial(
    ...     conditional_trans, condition=str.isnumeric, trans=float
    ... )
    >>> trans('not a number')
    'not a number'
    >>> trans('10')
    10.0

    To use this function but tell it to not transform some a specific input no matter
    what, wrap the input with ``Literal``

    >>> # from meshed import Literal
    >>> conditional_trans(LiteralVal('10'), str.isnumeric, float)
    '10'

    """
    # TODO: Maybe make Literal checking less sensitive to isinstance checks, using
    #   hasattr instead for example.
    if isinstance(obj, LiteralVal):  # If val is a Literal, return its value as is
        return obj.val
    elif condition(obj):  # If obj satisfies condition, return the alternative_obj
        return trans(obj)
    else:  # If not, just return object
        return obj


def replace_item_in_iterable(iterable, condition, replacement, *, egress=None):
    """Returns a list where all items satisfying ``condition(item)`` were replaced
    with ``replacement(item)``.

    If ``condition`` is not a callable, it will be considered as a value to check
    against using ``==``.

    If ``replacement`` is not a callable, it will be considered as the actual
    value to replace by.

    :param iterable: Input iterable of items
    :param condition: Condition to apply to item to see if it should be replaced
    :param replacement: (Conditional) replacement value or function
    :param egress: The function to apply to transformed iterable

    >>> replace_item_in_iterable([1,2,3,4,5], condition=2, replacement = 'two')
    [1, 'two', 3, 4, 5]
    >>> is_even = lambda x: x % 2 == 0
    >>> replace_item_in_iterable([1,2,3,4,5], condition=is_even, replacement = 'even')
    [1, 'even', 3, 'even', 5]
    >>> replace_item_in_iterable([1,2,3,4,5], is_even, replacement=lambda x: x * 10)
    [1, 20, 3, 40, 5]

    Note that if the input iterable is not a ``list``, ``tuple``, or ``set``,
    your output will be an iterator that you'll have to iterate through to gather
    transformed items.

    >>> g = replace_item_in_iterable(iter([1,2,3,4,5]), condition=2, replacement = 'two')
    >>> isinstance(g, Iterator)
    True

    Unless you specify an egress of your choice:

    >>> replace_item_in_iterable(
    ... iter([1,2,3,4,5]), is_even, lambda x: x * 10, egress=sorted
    ... )
    [1, 3, 5, 20, 40]

    """
    # If condition or replacement are not callable, make them so
    condition = conditional_trans(
        condition, _not_callable, lambda val: partial(_equality_checker, val=val)
    )
    replacement = conditional_trans(
        replacement, _not_callable, lambda val: partial(_return_val, val=val)
    )
    # Handle the egress argument
    if egress is None:
        if isinstance(iterable, (list, tuple, set)):
            egress = type(iterable)
        else:
            egress = lambda x: x  # that is return "as is"

    # Make the item replacer
    item_replacer = partial(conditional_trans, condition=condition, trans=replacement)

    return egress(map(item_replacer, iterable))


def _complete_dict_with_iterable_of_required_keys(
    to_complete: dict, complete_with: Iterable
):
    """Complete `to_complete` (in place) with `complete_with`
    `complete_with` contains values that must be covered by `to_complete`
    Those values that are not covered will be inserted in to_complete,
    with key=val

    >>> d = {'a': 'A', 'c': 'C'}
    >>> _complete_dict_with_iterable_of_required_keys(d, 'abc')
    >>> d
    {'a': 'A', 'c': 'C', 'b': 'b'}

    """
    keys_already_covered = set(to_complete)
    for required_key in complete_with:
        if required_key not in keys_already_covered:
            to_complete[required_key] = required_key


def inverse_dict_asserting_losslessness(d: dict):
    inv_d = {v: k for k, v in d.items()}
    assert len(inv_d) == len(d), (
        f"can't invert: You have some duplicate values in this dict: " f"{d}"
    )
    return inv_d


def _extract_values(d: dict, keys: Iterable):
    """generator of values extracted from d for keys"""
    for k in keys:
        yield d[k]


def extract_values(d: dict, keys: Iterable):
    """Extract values from dict ``d``, returning them:

    - as a tuple if len(keys) > 1

    - a single value if len(keys) == 1

    - None if not

    This is used as the default extractor in DAG

    >>> extract_values({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
    (1, 3)

    Order matters!

    >>> extract_values({'a': 1, 'b': 2, 'c': 3}, ['c', 'a'])
    (3, 1)

    """
    tup = tuple(_extract_values(d, keys))
    if len(tup) > 1:
        return tup
    elif len(tup) == 1:
        return tup[0]
    else:
        return None


def extract_items(d: dict, keys: Iterable):
    """generator of (k, v) pairs extracted from d for keys

    >>> list(extract_items({'a': 1, 'b': 2, 'c': 3}, ['a', 'c']))
    [('a', 1), ('c', 3)]

    """
    for k in keys:
        yield k, d[k]


def extract_dict(d: dict, keys: Iterable):
    """Extract items from dict ``d``, returning them as a dict.

    >>> extract_dict({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
    {'a': 1, 'c': 3}

    Order matters!

    >>> extract_dict({'a': 1, 'b': 2, 'c': 3}, ['c', 'a'])
    {'c': 3, 'a': 1}

    """
    return dict(extract_items(d, keys))


ParameterMerger = Callable[[Iterable[Parameter]], Parameter]
parameter_merger: ParameterMerger


# TODO: Be aware of i2.signatures.param_comparator in
#  https://github.com/i2mint/i2/blob/2bd43b350a3ae29f1e6c587dbe15d6f536635173/i2/signatures.py#L4247
#  and related funnctions, which are meant to be a more general approach. Consider
#  merging parameter_merger to use that general tooling.
# TODO: Make the ValidationError be even more specific, indicating what parameters
#  are different and how.
def parameter_merger(
    *params, same_name=True, same_kind=True, same_default=True, same_annotation=True
):
    """Validates that all the params are exactly the same, returning the first if so.

    This is used when hooking up functions that use the same parameters (i.e. arg
    names). When the name of an argument is used more than once, which kind, default,
    and annotation should be used in the interface of the DAG?

    If they're all the same, there's no problem.

    But if they're not the same, we need to provide control on which to ignore.

    >>> from inspect import Parameter as P
    >>> PK = P.POSITIONAL_OR_KEYWORD
    >>> KO = P.KEYWORD_ONLY
    >>> parameter_merger(P('a', PK), P('a', PK))
    <Parameter "a">
    >>> parameter_merger(P('a', PK), P('different_name', PK), same_name=False)
    <Parameter "a">
    >>> parameter_merger(P('a', PK), P('a', KO), same_kind=False)
    <Parameter "a">
    >>> parameter_merger(P('a', PK), P('a', PK,  default=42), same_default=False)
    <Parameter "a">
    >>> parameter_merger(P('a', PK, default=42), P('a', PK), same_default=False)
    <Parameter "a=42">
    >>> parameter_merger(P('a', PK, annotation=int), P('a', PK), same_annotation=False)
    <Parameter "a: int">
    """
    suggestion_on_error = """To resolve this you have several choices:

    - Change the properties of the param (kind, default, annotation) to be those you 
      want. For example, you can use ``i2.Sig.ch_param_attrs`` on the signatures 
      (or ``i2.Sig.ch_names``, ``i2.Sig.ch_defaults``, ``i2.Sig.ch_kinds``, 
      ``i2.Sig.ch_annotations``)
      to get a function decorator that will do that for you.
    - If you're making a DAG, consider specifying a different ``parameter_merge``.
      For example you can use ``functools.partial`` on 
      ``meshed.parameter_merger``, fixing ``same_kind``, ``same_default``, 
      and/or ``same_annotation`` to ``False`` to get a more lenient version of it.
      (See also i2.signatures.param_comparator.)

    See https://github.com/i2mint/i2/discussions/63 and 
    https://github.com/i2mint/meshed/issues/7 (description and comments) for more
    info.
    """
    first_param, *_ = params
    if same_name and not all(p.name == first_param.name for p in params):
        raise ValidationError(
            f"Some params didn't have the same name: {params}\n{suggestion_on_error}"
        )
    if same_kind and not all(p.kind == first_param.kind for p in params):
        raise ValidationError(
            f"Some params didn't have the same kind: {params}\n{suggestion_on_error}"
        )
    if same_default and not all(p.default == first_param.default for p in params):
        raise ValidationError(
            f"Some params didn't have the same default: {params}\n{suggestion_on_error}"
        )
    if same_annotation and not all(
        p.annotation == first_param.annotation for p in params
    ):
        raise ValidationError(
            f"Some params didn't have the same annotation: "
            f"{params}\n{suggestion_on_error}"
        )
    return first_param


conservative_parameter_merge: ParameterMerger = partial(
    parameter_merger, same_kind=True, same_default=True, same_annotation=True
)
```

## viz.py

```python
"""Visualization utilities for the meshed package."""

from typing import Any
from collections.abc import Iterable
from i2.signatures import Sig


def dot_lines_of_objs(objs: Iterable, start_lines=(), end_lines=(), **kwargs):
    r"""
    Get lines generator for the graphviz.DiGraph(body=list(...))

    >>> from meshed.base import FuncNode
    >>> def add(a, b=1):
    ...     return a + b
    >>> def mult(x, y=3):
    ...     return x * y
    >>> def exp(mult, a):
    ...     return mult ** a
    >>> func_nodes = [
    ...     FuncNode(add, out='x'),
    ...     FuncNode(mult, name='the_product'),
    ...     FuncNode(exp)
    ... ]
    >>> lines = list(dot_lines_of_objs(func_nodes))
    >>> assert lines == [
    ... 'x [label="x" shape="none"]',
    ... '_add [label="_add" shape="box"]',
    ... '_add -> x',
    ... 'a [label="a" shape="none"]',
    ... 'b [label="b=" shape="none"]',
    ... 'a -> _add',
    ... 'b -> _add',
    ... 'mult [label="mult" shape="none"]',
    ... 'the_product [label="the_product" shape="box"]',
    ... 'the_product -> mult',
    ... 'x [label="x" shape="none"]',
    ... 'y [label="y=" shape="none"]',
    ... 'x -> the_product',
    ... 'y -> the_product',
    ... 'exp [label="exp" shape="none"]',
    ... '_exp [label="_exp" shape="box"]',
    ... '_exp -> exp',
    ... 'mult [label="mult" shape="none"]',
    ... 'a [label="a" shape="none"]',
    ... 'mult -> _exp',
    ... 'a -> _exp'
    ... ]  # doctest: +SKIP

    >>> from meshed.util import dot_to_ascii
    >>>
    >>> print(dot_to_ascii('\n'.join(lines)))  # doctest: +SKIP
    <BLANKLINE>
                    a        ─┐
                              │
               │              │
               │              │
               ▼              │
             ┌─────────────┐  │
     b=  ──▶ │    _add     │  │
             └─────────────┘  │
               │              │
               │              │
               ▼              │
                              │
                    x         │
                              │
               │              │
               │              │
               ▼              │
             ┌─────────────┐  │
     y=  ──▶ │ the_product │  │
             └─────────────┘  │
               │              │
               │              │
               ▼              │
                              │
                  mult        │
                              │
               │              │
               │              │
               ▼              │
             ┌─────────────┐  │
             │    _exp     │ ◀┘
             └─────────────┘
               │
               │
               ▼
    <BLANKLINE>
                   exp
    <BLANKLINE>

    """
    # Should we validate here, or outside this module?
    # from meshed.base import validate_that_func_node_names_are_sane
    # validate_that_func_node_names_are_sane(func_nodes)
    yield from start_lines
    for obj in objs:
        yield from obj.dot_lines(**kwargs)
    yield from end_lines


dot_lines_of_func_nodes = dot_lines_of_objs  # backwards compatiblity alias


# TODO: Should we integrate this to dot_lines_of_func_parameters directly (decorator?)
def add_new_line_if_none(s: str):
    """Since graphviz 0.18, need to have a newline in body lines.
    This util is there to address that, adding newlines to body lines
    when missing."""
    if s and s[-1] != "\n":
        return s + "\n"
    return s


# ------------------------------------------------------------------------------
# Unused -- consider deleting
def _parameters_and_names_from_sig(
    sig: Sig,
    out=None,
    func_name=None,
):
    func_name = func_name or sig.name
    out = out or sig.name
    if func_name == out:
        func_name = "_" + func_name
    assert isinstance(func_name, str) and isinstance(out, str)
    return sig.parameters, out, func_name


# ------------------------------------------------------------------------------
# Old stuff


def visualize_graph(graph):
    import graphviz
    from IPython.display import display

    dot = graphviz.Digraph()

    # Add nodes to the graph
    for node in graph:
        dot.node(node)

    # Add edges to the graph
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            dot.edge(node, neighbor)

    # Render and display the graph in the notebook
    display(dot)


def visualize_graph_interactive(graph):
    import graphviz

    import networkx as nx
    import ipywidgets as widgets
    from IPython.display import display

    g = nx.DiGraph(graph)

    # Create an empty Graphviz graph
    dot = graphviz.Digraph()

    # Add nodes to the Graphviz graph
    for node in g.nodes:
        dot.node(str(node))

    # Add edges to the Graphviz graph
    for edge in g.edges:
        dot.edge(str(edge[0]), str(edge[1]))

    # Render the initial graph visualization
    graph_widget = widgets.HTML(value=dot.pipe(format="svg").decode("utf-8"))
    display(graph_widget)

    def add_edge(sender):
        source = source_node.value
        target = target_node.value
        if (source, target) not in g.edges:
            g.add_edge(source, target)
            dot.edge(str(source), str(target))
            graph_widget.value = dot.pipe(format="svg").decode("utf-8")
        source_node.value = ""
        target_node.value = ""

    def add_node(sender):
        node = new_node.value
        if node not in g.nodes:
            g.add_node(node)
            dot.node(str(node))
            graph_widget.value = dot.pipe(format="svg").decode("utf-8")
        new_node.value = ""

    def delete_edge(sender):
        source = str(delete_source.value)
        target = str(delete_target.value)
        if (source, target) in g.edges:
            g.remove_edge(source, target)
            dot.body.remove(f"\t{source} -> {target}\n")
            graph_widget.value = dot.pipe(format="svg").decode("utf-8")
        delete_source.value = ""
        delete_target.value = ""

    def delete_node(sender):
        node = delete_node_value.value
        if node in g.nodes:
            g.remove_node(node)
            dot.body[:] = [line for line in dot.body if str(node) not in line]
            graph_widget.value = dot.pipe(format="svg").decode("utf-8")
        delete_node_value.value = ""

    source_node = widgets.Text(placeholder="Source Node")
    target_node = widgets.Text(placeholder="Target Node")
    add_edge_button = widgets.Button(description="Add Edge")
    add_edge_button.on_click(add_edge)

    new_node = widgets.Text(placeholder="New Node")
    add_node_button = widgets.Button(description="Add Node")
    add_node_button.on_click(add_node)

    delete_source = widgets.Text(placeholder="Source Node")
    delete_target = widgets.Text(placeholder="Target Node")
    delete_edge_button = widgets.Button(description="Delete Edge")
    delete_edge_button.on_click(delete_edge)

    delete_node_value = widgets.Text(placeholder="Node")
    delete_node_button = widgets.Button(description="Delete Node")
    delete_node_button.on_click(delete_node)

    controls = widgets.HBox([source_node, target_node, add_edge_button])
    controls2 = widgets.HBox([new_node, add_node_button])
    controls3 = widgets.HBox([delete_source, delete_target, delete_edge_button])
    controls4 = widgets.HBox([delete_node_value, delete_node_button])
    display(controls)
    display(controls2)
    display(controls3)
    display(controls4)
```

## README.md

```python
# meshed

Object composition. 
In particular: Link functions up into callable objects (e.g. pipelines, DAGs, etc.)

To install: `pip install meshed`

[Documentation](https://i2mint.github.io/meshed/)

Note: The initial focuus of `meshed` was on DAGs, a versatile and probably most known kind of composition of functions, 
but `meshed` aims at capturing much more than that. 


# Quick Start

```python
from meshed import DAG

def this(a, b=1):
    return a + b
def that(x, b=1):
    return x * b
def combine(this, that):
    return (this, that)

dag = DAG((this, that, combine))
print(dag.synopsis_string())
```

    x,b -> that_ -> that
    a,b -> this_ -> this
    this,that -> combine_ -> combine


But what does it do?

It's a callable, with a signature:

```python
from inspect import signature
signature(dag)
```

    <Signature (x, a, b=1)>

And when you call it, it executes the dag from the root values you give it and
returns the leaf output values.

```python
dag(1, 2, 3)  # (a+b,x*b) == (2+3,1*3) == (5, 3)
```
    (5, 3)

```python
dag(1, 2)  # (a+b,x*b) == (2+1,1*1) == (3, 1)
```
    (3, 1)


You can see (and save image, or ascii art) the dag:

```python
dag.dot_digraph()
```

<img src="https://user-images.githubusercontent.com/1906276/127779463-ae75604b-0d69-4ac4-b206-80c2c5ae582b.png" width=200>


You can extend a dag

```python
dag2 = DAG([*dag, lambda this, a: this + a])
dag2.dot_digraph()
```

<img src="https://user-images.githubusercontent.com/1906276/127779748-70b47907-e51f-4e64-bc18-9545ee07e632.png" width=200>

You can get a sub-dag by specifying desired input(s) and outputs.

```python
dag2[['that', 'this'], 'combine'].dot_digraph()
```

<img src="https://user-images.githubusercontent.com/1906276/127779781-8aac40eb-ed52-4694-b50e-4af896cc30a2.png" width=150>



## Note on flexibility

The above DAG was created straight from the functions, using only the names of the
functions and their parameters to define how to hook the network up.

But if you didn't write those functions specifically for that purpose, or you want
to use someone else's functions, one would need to specify the relation between parameters, inputs and outputs.

For that purpose, functions can be adapted using the class FuncNode. The class allows you to essentially rename each of the parameters and also specify which output should be used as an argument for any other functions.

Let us consider the example below.

```python
def f(a, b):
    return a + b

def g(a_plus_b, d):
    return a_plus_b * d
```

Say we want the output of f to become the value of the parameter a_plus_b. We can do that by assigning the string 'a_plus_b' to the out parameter of a FuncNode representing the function f:

```python
f_node = FuncNode(func=f, out="a_plus_b")
```

We can now create a dag using our f_node instead of f:

```python
dag = DAG((f_node, g))
```

Our dag behaves as wanted:

```python
dag(a=1, b=2, d=3)
9
```

Now say we would also like for the value given to b to be also given to d. We can achieve that by binding d to b in the bind parameter of a FuncNode representing g:

```python
g_node = FuncNode(func=g, bind={"d": "b"})
```

The dag created with f_node and g_node has only two parameters, namely a and b:

```python
dag = DAG((f_node, g_node))
dag(a=1, b=2)
6
```




# Sub-DAGs


``dag[input_nodes:output_nodes]`` is the sub-dag made of intersection of all
descendants of ``input_nodes``
(inclusive) and ancestors of ``output_nodes`` (inclusive), where additionally,
when a func node is contained, it takes with it the input and output nodes
it needs.


```python
from meshed import DAG

def f(a): ...
def g(f): ...
def h(g): ...
def i(h): ...
dag = DAG([f, g, h, i])

dag.dot_digraph()
```

<img width="110" alt="image" src="https://user-images.githubusercontent.com/1906276/154749811-f9892ee6-617c-4fa6-9de9-1ebc509c04ae.png">


Get a subdag from ``g_`` (indicates the function here) to the end of ``dag``

```python
subdag = dag['g_',:]
subdag.dot_digraph()
```

<img width="100" alt="image" src="https://user-images.githubusercontent.com/1906276/154749842-c2320d1c-368d-4be8-ac57-9a77f1bb081d.png">

From the beginning to ``h_``

```python
dag[:, 'h_'].dot_digraph()
```

<img width="110" alt="image" src="https://user-images.githubusercontent.com/1906276/154750524-ece7f4b6-a3f3-46c6-a66d-7dc9b8ef254a.png">



From ``g_`` to ``h_`` (both inclusive)

```python
dag['g_', 'h_'].dot_digraph()
```

<img width="109" alt="image" src="https://user-images.githubusercontent.com/1906276/154749864-5a33aa13-0949-4aa7-945c-4d3fe7f07e7d.png">


Above we used function (node names) to specify what we wanted, but we can also
use names of input/output var-nodes. Do note the difference though.
The nodes you specify to get a sub-dag are INCLUSIVE, but when you
specify function nodes, you also get the input and output nodes of these
functions.

The ``dag['g_', 'h_']`` give us a sub-dag starting at ``f`` (the input node),
but when we ask ``dag['g', 'h_']`` instead, ``g`` being the output node of
function node ``g_``, we only get ``g -> h_ -> h``:

```python
dag['g', 'h'].dot_digraph()
```

<img width="88" alt="image" src="https://user-images.githubusercontent.com/1906276/154750753-737e2705-0ea3-4595-a93a-1567862a6edd.png">


If we wanted to include ``f`` we'd have to specify it:


```python
dag['f', 'h'].dot_digraph()
```

<img width="109" alt="image" src="https://user-images.githubusercontent.com/1906276/154749864-5a33aa13-0949-4aa7-945c-4d3fe7f07e7d.png">


Those were for simple pipelines, but let's now look at a more complex dag.

Note the definition: ``dag[input_nodes:output_nodes]`` is the sub-dag made of intersection of all 
descendants of ``input_nodes``
(inclusive) and ancestors of ``output_nodes`` (inclusive), where additionally,
when a func node is contained, it takes with it the input and output nodes
it needs.

We'll let the following examples self-comment:

```python
from meshed import DAG


def f(u, v): ...

def g(f): ...

def h(f, w): ...

def i(g, h): ...

def j(h, x): ...

def k(i): ...

def l(i, j): ...

dag = DAG([f, g, h, i, j, k, l])

dag.dot_digraph()
```

<img width="248" alt="image" src="https://user-images.githubusercontent.com/1906276/154748574-a7026125-659f-465b-9bc3-14a1864d14b2.png">

```python
dag[['u', 'f'], 'h'].dot_digraph()
```

<img width="190" alt="image" src="https://user-images.githubusercontent.com/1906276/154748685-24e706ce-b68f-429a-b7b8-7bda62ccdf36.png">


```python
dag['u', 'h'].dot_digraph()
```

<img width="183" alt="image" src="https://user-images.githubusercontent.com/1906276/154748865-6e729094-976a-4af3-87f0-b6dd3900fb8c.png">


```python
dag[['u', 'f'], ['h', 'g']].dot_digraph()
```

<img width="199" alt="image" src="https://user-images.githubusercontent.com/1906276/154748905-4eaeccbe-6cca-4492-a7a2-48f7c9937b95.png">


```python
dag[['x', 'g'], 'k'].dot_digraph()
```

<img width="133" alt="image" src="https://user-images.githubusercontent.com/1906276/154748937-7a278b25-6f0f-467c-a977-89a175e15abb.png">

```python
dag[['x', 'g'], ['l', 'k']].dot_digraph()
```

<img width="216" alt="image" src="https://user-images.githubusercontent.com/1906276/154748958-135792a6-ce16-4561-9cbe-4662113a1022.png">



# Examples

## A train/test ML pipeline

Consider a simple train/test ML pipeline that looks like this.

![image](https://user-images.githubusercontent.com/1906276/135151068-179d958e-9e96-48aa-9188-52ae22919c6e.png)

With this, we might decide we want to give the user control over how to do 
`train_test_split` and `learner`, so we offer this interface to the user:

![image](https://user-images.githubusercontent.com/1906276/135151094-661850c0-f10c-49d8-ace2-46b3d994de80.png)

With that, the user can just bring its own `train_test_split` and `learner` 
functions, and as long as it satisfied the 
expected (and even better; declared and validatable) protocol, things will work. 

In some situations we'd like to fix some of how `train_test_split` and 
`learner` work, allowing the user to control only some aspects of them. 
This function would look like this:

![image](https://user-images.githubusercontent.com/1906276/135151137-3d9a290f-d5e7-4f24-a418-82f1edb8a46a.png)

And inside, it does:

![image](https://user-images.githubusercontent.com/1906276/135151114-926b52b8-0536-4565-bd56-95099f21e4ff.png)

`meshed` allows us to easily manipulate such functional structures to 
adapt them to our needs.


# itools module
Tools that enable operations on graphs where graphs are represented by an adjacency Mapping.

Again. 

Graphs: You know them. Networks. 
Nodes and edges, and the ecosystem descriptive or transformative functions surrounding these.
Few languages have builtin support for the graph data structure, but all have their libraries to compensate.

The one you're looking at focuses on the representation of a graph as `Mapping` encoding 
its [adjacency list](https://en.wikipedia.org/wiki/Adjacency_list). 
That is, a dictionary-like interface that specifies the graph by specifying for each node
what nodes it's adjacent to:

```python
assert graph[source_node] == set_of_nodes_that_source_node_has_edges_to
```

We emphasize that there is no specific graph instance that you need to squeeze your graph into to
be able to use the functions of `meshed`. Suffices that your graph's structure is expressed by 
that dict-like interface 
-- which grown-ups call `Mapping` (see the `collections.abc` or `typing` standard libs for more information).

You'll find a lot of `Mapping`s around pythons. 
And if the object you want to work with doesn't have that interface, 
you can easily create one using one of the many tools of `py2store` meant exactly for that purpose.


# Examples

```pydocstring
>>> from meshed.itools import edges, nodes, isolated_nodes
>>> graph = dict(a='c', b='ce', c='abde', d='c', e=['c', 'b'], f={})
>>> sorted(edges(graph))
[('a', 'c'), ('b', 'c'), ('b', 'e'), ('c', 'a'), ('c', 'b'), ('c', 'd'), ('c', 'e'), ('d', 'c'), ('e', 'b'), ('e', 'c')]
>>> sorted(nodes(graph))
['a', 'b', 'c', 'd', 'e', 'f']
>>> set(isolated_nodes(graph))
{'f'}
>>>
>>> from meshed.makers import edge_reversed_graph
>>> g = dict(a='c', b='cd', c='abd', e='')
>>> assert edge_reversed_graph(g) == {'c': ['a', 'b'], 'd': ['b', 'c'], 'a': ['c'], 'b': ['c'], 'e': []}
>>> reverse_g_with_sets = edge_reversed_graph(g, set, set.add)
>>> assert reverse_g_with_sets == {'c': {'a', 'b'}, 'd': {'b', 'c'}, 'a': {'c'}, 'b': {'c'}, 'e': set([])}
```
```