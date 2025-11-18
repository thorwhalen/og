## __init__.py

```python
"""main creek objects"""
from creek.base import Creek

from creek.infinite_sequence import InfiniteSeq, IndexedBuffer
from creek.tools import (
    filter_and_index_stream,
    dynamically_index,
    DynamicIndexer,
    count_increments,
    size_increments,
    BufferStats,
    Segmenter,
)
```

## automatas.py

```python
"""Automatas (finite state machines etc.)"""

from typing import Union, TypeVar, Mapping, Iterable, Callable, Tuple
from functools import partial
from dataclasses import dataclass

State = TypeVar('State')
Symbol = TypeVar('Symbol')
Automata = Callable[[State, Iterable[Symbol]], Iterable[State]]

TransitionFunc = Callable[[State, Symbol], State]
AutomataFactory = Callable[[TransitionFunc], Automata]


def mapping_to_transition_func(
    mapping: Mapping[Tuple[State, Symbol], State], strict: bool = True
) -> TransitionFunc:
    """
    Helper to make a transition function from a mapping of (state, symbol)->state.
    """
    if strict:

        def transition_func(state: State, symbol: Symbol) -> State:
            return mapping[(state, symbol)]

    else:

        def transition_func(state: State, symbol: Symbol) -> State:
            return mapping.get((state, symbol), state)

    return transition_func


StateMapper = Union[Callable[[State], State], Mapping[State, State]]


@dataclass
class MappingTransitionFunc:
    mapping: Mapping[Tuple[State, Symbol], State]
    strict: bool = True

    def __call__(self, state: State, symbol: Symbol) -> State:
        if self.strict:
            return self.mapping[(state, symbol)]
        else:
            return self.mapping.get((state, symbol), state)

    def map_states(self, state_mapper: StateMapper) -> 'MappingTransitionFunc':
        """Return a new MappingTransitionFunc with the same mapping but with
        state_mapper applied to the states."""
        if isinstance(state_mapper, Mapping):
            state_mapper = state_mapper.get
        new_mapping = {
            (state_mapper(state), symbol): state_mapper(state)
            for (state, symbol), state in self.mapping.items()
        }
        return MappingTransitionFunc(new_mapping, self.strict)


# functional version
def _basic_automata(
    transition_func: TransitionFunc, state: State, symbols: Iterable[Symbol]
) -> State:
    for symbol in symbols:
        # Note: if the (state, symbol) combination is not in the transitions
        #     mapping, the state is left unchanged.
        state = transition_func(state, symbol)
        yield state


basic_automata: AutomataFactory
BasicAutomata: AutomataFactory


# # NerdNote: Could do it like this too
# basic_automata: AutomataFactory = partial(partial, _basic_automata)
def basic_automata(transition_func: TransitionFunc, state: State) -> Automata:
    return partial(_basic_automata, transition_func, state)


@dataclass
class BasicAutomata:
    transition_func: TransitionFunc
    state: State = None

    def __post_init__(self):
        self._initial_state = self.state

    def __call__(self, state: State, symbols: Iterable[Symbol]) -> State:
        self.state = state
        for symbol in symbols:
            yield self.transition(symbol)

    def transition(self, symbol: Symbol) -> State:
        self.state = self.transition_func(self.state, symbol)
        return self.state

    def reset(self):
        """Reset state to initial state and return self."""
        self.state = self._initial_state
        return self


automata: AutomataFactory = basic_automata  # back-compatibility alias
```

## base.py

```python
"""The base objects of creek"""

from creek.util import cls_wrap, static_identity_method, no_such_item


class Creek:
    """A layer-able version of the stream interface

    There are three layering methods -- `pre_iter`, `data_to_obj`, and `post_iter`
    -- whose use is demonstrated in the iteration code below:

    >>> from io import StringIO
    >>>
    >>> src = StringIO(
    ... '''a, b, c
    ... 1,2, 3
    ... 4, 5,6
    ... '''
    ... )
    >>>
    >>> from creek.base import Creek
    >>>
    >>> class MyCreek(Creek):
    ...     def data_to_obj(self, line):
    ...         return [x.strip() for x in line.strip().split(',')]
    ...
    >>> stream = MyCreek(src)
    >>>
    >>> list(stream)
    [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]

    If we try that again, we'll get an empty list since the cursor is at the end.

    >>> list(stream)
    []

    But if the underlying stream has a seek, so does the creek, so we can "rewind"

    >>> stream.seek(0)
    0

    >>> list(stream)
    [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]

    You can also use ``next`` to get stream items one by one

    >>> stream.seek(0)  # rewind again to get back to the beginning
    0
    >>> next(stream)
    ['a', 'b', 'c']
    >>> next(stream)
    ['1', '2', '3']

    Let's add a filter! There's two kinds you can use.
    One that is applied to the line before the data is transformed by data_to_obj,
    and the other that is applied after (to the obj).

    >>> from creek.base import Creek
    >>> from io import StringIO
    >>>
    >>> src = StringIO(
    ...     '''a, b, c
    ... 1,2, 3
    ... 4, 5,6
    ... ''')
    >>> class MyFilteredCreek(MyCreek):
    ...     def post_iter(self, objs):
    ...         yield from filter(lambda obj: str.isnumeric(obj[0]), objs)
    >>>
    >>> s = MyFilteredCreek(src)
    >>>

    >>> list(s)
    [['1', '2', '3'], ['4', '5', '6']]
    >>> s.seek(0)
    0
    >>> next(s)
    ['1', '2', '3']
    >>> next(s)
    ['4', '5', '6']

    Recipes:

    - `pre_iter`: involving `itertools.islice` to skip header lines
    - `pre_iter`: involving enumerate to get line indices in stream iterator
    - `pre_iter = functools.partial(map, pre_proc_func)` to preprocess all streamitems \
        with `pre_proc_func`
    - `pre_iter`: include filter before obj
    - `post_iter`: `chain.from_iterable` to flatten a chunked/segmented stream
    - `post_iter`: `functools.partial(filter, condition)` to filter yielded objs

    """

    def __init__(self, stream):
        self.stream = stream

    wrap = classmethod(cls_wrap)

    def __getattr__(self, attr):
        """Delegate method to wrapped stream"""
        return getattr(self.stream, attr)

    def __dir__(self):
        return list(
            {*dir(self.__class__), *dir(self.stream)}
        )  # to forward dir to delegated stream as well
        # return list(set(dir(self.__class__)).union(self.stream.__dir__()))  # to forward dir to delegated stream as well

    def __hash__(self):
        return self.stream.__hash__()

    # _data_of_obj = static_identity_method  # for write methods
    pre_iter = static_identity_method
    data_to_obj = static_identity_method
    # post_filt = stream_util.always_true
    post_iter = static_identity_method

    def __iter__(self):
        yield from self.post_iter(map(self.data_to_obj, self.pre_iter(self.stream)))

        # for line in self.pre_iter(self.stream):
        #     obj = self.data_to_obj(line)
        #     if self.post_filt(obj):
        #         yield obj

        # TODO: See pros and cons of above vs below:
        # yield from filter(self.post_filt,
        #                   map(self.data_to_obj,
        #                       self.pre_iter(self.stream)))

    # _wrapped_methods = {'__iter__'}

    def __next__(self):  # TODO: Pros and cons of having a __next__?
        """by default: next(iter(self))
        """
        return next(iter(self))

    def __enter__(self):
        self.stream.__enter__()
        return self
        # return self._pre_proc(self.stream) # moved to iter to

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.stream.__exit__(
            exc_type, exc_val, exc_tb
        )  # TODO: Should we have a _post_proc? Uses?


# class Brook(Creek):
#     post_iter = static_identity_method
#
#     def __iter__(self):
#         yield from self.post_iter(
#             filter(self.post_filt,
#                    map(self.data_to_obj,
#                        self.pre_iter(
#                            self.stream))))
```

## infinite_sequence.py

```python
"""
Objects that support some list-like read operations on an unbounded stream.
Essentially, trying to give you the impression that you have read access to infinite list,
with some (parametrizable) limitations.

"""
# TODO: Build up extensive relations expression and handling, but InfiniteSeq only uses BEFORE (past).
#  Consider simplifying.

from collections import deque
from typing import Iterable, Tuple, Union, Callable, Any, NewType
from functools import wraps, partial, partialmethod
from enum import Enum
from operator import le, lt, ge, gt, itemgetter
from threading import Lock
from itertools import count, islice

Number = Union[int, float]  # TODO: existing builtin to specify real numbers?

opposite_op = {
    le: gt,
    lt: ge,
    ge: lt,
    gt: le,
}


# TODO: Check performance for string versus int enum values
class Relations(Enum):
    """Point-interval and interval-interval relations.

    See Allen's interval algebra for (some of the) interval relations
    (https://en.wikipedia.org/wiki/Allen%27s_interval_algebra).
    """

    # simple relations, that can be used between
    # (X: point, Y: interval) or (X: interval, Y: interval) pairs
    BEFORE = 'Some of X happens BEFORE Y'
    DURING = 'All of X happens within Y'
    AFTER = 'Some of X happens AFTER Y'

    # Extras (Allen's)
    PRECEDES = 'X precedes Y: All of X happens before Y'
    MEETS = 'X meets Y: When X ends, Y starts'
    OVERLAPS = 'X overlaps Y: Point is AFTER interval'
    STARTS = 'X starts at the same time as Y (and finishes no later)'
    FINISHES = 'X finishes Y: Point is AFTER interval'
    EQUAL = 'X is equal to Y'


def validate_interval(interval):
    """Asserts that input is a valid interval, raising a ValueError if not"""
    try:
        bt, tt = interval
        assert bt <= tt
        return bt, tt
    except Exception as e:
        raise ValueError(f'Not a valid interval: {interval}')


# TODO: Validate intervals (assert x[0] <= x[1] and )?
def simple_interval_relationship(
    x: Tuple[Number, Number],
    y: Tuple[Number, Number],
    above_bt: Callable = ge,
    below_tt: Callable = lt,
):
    """Get the simple relationship between intervals x and y.

    :param x: An point (a number) or an interval (a 2-tuple of numbers).
    :param y: An interval; a 2-tuple of numbers.
    :param above_bt: a above_bt(x_bt, y_bt) boolean function (ge or gt) deciding if x starts after y does.
    :param below_tt: a below_tt(x_tt, y_tt) boolean function (lt or le) deciding if x ends before y does.
    :return: One of three relations
        Relations.BEFORE if some of x is below y,
        Relations.AFTER if some of x is after y,
        Relations.DURING if x is entirely with y

    The target ``y`` interval is expressed only by it's bounds, but we don't know if
     these are inclusive or not. The ``below_bt`` and ``above_tt`` allow us to express
     that by expressing how below the lowest (bt) bound and what higher than highest
     (tt) bound are defined.

    The function is meant to be curried (partial), for example:

    >>> from functools import partial
    >>> from operator import le, lt, ge, gt
    >>> default = simple_interval_relationship  # uses below_bt=ge, above_tt=lt
    >>> including_bounds = partial(simple_interval_relationship, above_bt=ge, below_tt=le)
    >>> excluding_bounds = partial(simple_interval_relationship, above_bt=gt, below_tt=lt)

    Take ``(4, 8)`` as the target interval, and want to query the relationship of other
    points and intervals with it.
    No matter what the function is, they will always agree on any intervals that don't
    share any bounds.

    >>> for relation_func in (default, including_bounds, excluding_bounds):
    ...     print (
    ...         relation_func(3, (4, 8)),
    ...         relation_func(5, (4, 8)),
    ...         relation_func(9, (4, 8)),
    ...         relation_func((3, 7), (4, 8)),
    ...         relation_func((5, 7), (4, 8)),
    ...         relation_func((7, 9), (4, 8))
    ... )
    Relations.BEFORE Relations.DURING Relations.AFTER Relations.BEFORE Relations.DURING Relations.AFTER
    Relations.BEFORE Relations.DURING Relations.AFTER Relations.BEFORE Relations.DURING Relations.AFTER
    Relations.BEFORE Relations.DURING Relations.AFTER Relations.BEFORE Relations.DURING Relations.AFTER

    But if the two intervals share some bounds, these functions will diverge.

    >>> for relation_func in (default, including_bounds, excluding_bounds):
    ...     print (
    ...         relation_func(4, (4, 8)),
    ...         relation_func(8, (4, 8)),
    ...         relation_func((4, 7), (4, 8)),
    ...         relation_func((4, 8), (4, 8)),
    ...         relation_func((5, 8), (4, 8))
    ... )
    Relations.DURING Relations.AFTER Relations.DURING Relations.AFTER Relations.AFTER
    Relations.DURING Relations.DURING Relations.DURING Relations.DURING Relations.DURING
    Relations.BEFORE Relations.AFTER Relations.BEFORE Relations.BEFORE Relations.AFTER

    The function can be used with the FIRST argument being a slice object as well.
    This can then be used to enable [i:j] access.

    >>> for relation_func in (default, including_bounds, excluding_bounds):
    ...     print (
    ...         relation_func(slice(4, 7), (4, 8)),
    ...         relation_func(slice(4, 8), (4, 8)),
    ...         relation_func(slice(5, 8), (4, 8))
    ... )
    Relations.DURING Relations.AFTER Relations.AFTER
    Relations.DURING Relations.DURING Relations.DURING
    Relations.BEFORE Relations.BEFORE Relations.AFTER

    """
    if isinstance(x, slice):
        x_bt, x_tt = validate_interval((x.start or 0, x.stop or 0))
    elif isinstance(x, Iterable):
        x_bt, x_tt = validate_interval(x)
    else:
        x_bt, x_tt = x, x  # consider a point to be the (x, x) interval
    y_bt, y_tt = validate_interval(y)
    if not above_bt(x_bt, y_bt):  # meaning x_bt > y_bt (or >=, depending on above_bt)
        return Relations.BEFORE
    elif below_tt(x_tt, y_tt):  # meaning x_bt < y_tt (or <=, depending on below_tt)
        return Relations.DURING
    else:
        return Relations.AFTER


# Error handling #######################################################################################################
class ExceptionRaiserCallbackMixin:
    """Make the instance callable and have the effect of raising the instance.
    Meant to add to an exception class so that instances of this class can be used as callbacks that raise the error"""

    dflt_args = ()

    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            if isinstance(self.dflt_args, str):
                self.dflt_args = (self.dflt_args,)
            args = self.dflt_args
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        raise self


# TODO: Include all 13 of Allen's interval algebra relations? Enum/sentinels and errors for these events?
#  (https://en.wikipedia.org/wiki/Allen%27s_interval_algebra#Relations)
class NotDuringError(ExceptionRaiserCallbackMixin, IndexError):
    """IndexError that indicates that there was an attempt to index some data that is not contained in the buffer
    (i.e. is that a part of the request is NO LONGER, or NOT YET covered by the buffer)"""

    dflt_args = 'Some of the data requested was in the past or in the future'


class OverlapsPastError(NotDuringError):
    """IndexError that indicates that there was an attempt to index some data that is in the PAST
    (i.e. is NO LONGER completely covered by the buffer)"""

    dlft_args = 'Some of the data requested is in the past'


class OverlapsFutureError(NotDuringError):
    """IndexError that indicates that there was an attempt to index some data that is in the FUTURE
    (i.e. is NOT YET completely covered by the buffer)"""

    dlft_args = 'Some of the data requested is in the future'


class RelationNotHandledError(TypeError):
    """TypeError that indicates that a relation is either not a valid one, or not handled by conditional clause."""


not_during_error = NotDuringError()
overlaps_past_error = OverlapsPastError()
overlaps_future_error = OverlapsFutureError()


# The IndexedBuffer, finally #############################################################################################


def _not_implemented(self, method_name, *args, **kwargs):
    raise NotImplementedError('')


# ram heavier, cpu lighter extend
def _extend_cpu_lighter_ram_heavier(self, iterable: Iterable) -> None:
    """Extend buffer with an iterable of items"""
    iterable = list(iterable)
    with self._lock:
        self._deque.extend(iterable)
        self.max_idx += len(iterable)


# cpu heavier, ram lighter extend
def _extend_ram_lighter_cpu_heavier(self, iterable: Iterable) -> None:
    """Extend buffer with an iterable of items"""
    c = count()
    counting_iter = map(itemgetter(0), zip(iterable, c))
    with self._lock:
        self._deque.extend(counting_iter)
        self.max_idx += next(c)


def none_safe_addition(x, y):
    """Adds the two numbers if x is not None, or return None if not"""
    if x is None:
        return None
    else:
        return x + y


def slice_args(slice_obj):
    return slice_obj.start, slice_obj.stop, slice_obj.step


def shift_slice(slice_obj, shift: Number):
    return slice(
        none_safe_addition(slice_obj.start, shift),
        none_safe_addition(slice_obj.stop, shift),
        slice_obj.step,
    )


def absolute_item(item, max_idx):
    """Returns an item with absolute references: i.e. with negative indices idx
    resolved to max_idx + idx

    >>> absolute_item(-1, 10)
    9
    >>> absolute_item(slice(-4, -2, 2), 10)
    slice(6, 8, 2)

    But anything else that's not a slice or int will be left untouched
    (and will probably result in errors if you use with IndexedBuffer)

    >>> absolute_item((-7, -2), 10)
    (-7, -2)

    """
    if isinstance(item, slice):
        start, stop, step = slice_args(item)
        if start is not None and start < 0:
            start = max_idx + start
        if stop is not None and stop < 0:
            stop = max_idx + stop
        return slice(start, stop, step)
    elif isinstance(item, int) and item < 0:
        return item + max_idx
    else:
        return item


class IndexedBuffer:
    """A list-like object that gives a limited-past read view of an unbounded stream

    For example, say we had the stream of increasing integers 0, 1, 2, ...
    that is being fed to indexedBuffer

    What IndexedBuffer(maxlen=4) offers is access to the buffer's contents,
    but using the indices that
    the stream (if it were one big list in memory) would use instead of the buffer's index.
        0 1 2 3 [4 5 6 7] 8 9

    IndexedBuffer uses collections.deque, exposing the append, extend,
    and clear methods, updating the index reference in a thread-safe manner.

    >>> s = IndexedBuffer(buffer_len=4)
    >>> s.extend(range(4))  # adding 4 elements in bulk (filling the buffer completely)
    >>> list(s)
    [0, 1, 2, 3]
    >>> s[2]
    2
    >>> s[1:2]
    [1]
    >>> s[1:1]
    []

    Let's add two more elements (using append this time), making the buffer "shift"

    >>> s.append(4)
    >>> s.append(5)
    >>> list(s)
    [2, 3, 4, 5]
    >>> s[2]
    2
    >>> s[5]
    5
    >>> s[2:5]
    [2, 3, 4]
    >>> s[3:6]
    [3, 4, 5]
    >>> assert s[2:6] == list(range(2, 6))

    You can slice with step:

    >>> s[2:6:2]
    [2, 4]

    You can slice with negatives
    >>> s[2:-2]
    [2, 3]

    On the other hand, if you ask for something that is not in the buffer (anymore, or yet), you'll get an
    error that tells you so:

    >>> # element for idx 1 is missing in [2, 3, 4, 5]
    >>> s[1:4]  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    OverlapsPastError: You asked for slice(1, 4, None), but the buffer only contains the index range: 2:6

    >>> # elements for 0:2 are missing (as well as 6:9, but OverlapsPastError trumps OverlapsFutureError
    >>> s[0:9]  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    OverlapsPastError: You asked for slice(0, 9, None), but the buffer only contains the index range: 2:6

    >>> # element for 6:9 are missing in [2, 3, 4, 5]
    >>> s[4:9]  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    OverlapsFutureError: You asked for slice(4, 9, None), but the buffer only contains the index range: 2:6
    """

    def __init__(
        self,
        buffer_len,
        prefill=(),
        if_overlaps_past=overlaps_past_error,
        if_overlaps_future=overlaps_future_error,
        slice_get_postproc: Callable = list,
    ):
        self._deque = deque(prefill, buffer_len)
        self.buffer_len = self._deque.maxlen
        self.max_idx = 0  # should correspond to the number of items added
        self.if_overlaps_past = if_overlaps_past
        self.if_overlaps_future = if_overlaps_future
        self.slice_get_postproc = slice_get_postproc
        self._lock = Lock()

    def __repr__(self):
        return (
            f'{type(self).__name__}('
            f'buffer_len={self.buffer_len}, '
            f'min_idx={self.min_idx}, '
            f'max_idx={self.max_idx}, ...)'
        )

    def __iter__(self):
        yield from self._deque

    @property
    def min_idx(self):
        return max(self.max_idx - self.buffer_len, 0)

    # TODO: Use singledispathmethod?
    def outer_to_buffer_idx(self, idx):
        if isinstance(idx, slice):
            return shift_slice(idx, -self.min_idx)
        elif isinstance(idx, int):
            if idx >= 0:
                return idx - self.min_idx
            else:  # idx < 0
                return self.buffer_len + idx
        elif isinstance(idx, Iterable):
            return tuple(x - self.min_idx for x in idx)
        else:
            raise TypeError(
                f'{type(idx)} are not handled. '
                f'You requested the outer_to_buffer_idx of {idx}'
            )

    def __getitem__(self, item):
        item = absolute_item(
            item, self.max_idx
        )  # Note: Overhead for convenience of negative indices use (worth it?)
        relationship = simple_interval_relationship(
            item, (self.min_idx, self.max_idx + 1)
        )
        if relationship == Relations.DURING:
            item = self.outer_to_buffer_idx(item)
            if isinstance(item, slice):
                return self.slice_get_postproc(islice(self._deque, *slice_args(item)))
            else:
                try:
                    return self._deque[item]
                except IndexError:
                    if len(self._deque) < self.buffer_len:
                        raise self._overlaps_future_error(item)
                    else:
                        raise
        elif relationship == Relations.AFTER:
            raise self._overlaps_future_error(item)
        elif relationship == Relations.BEFORE:
            raise self._overlaps_past_error(item)
        else:
            raise RelationNotHandledError(
                f'The relation {relationship} is not handled.'
            )

    def _overlaps_past_error(self, item):
        return OverlapsPastError(
            f'You asked for {item}, but the buffer only contains the index range: {self.min_idx}:{self.max_idx}'
        )

    def _overlaps_future_error(self, item):
        return OverlapsFutureError(
            f'You asked for {item}, but the buffer only contains the index range: {self.min_idx}:{self.max_idx}'
        )

    def append(self, x) -> None:
        with self._lock:
            self._deque.append(x)
            self.max_idx += 1

    extend = _extend_ram_lighter_cpu_heavier

    def clear(self):
        with self._lock:
            self._deque.clear()
            self.max_idx = 0

    # # TODO: Sanity check.
    # def __len__(self):
    #     """Length in the sense of "number of items that passed through buffer so far -- not the length of the buff """
    #     return self.max_idx


def consume(gen, n):
    """Consume n iterations of generator (without returning elements)"""
    try:
        for _ in range(n):
            next(gen)
    except StopIteration:
        return None


from dataclasses import dataclass
from typing import Iterator


# TODO: Add some mechanism to deal with ITERABLE instead of just iterator. As it is we have some unwanted behavior with
#   iterables
@dataclass
class InfiniteSeq:
    """A list-like (read) view of an unbounded sequence/stream.

    It is the combination of `IndexedBuffer` and an iterator that will be used to
    source the buffer according to the slices that are requested.

    If a slice is requested whose data is "in the future", the iterator will be
    consumed until the buffer can satisfy that request.
    If the requested slice has any part of it that is "in the past", that is,
    has already been iterated through and is not in the buffer anymore, a
    `OverlapsPastError` will be raised.

    Therefore, `InfiniteSeq` is meant for ordered slice queries of size no more than
    the buffer size.
    If these conditions are satisfied, an `InfiniteSeq` will behave (with `i:j`
    queries) as if it were one long list in memory.

    Can be used with a live stream of data as long as the buffer size is big enough
    to handle the data production and query rates.

    For example, take an iterator that cycles from 0 to 99 forever:

    >>> from itertools import cycle
    >>> iterator = cycle(range(100))

    Let's make an `InfiniteSeq` instance for this stream, accomodating for a view of
    up to 11 items.

    >>> s = InfiniteSeq(iterator, buffer_len=11)

    Let's ask for element 15 (which is the (15 + 1)th element (and should have a value of 15).

    >>> s[15]
    15

    Now, to get this value, the iterator will move forward up to that point;
    that is, until the buffer's head (i.e. most recent) item contains that requested (15 + 1)th element.
    But the buffer is of size 11, so we still have access to a few previous elements:

    >>> s[11]
    11
    >>> s[5:15]
    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    But if we asked for anything before index 5...

    >>> s[2:7]  #doctest: +SKIP
    Traceback (most recent call last):
        ...
    OverlapsPastError: You asked for slice(2, 7, None), but the buffer only contains the index range: 5:16

    So we can't go backwards. But we can always go forwards:

    >>> s[95:105]
    [95, 96, 97, 98, 99, 0, 1, 2, 3, 4]

    You can also use slices with step and with negative integers (referencing the head of the buffer)

    >>> s[120:130:2]
    [20, 22, 24, 26, 28]
    >>> s[120:130]
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    >>> s[-8:-2]
    [22, 23, 24, 25, 26, 27]

    but you cannot slice farther back than the buffer

    >>> try:
    ...     s[-20:-2]
    ... except OverlapsPastError as e:
    ...     msg_text = str(e)
    >>> print(msg_text)
    You asked for slice(110, 128, None), but the buffer only contains the index range: 119:130

    Sometimes the source provides data in chunks. Sometimes these chunks are not even of fixed size.
    In those situations, you can use ``itertools.chain`` to "flatten" the iterator as in the following example:


    >>> from creek.infinite_sequence import InfiniteSeq
    >>> from typing import Mapping
    >>>
    >>> class Source(Mapping):
    ...     n = 100
    ...
    ...     __len__ = lambda self: self.n
    ...
    ...     def __iter__(self):
    ...         yield from range(self.n)
    ...
    ...     def __getitem__(self, k):
    ...         print(f"Asking for {k}")
    ...         return list(range(k * 10, (k + 1) * 10))
    ...
    >>>
    >>> source = Source()
    >>>

    See that when we ask for a chunk of data, there's a print notification about it.

    >>> assert source[3] == [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    Asking for 3

    Now let's make an iterator of the data and an InfiniteSeq (with buffer length 10) on top of it.

    >>> from itertools import chain
    >>> iterator = chain.from_iterable(source.values())
    >>> s = InfiniteSeq(iterator, 10)

    See that when you ask for :5, you see that chunk 0 is requested.

    >>> s[:5]
    Asking for 0
    [0, 1, 2, 3, 4]

    If you ask for something that's already in the buffer, you won't see the print notification though.

    >>> s[4:8]
    [4, 5, 6, 7]

    The following shows you how InfiniteSeq "hits" the data source as it's getting the data it needs for the request.

    >>> s[8:12]
    Asking for 1
    [8, 9, 10, 11]
    >>>
    >>> s[40:42]
    Asking for 2
    Asking for 3
    Asking for 4
    [40, 41]

    """

    iterator: Iterator
    buffer_len: int

    def __post_init__(self):
        self.indexed_buffer = IndexedBuffer(self.buffer_len)

    def __getitem__(self, item):
        if isinstance(item, slice):
            n_ticks_in_the_future = item.stop - self.indexed_buffer.max_idx
            if n_ticks_in_the_future > 0:
                # TODO: If indexed_buffer had a "fast-forward" (perhaps "peek")
                #  we could waste less buffer writes
                self.indexed_buffer.extend(islice(self.iterator, n_ticks_in_the_future))
                # consume(self.iterator, n_ticks_in_the_future)
            return self.indexed_buffer[item]
        elif isinstance(item, int):
            return self[slice(item, item + 1)][0]


def new_type(name, typ, doc=None):
    t = NewType(name, type)
    if doc is not None:
        t.__doc__ = doc
    return t


BufferInput = new_type(
    'BufferInput', Any, 'input_of/what_we_insert_in a buffer (before transformation)'
)

BufferItem = new_type(
    'BufferItem', Any, 'An item of a buffer (after input is transformed)'
)

InputDataTrans = new_type(
    'InputDataTrans',
    Callable[[BufferInput], BufferItem],
    'A function that transforms a BufferInput in to a BufferItem',
)
Query = new_type('Query', Any, 'A query (i.e. key, selection specification, etc.)')
QueryTrans = new_type(
    'QueryTrans',
    Callable[[Query], Query],
    'A function transforming a query into another, ready to be applied form',
)

FiltFunc = Callable[[BufferItem], bool]


def asis(obj):
    return obj


# TODO: Finish up and document
class BufferedGetter:
    """
    `BufferedGetter` is intended to be a more general (but not optimized) class that
    offers a query-interface to a buffer, intended to be used when the buffer is
    being filled by a (possibly live) stream of data items.

    By contrast...
    The `IndexedBuffer` is a particular case where the queries are slices and the index
    that is sliced on is an enumeration one.
    The `InfiniteSeq` is a class combining `IndexedBuffer` with a data source it can
    pull data from (according to the demands of the query).

    >>> from creek.infinite_sequence import BufferedGetter
    >>>
    >>>
    >>> b = BufferedGetter(20)
    >>> b.extend([
    ...     (1, 3, 'completely before'),
    ...     (2, 4, 'still completely before (upper bounds are strict)'),
    ...     (3, 6, 'partially before, but overlaps bottom'),
    ...     (4, 5, 'totally', 'inside'),  # <- note this tuple has 4 elements
    ...     (5, 8),  # <- note this tuple has only the minimum (2) elements,
    ...     (7, 10, 'partially after, but overlaps top'),
    ...     (8, 11, 'completely after (strict upper bound)'),
    ...     (100, 101, 'completely after (obviously)')
    ... ])
    >>> b[lambda x: 3 < x[0] < 8]  # doctest: +NORMALIZE_WHITESPACE
    [(4, 5, 'totally', 'inside'),
     (5, 8),
     (7, 10, 'partially after, but overlaps top')]
    """

    def __init__(
        self,
        buffer_len,
        prefill=(),
        input_data_trans=asis,
        query_trans=asis,
        # if_overlaps_past=overlaps_past_error,
        # if_overlaps_future=overlaps_future_error,
        slice_get_postproc: Callable = list,
    ):
        self._deque = deque(prefill, buffer_len)
        self.buffer_len = self._deque.maxlen
        self.input_data_trans = input_data_trans
        self.query_trans = query_trans
        # self.max_idx = 0  # should correspond to the number of items added
        # self.if_overlaps_past = if_overlaps_past
        # self.if_overlaps_future = if_overlaps_future
        self.slice_get_postproc = slice_get_postproc
        self._lock = Lock()

    def ingress(self, x: BufferInput) -> BufferItem:
        return x

    def append(self, x: BufferInput) -> None:
        with self._lock:
            x = self.input_data_trans(x)
            self._deque.append(x)

    def extend(self, iterable):
        for x in iterable:
            self.append(x)

    def clear(self):
        with self._lock:
            self._deque.clear()

    def __getitem__(self, q: Query):
        return self._getitem(q)

    def _getitem(self, q: Query):
        # note: just a composition
        return self.slice_get_postproc(self.filter(self.query_trans(q)))

    def filter(self, filt: Callable):
        # assert callable(filt), f'filt must be callable, was: {filt}'
        return filter(filt, self._deque)

    def __iter__(self):  # to dodge the iteration falling back to __getitem__(i)
        yield from self._deque
```

## labeling.py

```python
"""
Tools to label/annotate stream elements

The motivating example is the case of an incoming stream
that we need to segment, according to the detection of an event.

For example, take a stream of integers and detect the event "multiple of 5":

.. code-block:: python

    1->2->3->4->'multiple of 5'->6->7->...



When the stream is "live", we don't want to process it immediately, but instead
we prefer to annotate it on the fly, by adding some metadata to it.

The simplest addition of metadata information could look like:

.. code-block:: python

    3->4->('multiple of 5', 5) -> 6 -> ...


One critique of using  tuples to contain both annotated (here, ``5``) and annotation (here, ``'multiple of 5'``)
is that the semantics aren't explicit.
The fact that the original element was annotated the distinction of annotation and annotated is based on an
implicit convention.
This is not too much of a problem here, but becomes unwieldy in more complex situations, for example,
if we want to accommodate for multiple labels.

A ``LabelledElement`` ``x`` has an attribute ``x.element``,
and a container of labels ``x.labels`` (list, set or dict).

``Multilabels`` can be used to segment streams into overlapping segments.

.. code-block:: python

    (group0)->(group0)->(group0, group1)->(group0, group1)-> (group1)->(group1)->...


"""

from typing import NewType, Iterable, Callable, Any, TypeVar, Union
from abc import ABC, abstractmethod

KT = TypeVar('KT')  # Key type.
VT = TypeVar('VT')  # Value type.
Element = NewType('Element', Any)
Label = NewType('Label', Any)
Labels = Iterable[Label]
LabelFactory = Callable[[], Label]
AddLabel = Callable[[Labels, Label], Any]


class LabeledElement(ABC):
    """
    Abstract class to label elements -- that is, associate some metadata to an element.

    To make a concrete LabeledElement, one must subclass LabeledElement and provide

    - a `mk_new_labels_container`, a `LabelFactory`, which is a callable that takes no \
    input and returns a new empty labels container
    - a `add_new_label`, an `AddLabel`, a (Labels, Label) callable that adds a single \
    label to the labels container.
    """

    def __init__(self, element: Element):
        self.element = element
        self.labels = self.mk_new_labels_container()

    @staticmethod
    @abstractmethod
    def mk_new_labels_container(self) -> Labels:
        raise NotImplemented('Need to implement mk_new_labels_container')

    add_new_label: AddLabel

    @staticmethod
    @abstractmethod
    def add_new_label(labels: Labels, label: Label):
        raise NotImplemented('Need to implement add_new_label')

    def __repr__(self):
        return f'{type(self).__name__}({self.element})'

    def add_label(self, label):
        self.add_new_label(self.labels, label)
        return self

    def __contains__(self, label):
        return label in self.labels


class DictLabeledElement(LabeledElement):
    """A LabeledElement that uses a `dict` as the labels container.
    Use this when you need to keep labels classified and have quick access to the
    a specific class of labels.
    Note that when adding a label, you need to specify it as a `{key: val, ...}`
    `dict`, the keys being the (hashable) label kinds,
    and the vals being the values for those kinds.

    >>> x = DictLabeledElement(42).add_label({'string': 'forty-two'})
    >>> x.element
    42
    >>> x.labels
    {'string': 'forty-two'}
    >>> x.add_label({'type': 'number', 'prime': False})
    DictLabeledElement(42)
    >>> x.element
    42
    >>> assert x.labels == {'string': 'forty-two', 'type': 'number', 'prime': False}
    """

    mk_new_labels_container = staticmethod(dict)

    @staticmethod
    def add_new_label(labels: dict, label: dict):
        labels.update(label)


class SetLabeledElement(LabeledElement):
    """A LabeledElement that uses a `set` as the labels container.
    Use this when you want to get fast `label in labels` check and/or maintain the
    labels unduplicated.
    Note that since `set` is the container, the labels will have to be hashable.

    >>> x = SetLabeledElement(42).add_label('forty-two')
    >>> x.element
    42
    >>> x.labels
    {'forty-two'}
    >>> x.add_label('number')
    SetLabeledElement(42)
    >>> x.element
    42
    >>> assert x.labels == {'forty-two', 'number'}
    """

    mk_new_labels_container = staticmethod(set)
    add_new_label = staticmethod(set.add)


class ListLabeledElement(LabeledElement):
    """A LabeledElement that uses a `list` as the labels container.
    Use this when you need to use unhashable labels, or label insertion order matters,
    or don't need fast `label in labels` checks or label deduplication.

    >>> x = ListLabeledElement(42).add_label('forty-two')
    >>> x.element
    42
    >>> x.labels
    ['forty-two']
    >>> x.add_label('number')
    ListLabeledElement(42)
    >>> x.element
    42
    >>> assert x.labels == ['forty-two', 'number']
    """

    mk_new_labels_container = staticmethod(list)
    add_new_label = staticmethod(list.append)


def label_element(
    elem: Union[Element, LabeledElement],
    label: Label,
    labeled_element_cls,  # TODO: LabeledElement annotation makes linter complain!?
) -> LabeledElement:
    """Label `element` with `label` (or add this label to the existing labels).

    The `labeled_element_cls`, the `LabeledElement` class to use to label the element,
    is meant to be "partialized out", like this:

    >>> from functools import partial
    >>> from creek.labeling import DictLabeledElement
    >>> my_label_element = partial(label_element, labeled_element_cls=DictLabeledElement)
    >>> # and then just use my_label_element(elem, label) to label elem

    You'll probably often want to use `DictLabeledElement`, because, for example:

    ```
    {'n_channels': 2, 'phase', 2, 'session': 16987485}
    ```

    is a lot easier (and less dangerous) to use then, say:

    ```
    [2, 2, 16987485]
    ```

    But there are cases where, say:

    >>> from creek.labeling import SetLabeledElement
    >>> my_label_element = partial(label_element, labeled_element_cls=SetLabeledElement)
    >>> x = my_label_element(42, 'divisible_by_seven')
    >>> _ = my_label_element(x, 'is_a_number')
    >>> 'divisible_by_seven' in x  # equivalent to 'divisible_by_seven' in x.labels
    True
    >>> x.labels.issuperset({'is_a_number', 'divisible_by_seven'})
    True

    is more convenient to use then using a dict with boolean values to do the same

    :param elem: The element that is being labeled
    :param label: The label to add to the element
    :param labeled_element_cls: The `LabeledElement` class to use to label the element
    :return:
    """
    if not isinstance(elem, labeled_element_cls):
        return labeled_element_cls(elem).add_label(label)
    else:  # elem is already an labeled_element_cls itself, so
        return elem.add_label(label)
```

## multi_streams.py

```python
"""Tools for multi-streams"""

from itertools import product
from typing import Mapping, Iterable, Any, Optional, Callable
import heapq
from dataclasses import dataclass
from functools import partial
from operator import itemgetter

from creek.util import Pipe, identity_func

StreamsMap = Mapping[Any, Iterable]  # a map of {name: stream} pairs


@dataclass
class MergedStreams:
    """Creates an iterable of ``(stream_id, stream_item)`` pairs from a stream Mapping,
    that is, ``{stream_id: stream, ...}``: A sort of "flattening" of the Mapping.

    This can be useful, for instance, if you want to make "slabs" of data, gathering 
    together all the data for a given time period, from multiple streams.

    The ``stream_item`` will be yield in sorted order.
    Sort behavior can be modified by the ``sort_key`` argument which behaves like ``key``
    arguments of built-in like ``sorted``, ``heapq.merge``, ``itertools.groupby``, etc.

    If given, the `sort_key` function applies to ``stream_item`` (not to ``stream_id``).

    Important: To function as expected, the streams should be already sorted (according
    to the ``sort_key`` order).

    The cannonical use case of this function is to "flatten", or "weave together"
    multiple streams of timestamped data. We're given several streams that provide
    ``(timestamp, data)`` items (where timestamps arrive in order within each stream)
    and we get a single stream of ``(stream_id, (timestamp, data))`` items where
    the ``timestamp``s are yield in sorted order.

    The following example uses a dict pointing to a fixed-size list as the ``stream_map``
    but in general the ``stream_map`` will be a ``Mapping`` (not necessarily a dict)
    whose values are potentially bound-less streams.

    >>> streams_map = {
    ...     'hello': [(2, 'two'), (3, 'three'), (5, 'five')],
    ...     'world': [(0, 'zero'), (1, 'one'), (3, 'three'), (6, 'six')]
    ... }
    >>> streams_items = MergedStreams(streams_map)
    >>> it = iter(streams_items)
    >>> list(it)  # doctest: +NORMALIZE_WHITESPACE
    [('world', (0, 'zero')),
     ('world', (1, 'one')),
     ('hello', (2, 'two')),
     ('hello', (3, 'three')),
     ('world', (3, 'three')),
     ('hello', (5, 'five')),
     ('world', (6, 'six'))]
    """

    streams_map: StreamsMap
    sort_key: Optional[Callable] = None

    def __post_init__(self):
        if self.sort_key is None:
            self.effective_sort_key = itemgetter(1)
        else:
            self.effective_sort_key = Pipe(itemgetter(1), self.sort_key)

    def __iter__(self):
        for item in heapq.merge(
            *multi_stream_items(self.streams_map), key=self.effective_sort_key
        ):
            yield item


def multi_stream_items(streams_map: StreamsMap):
    """Provides a iterable of (k1, v1_1), (k1, v1_2), ...
    
    >>> streams_map = {'hello': 'abc', 'world': [1, 2]}
    >>> hello_items, world_items = multi_stream_items(streams_map)
    >>> list(hello_items)
    [('hello', 'a'), ('hello', 'b'), ('hello', 'c')]
    >>> list(world_items)
    [('world', 1), ('world', 2)]
    """
    for stream_id, stream in streams_map.items():
        yield product([stream_id], stream)


def transform_methods(cls, method_trans=staticmethod):
    """Applies method_trans to all the methods of `cls`

    >>> from functools import partial
    >>> staticmethods = partial(transform_methods, method_trans=staticmethod)

    Now staticmethods is a class decorator that can be used to make all methods
    be defined as staticmethods in bulk

    >>> @staticmethods
    ... class C:
    ...     foo = lambda x: x + 1
    ...     bar = lambda y: y * 2
    >>> c = C()
    >>> c.foo(6)
    7
    >>> c.bar(6)
    12
    """
    for attr_name in vars(cls):
        attr = getattr(cls, attr_name)
        if callable(attr):
            setattr(cls, attr_name, method_trans(attr))

    return cls


staticmethods = partial(transform_methods, method_trans=staticmethod)


@staticmethods
class SortKeys:
    all_but_last = itemgetter(-1)
    second_item = itemgetter(1)
```

## scrap/__init__.py

```python

```

## scrap/annotators.py

```python
"""Tools to make annotators.

"""

from typing import Union, Iterable, Tuple, Callable, KT, VT
from operator import itemgetter
from functools import partial
from i2 import Pipe

# ------------------- Types -------------------
# TODO: Should we use the term KV (key-value-pair) instead of annotation?
# TODO: Should we use some Time (numerical) type instead of KT here?
IndexAnnot = Tuple[KT, VT]
IndexAnnot.__doc__ = '''An annotation whose key is an (time) index. 
KT is usually numerical and represents time. 
VT holds the value (info) of the annotation.
'''

Interval = Tuple[KT, KT]  # note this it's two KTs here, usually numerical.
Interval.__doc__ = '''An interval; i.e. a pair of indices'''

IntervalAnnot = Tuple[Interval, VT]
IntervalAnnot.__doc__ = '''An annotation whose key is an interval.'''

KvExtractor = Callable[[Iterable], Iterable[IndexAnnot]]
IntervalAnnot.__doc__ = '''A function that extracts annotations from an iterable.'''

FilterFunc = Callable[..., bool]


def always_true(x):
    return True


# ------------------- Annotators -------------------
def track_intervals(
    indexed_tags: Iterable[IndexAnnot], track_tag: FilterFunc = always_true
) -> Iterable[IntervalAnnot]:
    """Track intervals of tags in an iterable of indexed tags.

    Example usage:

    >>> iterable = ['a', 'b', 'a', 'b', 'c', 'c', 'd', 'd']
    >>> list(track_intervals(enumerate(iterable)))
    [((0, 2), 'a'), ((1, 3), 'b'), ((4, 5), 'c'), ((6, 7), 'd')]
    >>> list(track_intervals(enumerate(iterable), track_tag=lambda x: x in {'a', 'd'}))
    [((0, 2), 'a'), ((6, 7), 'd')]

    """
    open_tags = {}
    for index, tag in indexed_tags:
        if track_tag(tag):
            if tag not in open_tags:
                open_tags[tag] = index
            else:
                yield ((open_tags[tag], index), tag)
                del open_tags[tag]


def mk_interval_extractor(
    *,
    kv_extractor: KvExtractor = enumerate,
    include_tag: bool = True,
    track_tag: FilterFunc = always_true
) -> Callable[[Iterable], Iterable[Union[Interval, IntervalAnnot]]]:
    """Make an interval extractor from a key-value extractor.

    Example usage:

    >>> iterable = ['a', 'b', 'a', 'b', 'c', 'c', 'd', 'd']
    >>> extract_intervals = mk_interval_extractor()
    >>> list(extract_intervals(iterable))
    [((0, 2), 'a'), ((1, 3), 'b'), ((4, 5), 'c'), ((6, 7), 'd')]
    >>> extract_intervals_without_tags = mk_interval_extractor(include_tag=False)
    >>> list(extract_intervals_without_tags(iterable))
    [(0, 2), (1, 3), (4, 5), (6, 7)]

    See:

    """
    interval_extractor = Pipe(
        kv_extractor, partial(track_intervals, track_tag=track_tag)
    )
    if not include_tag:
        only_interval = partial(map, itemgetter(0))
        interval_extractor = Pipe(interval_extractor, only_interval)
    return interval_extractor
```

## scrap/async_utils.py

```python
"""Utils to deal with async iteration


Making singledispatch work:

.. code-block:: python

    from typing import Generator, Iterator, Iterable, AsyncIterable, AsyncIterator
    from functools import singledispatch, partial

    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class IterableType(Protocol):
        def __iter__(self):
            pass

    @runtime_checkable
    class CursorFunc(Protocol):
        def __call__(self):
            pass

    @singledispatch
    def to_iterator(x: IterableType):
        return iter(x)

    will_never_happen = object()

    @to_iterator.register
    def _(x: CursorFunc):
        return iter(x, will_never_happen)

    assert list(to_iterator([1,2,3])) == [1, 2, 3]

    f = partial(next, iter([1,2,3]))
    assert list(to_iterator(f)) == [1, 2, 3]

Trying to make async iterators/iterables/cursor_funcs utils

.. code-block:: python

    import asyncio


    async def ticker(to=3, delay=0.5):
        # Yield numbers from 0 to `to` every `delay` seconds.
        for i in range(to):
            yield i
            await asyncio.sleep(delay)

    async def my_aiter(async_iterable):
        async for i in async_iterable:
            yield i

    t = [i async for i in my_aiter(ticker(3, 0.2))]
    assert t == [0, 1, 2]

    # t = list(my_aiter(ticker(3, 0.2)))
    # # TypeError: 'async_generator' object is not iterable
    # # and
    # t = await list(ticker(3, 0.2))
    # # TypeError: 'async_generator' object is not iterable

    # But...

    async def alist(async_iterable):
        return [i async for i in async_iterable]

    t = await alist(ticker(3, 0.2))
    assert t == [0, 1, 2]


"""

from functools import partial
from typing import (
    Callable,
    Any,
    NewType,
    Iterable,
    AsyncIterable,
    Iterator,
    AsyncIterator,
    Union,
)

IterableType = Union[Iterable, AsyncIterable]
IteratorType = Union[Iterator, AsyncIterator]
CursorFunc = NewType('CursorFunc', Callable[[], Any])
CursorFunc.__doc__ = "An argument-less function returning an iterator's values"


# ---------------------------------------------------------------------------------------
# iteratable, iterator, cursors
no_sentinel = type('no_sentinel', (), {})()

try:
    aiter  # exists in python 3.10+
    # Note: doesn't have the sentinel though!!
except NameError:

    async def aiter(iterable: AsyncIterable) -> AsyncIterator:
        if not isinstance(iterable, AsyncIterable):
            raise TypeError(f'aiter expected an AsyncIterable, got {type(iterable)}')
        if isinstance(iterable, AsyncIterator):
            return iterable
        return (i async for i in iterable)


async def aiter_with_sentinel(cursor_func: CursorFunc, sentinel: Any) -> AsyncIterator:
    """Like iter(async_callable, sentinel) builtin but for async callables"""
    while (value := await cursor_func()) is not sentinel:
        yield value


def iterable_to_iterator(iterable: IterableType) -> IteratorType:
    """Get an iterator from an iterable (whether async or not)

    >>> iterable = [1, 2, 3]
    >>> iterator = iterable_to_iterator(iterable)
    >>> assert isinstance(iterator, Iterator)
    >>> assert list(iterator) == iterable
    """
    if isinstance(iterable, AsyncIterable):
        return aiter(iterable)
    return iter(iterable)


def iterator_to_cursor(iterator: Iterator) -> CursorFunc:
    return partial(next, iterator)


def cursor_to_iterator(cursor: CursorFunc, sentinel=no_sentinel) -> Iterator:
    return iter(cursor, no_sentinel)


def iterable_to_cursor(iterable: Iterable, sentinel=no_sentinel) -> CursorFunc:
    iterator = iterable_to_iterator(iterable)
    if sentinel is no_sentinel:
        return iterator_to_cursor(iterator)
    else:
        return partial(next, iterator, sentinel)
```

## scrap/creek_layers.py

```python
"""
Wrapper interfaces
##################

Inner-class
***********

.. code-block:: python

   def intify(self, data):
       return tuple(map(int, data))


.. code-block:: python

    class D(Creek):
        # subclassing `CreekLayer` indicates that this class is a layering class
        # could also use decorator for this: Allowing simple injection of external classes
        class Lay(CreekLayer):
            # name indicates what kind of layer this is (i.e. where/how to apply it)
            def pre_iter(stream):
                next(stream)  # skip one

            @data_to_obj  # decorator to indicate what kind of layer this is (i.e. where/how to apply it
            def strip_and_split(data):  # function can be a method (first arg is instance) or static (data_to_obj figures it out)
                return data.strip().split(',')

            another_data_to_obj_layer = data_to_obj(intify)  # decorator can be used to inject function defined externally


Decorators
**********

.. code-block:: python

    @lay(kind='pre_iter', func=func)
    @lay.data_to_obj(func=func)
    class D(Creek):
        pass


Fluid interfaces
****

.. code-block:: python

    D = (Creek
        .lay('pre_iter', func)
        .lay.data_to_obj(func)...
    )



Backend
******

Use lists to stack layers.

Compile the layers to increase resource use.

Uncompile to increase debugibility.
"""

from typing import Iterable, Callable
from dataclasses import dataclass
from inspect import Signature, signature


def identity_func(x):
    return x


static_identity_method = staticmethod(identity_func)


class Compose:
    def __init__(self, *funcs, default=identity_func):
        if len(funcs) == 0:
            self.first_func = (default,)
            self.other_funcs = ()
        else:
            self.first_func, *self.other_funcs = funcs
        # The following so that __call__ gets the same signature as first_func:
        self.__signature__ = Signature(
            list(signature(self.first_func).parameters.values())
        )

    def __call__(self, *args, **kwargs):
        out = self.first_func(*args, **kwargs)
        for func in self.other_funcs:
            out = func(out)
        return out


FuncSequence = Iterable[Callable]


# TODO: Use descriptors to manage the pre_iters/pre_iter relationship.
@dataclass
class CreekLayer:
    pre_iters: FuncSequence = ()
    data_to_objs: FuncSequence = ()
    post_iters: FuncSequence = ()

    def pre_iter(self, data_stream):
        return Compose(*self.pre_iters)(data_stream)

    def data_to_obj(self, data):
        return Compose(*self.data_to_objs)(data)

    def post_iter(self, obj_stream):
        return Compose(*self.post_iters)(obj_stream)

    def lay(self, **kwargs):
        pass
```

## scrap/multi_streams.py

```python
from creek.multi_streams import *

from warnings import warn

warn(f'Moved to creek.multi_streams')

from dataclasses import dataclass

from typing import Any, NewType, Callable, Iterable, Union, Tuple
from numbers import Number


def MyType(
    name: str,
    tp: type = Any,
    doc: Optional[str] = None,
    aka: Optional[Union[str, Iterable[str]]] = None,
):
    """Like `typing.NewType` with some extras (`__doc__` and `_aka` attributes, etc.)
    """

    new_tp = NewType(name, tp)
    if doc is not None:
        new_tp.__doc__ = doc
    if aka is not None:
        new_tp._aka = aka
    return new_tp


TimeIndex = MyType(
    'TimeIndex',
    Number,
    doc='A number indexing time. Could be in an actual time unit, or could just be '
    'an enumerator (i.e. "ordinal time")',
)
BT = MyType(
    'BT',
    TimeIndex,
    doc='TimeIndex for the lower bound of an interval of time. '
    'Stands for "Bottom Time". By convention, a BT is inclusive.',
)
TT = MyType(
    'TT',
    TimeIndex,
    doc='TimeIndex for the upper bound of an interval of time. '
    'Stands for "Upper Time". By convention, a TT is exclusive.',
)
IntervalTuple = MyType(
    'IntervalTuple',
    Tuple[BT, TT],
    doc='Denotes an interval of time by specifying the (BT, TT) pair',
)
IntervalSlice = MyType(
    'IntervalSlice',
    slice,  # Note: extra condition: non-None .start and .end, and no .step
    doc='Denotes an interval of time by specifying the (BT, TT) pair',
)

Intervals = Iterable[IntervalTuple]
BufferedIntervals = Intervals
RetrievedIntervals = Intervals
QueryInterval = Intervals
IntervalsRetriever = Callable[[QueryInterval, BufferedIntervals], RetrievedIntervals]


@dataclass
class IntervalSlicer:
    match_intervals: IntervalsRetriever
```

## scrap/sequences.py

```python
"""
Sequence data object layer.

It's not clear that it's really a ``Sequence`` that we want, in the sense of
``collections.abc`` (https://docs.python.org/3/library/collections.abc.html).
Namely, we don't care about ``Reversible`` in most cases.
Perhaps we need an intermediate type?

"""

from collections.abc import Sequence


# No wrapper hooks have been added (yet)
# This was just to verify that ``Sequence`` mixins can resolve all methods from
# just __len__ and __getitem__ (but how, I don't know yet, seems it's burried in C)
class Seq(Sequence):
    def __init__(self, seq):
        self.seq = seq

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, item):
        return self.seq[item]
```

## tests/__init__.py

```python

```

## tests/automatas.py

```python
"""Test automatas."""
import pytest
from creek.automatas import (
    mapping_to_transition_func,
    State,
    Symbol,
    basic_automata,
    BasicAutomata,
)


@pytest.mark.parametrize('automata', [basic_automata, BasicAutomata])
def test_automata(automata):
    # A automata that even number of 0s and 1s
    transition_func = mapping_to_transition_func(
        {('even', 0): 'odd', ('odd', 1): 'even',},
        strict=False,  # so that all (state, symbol) combinations not listed have no effect
    )
    fa = automata(transition_func)
    symbols = [0, 1, 0, 1, 1]
    it = fa('even', symbols)
    next(it) == 'even'  # reads 0, stays on state 'even'
    next(it) == 'odd'  # reads 1, applies the ('even', 0): 'odd' transition rule
    next(it) == 'odd'  # reads 0, stays on state 'odd'
    next(it) == 'even'  # reads 1, applies the ('odd', 1): 'even' transition rule
    next(it) == 'odd'  # reads 1, applies the ('even', 0): 'odd' transition rule

    # We can feed the automata with any iterable, and gather the "trace" of all the states
    # it went through like follows. Notice that we can really put any symbol in the
    # iterable, not just 0s and 1s. The automata will just ignore the symbols (remainingin
    # in the same state) if there is no transition for the current state and the symbol.
    assert list(fa('even', [0, 1, 0, 42, 'not_even_a_number', 1])) == [
        'odd',
        'even',
        'odd',
        'odd',
        'odd',
        'even',
    ]

    # When the automata only works on a finite set of states and symbols, we call it
    # a "finite automata". Above, we could have specified strict=True and explicitly
    # list all possible combinations in the transition mapping to get a finite automata.
    # Here's an example where neither states nor symbols have to be from a finite set
    def my_special_transition(state: State, symbol: Symbol) -> State:
        if state in symbol or symbol in state:
            # if the symbol is part of, or contains the state, we return the symbol
            return symbol
        else:  # if not we remain in the same state
            return state

    fa = automata(my_special_transition)
    symbols = ['a', 'b', 'c', 'ab', 'abc', 'abdc', 'abcd']
    assert list(fa('a', symbols)) == ['a', 'a', 'a', 'ab', 'abc', 'abc', 'abcd']
```

## tests/infinite_sequence.py

```python
import pytest
from creek.infinite_sequence import (
    InfiniteSeq,
    IndexedBuffer,
    OverlapsFutureError,
    OverlapsPastError,
)


def test_infinite_seq():
    from itertools import cycle

    iterator = cycle(range(100))
    # Let's make an InfiniteSeq instance for this stream, accomodating for a view of up to 11 items.
    s = InfiniteSeq(iterator, buffer_len=11)
    # Let's ask for element 15 (which is the (15 + 1)th element (and should have a value of 15).
    assert s[15] == 15
    # Now, to get this value, the iterator will move forward up to that point;
    # that is, until the buffer's head (i.e. most recent) item contains that requested (15 + 1)th element.
    # But the buffer is of size 11, so we still have access to a few previous elements:
    assert s[11] == 11
    assert s[5:15] == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    # But if we asked for anything before index 5...
    with pytest.raises(OverlapsPastError):
        _ = s[2:7]

    # So we can't go backwards. But we can always go forwards:

    assert s[95:105] == [95, 96, 97, 98, 99, 0, 1, 2, 3, 4]

    # You can also use slices with step and with negative integers (referencing the head of the buffer)
    assert s[120:130:2] == [20, 22, 24, 26, 28]
    assert s[-8:-2] == [22, 23, 24, 25, 26, 27]

    # What to do if your iterator provides "chunks"? Example below.
    from itertools import chain

    data_gen_source = [
        range(0, 5),
        range(5, 12),
        range(12, 22),
        range(22, 29),
    ]

    data_gen = lambda: chain.from_iterable(data_gen_source)
    assert list(data_gen()) == list(range(29))

    s = InfiniteSeq(iterator=data_gen(), buffer_len=5)
    assert s[2:6] == [2, 3, 4, 5]
    assert s[1] == 1  # still in the buffer
    assert s[10:15] == [10, 11, 12, 13, 14]
    assert s[23:24] == [23]


def test_indexed_buffer_common_case():
    # from creek.scrap.infinite_sequence import InfiniteSeq
    s = IndexedBuffer(maxlen=4)
    s.extend(range(4))
    assert list(s) == [0, 1, 2, 3]
    assert s[2] == 2
    assert s[1:2] == [1]
    assert s[1:1] == []
    s.append(4)
    s.append(5)
    assert list(s) == [2, 3, 4, 5]
    assert s[2] == 2
    assert s[5] == 5
    assert s[2:5] == [2, 3, 4]
    assert s[3:6] == [3, 4, 5]
    assert s[2:6] == list(range(2, 6))
    with pytest.raises(OverlapsPastError) as excinfo:
        s[1:4]  # element for idx 1 is missing in [2, 3, 4, 5]
        assert 'in the past' in excinfo.value
    with pytest.raises(OverlapsPastError) as excinfo:
        s[
            0:9
        ]  # elements for 0:2 are missing (as well as 6:9, but OverlapsPastError trumps OverlapsFutureError
    with pytest.raises(OverlapsFutureError) as excinfo:
        s[4:9]  # element for 6:9 are missing in [2, 3, 4, 5]
        assert 'in the future' in excinfo.value


# simple_test()


def test_indexed_buffer_extreme_cases():
    s = IndexedBuffer(maxlen=7)
    # when there's nothing and you ask for something
    with pytest.raises(OverlapsFutureError):
        s[0]
    with pytest.raises(OverlapsFutureError):
        s[:3]

    # when there's something, but buffer is not full, but you ask for something that hasn't happened yet.
    s.extend(range(5))  # buffer now has the 0:5 view (but is not full!)
    assert list(s) == [0, 1, 2, 3, 4]  # this is what's in the buffer now
    with pytest.raises(OverlapsFutureError) as excinfo:
        s[
            10:14
        ]  # completely in the future (0:5 "happens before" 10:14 (Allen's interval algebra terminology))
        assert 'in the future' in excinfo.value
    with pytest.raises(OverlapsFutureError) as excinfo:
        s[3:7]  # overlaps with 0:5
        assert 'in the future' in excinfo.value

    s.extend(range(5, 10))  # add more data (making the buffer full and shifted)
    assert list(s) == [3, 4, 5, 6, 7, 8, 9]  # this is what's in the buffer now
    assert s[3:9:2] == [3, 5, 7], "slices with steps don't work"

    # use negative indices
    assert s[4:-1] == [4, 5, 6, 7, 8]
    assert s[-4:-1] == [6, 7, 8]


def test_source(capsys):
    from creek.infinite_sequence import InfiniteSeq
    from collections import Mapping

    def assert_prints(print_str):
        out, err = capsys.readouterr()
        assert out == print_str

    class Source(Mapping):
        n = 100

        __len__ = lambda self: self.n

        def __iter__(self):
            yield from range(self.n)

        def __getitem__(self, k):
            print(f'Asking for {k}')
            return list(range(k * 10, (k + 1) * 10))

    source = Source()

    assert source[3] == [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    assert_prints('Asking for 3\n')

    from itertools import chain

    iterator = chain.from_iterable(source.values())

    s = InfiniteSeq(iterator, 10)

    assert s[:5] == [0, 1, 2, 3, 4]
    assert_prints('Asking for 0\n')

    assert s[4:8] == [4, 5, 6, 7]

    assert s[8:12] == [8, 9, 10, 11]
    assert_prints('Asking for 1\n')

    assert s[40:42] == [40, 41]
    assert_prints('Asking for 2\nAsking for 3\nAsking for 4\n')
```

## tests/labeling.py

```python
import pytest
from creek.labeling import (
    DictLabeledElement,
    SetLabeledElement,
    ListLabeledElement,
)


def test_dummy():
    assert True


def test_dict_labeledelement():
    x = DictLabeledElement(42).add_label({'string': 'forty-two'})
    assert x.element == 42
    assert x.labels == {'string': 'forty-two'}
    x.add_label({'type': 'number', 'prime': False})
    assert x.element == 42
    assert x.labels == {
        'string': 'forty-two',
        'type': 'number',
        'prime': False,
    }


def test_set_labeledelement():
    x = SetLabeledElement(42).add_label('forty-two')
    assert x.element == 42
    assert x.labels == {'forty-two'}
    x.add_label('number')
    assert x.element == 42
    assert x.labels == {'forty-two', 'number'}


def test_list_labeledelement():
    x = ListLabeledElement(42).add_label('forty-two')
    assert x.element == 42
    assert x.labels == ['forty-two']
    x.add_label('number')
    assert x.element == 42
    assert x.labels == ['forty-two', 'number']
```

## tools.py

```python
"""Tools to work with creek objects"""

import time
from collections import deque
from typing import (
    Tuple,
    TypeVar,
    Callable,
    Any,
    Iterable,
    Sequence,
    cast,
)
from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from functools import partial

from creek.util import Pipe

Index = Any
DataItem = TypeVar('DataItem')
# TODO: Could have more args. How to specify this in typing?
IndexUpdater = Callable[[Index, DataItem], Index]
Indexer = Callable[[DataItem], Tuple[Index, DataItem]]
T = TypeVar('T')


# TODO: Possible performance enhancement by using class with precompiled slices
# TODO: Compare with apply_and_fanout (no cast here, and slice instead of :)
# Note: I hesitated to make the signature (func, apply_to_idx, seq) instead
# Note: This would have allowed to use partial(apply_func_to_index, func, idx)
# Note: instead, but NOT partial(apply_func_to_index, func=func, apply_to_idx=idx)!!
def apply_func_to_index(seq, apply_to_idx, func):
    """
    >>> apply_func_to_index([1, 2, 3], 1, lambda x: x * 10)
    (1, 20, 3)

    If you're going to apply the same function to the same index, you might
    want to partialize ``apply_func_to_index`` to be able to reuse it simply:

    >>> from functools import partial
    >>> f = partial(apply_func_to_index, apply_to_idx=0, func=str.upper)
    >>> list(map(f, ['abc', 'defgh']))
    [('A', 'b', 'c'), ('D', 'e', 'f', 'g', 'h')]

    """
    apply_to_element, *_ = seq[slice(apply_to_idx, apply_to_idx + 1)]
    return tuple(
        chain(
            seq[slice(None, apply_to_idx)],
            [func(apply_to_element)],
            seq[slice(apply_to_idx + 1, None)],
        )
    )


def apply_and_fanout(
    seq: Sequence, func: Callable[[Any], Iterable], idx: int
) -> Iterable[tuple]:
    """Apply function (that returns an Iterable) to an element of a sequence
    and fanout (broadcast) the resulting items, to produce an iterable of tuples each
    containing
    one of these items along with 'a copy' of the other tuple elements

    >>> list(apply_and_fanout([1, 'abc', 3], iter, 1))
    [(1, 'a', 3), (1, 'b', 3), (1, 'c', 3)]
    >>> list(apply_and_fanout(['bob', 'alice', 2], lambda x: x * ['hi'], 2))
    [('bob', 'alice', 'hi'), ('bob', 'alice', 'hi')]
    >>> list(apply_and_fanout(["bob", "alice", 2], lambda x: x.upper(), 1))
    [('bob', 'A', 2), ('bob', 'L', 2), ('bob', 'I', 2), ('bob', 'C', 2), ('bob', 'E', 2)]

    See Also:
        ``fanout_and_flatten`` and ``fanout_and_flatten_dicts``
    """
    seq = tuple(seq)  # TODO: Overhead: Should we impose seq to be tuple?
    # TODO: See how apply_func_to_index takes care of this problem with chain
    left_seq = seq[0 : max(idx, 0)]  # TODO: Use seq[None:idx] instead?
    right_seq = seq[(idx + 1) :]
    return (left_seq + (item,) + right_seq for item in func(seq[idx]))


def fanout_and_flatten(iterable_of_seqs, func, idx, aggregator=chain.from_iterable):
    """Apply apply_and_fanout to an iterable of sequences.

    >>> seq_iterable = [('abcdef', 'first'), ('ghij', 'second')]
    >>> func = lambda a: zip(*([iter(a)] * 2))  # func is a chunker
    >>> assert list(fanout_and_flatten(seq_iterable, func, 0)) == [
    ...     (('a', 'b'), 'first'),
    ...     (('c', 'd'), 'first'),
    ...     (('e', 'f'), 'first'),
    ...     (('g', 'h'), 'second'),
    ...     (('i', 'j'), 'second')
    ... ]
    """
    apply = partial(apply_and_fanout, func=func, idx=idx)
    return aggregator(map(apply, iterable_of_seqs))


def fanout_and_flatten_dicts(
    iterable_of_dicts, func, fields, idx_field, aggregator=chain.from_iterable
):
    """Apply apply_and_fanout to an iterable of dicts.

    >>> iterable_of_dicts = [
    ...     {'wf': 'abcdef', 'tag': 'first'}, {'wf': 'ghij', 'tag': 'second'}
    ... ]
    >>> func = lambda a: zip(*([iter(a)] * 2))  # func is a chunker
    >>> fields = ['wf', 'tag']
    >>> idx_field = 'wf'
    >>> assert list(
    ...     fanout_and_flatten_dicts(iterable_of_dicts, func, fields, idx_field)) == [
    ...         {'wf': ('a', 'b'), 'tag': 'first'},
    ...         {'wf': ('c', 'd'), 'tag': 'first'},
    ...         {'wf': ('e', 'f'), 'tag': 'first'},
    ...         {'wf': ('g', 'h'), 'tag': 'second'},
    ...         {'wf': ('i', 'j'), 'tag': 'second'}
    ... ]
    """
    egress = Pipe(partial(zip, fields), dict)
    return map(
        egress,
        fanout_and_flatten(
            map(itemgetter(*fields), iterable_of_dicts),
            func=func,
            idx=fields.index(idx_field),
            aggregator=aggregator,
        ),
    )


def filter_and_index_stream(
    stream: Iterable, data_item_filt, timestamper: Indexer = enumerate
):
    """Index a stream and filter it (based only on the data items).

    >>> assert (
    ... list(filter_and_index_stream('this  is   a   stream', data_item_filt=' ')) == [
    ... (0, 't'),
    ... (1, 'h'),
    ... (2, 'i'),
    ... (3, 's'),
    ... (6, 'i'),
    ... (7, 's'),
    ... (11, 'a'),
    ... (15, 's'),
    ... (16, 't'),
    ... (17, 'r'),
    ... (18, 'e'),
    ... (19, 'a'),
    ... (20, 'm')
    ... ])

    >>> list(filter_and_index_stream(
    ...     [1, 2, 3, 4, 5, 6, 7, 8],
    ...     data_item_filt=lambda x: x % 2))
    [(0, 1), (2, 3), (4, 5), (6, 7)]
    """
    if not callable(data_item_filt):
        sentinel = data_item_filt
        data_item_filt = lambda x: x != sentinel
    return filter(lambda x: data_item_filt(x[1]), timestamper(stream))


# TODO: Refactor dynamic indexing set up so that
#  IndexUpdater = Callable[[DataItem, Index, ...], Index] (data and index inversed)
#  Rationale: One can (and usually would want to) have a default current_idx, which
#  can be used as the start index too.
count_increments: IndexUpdater


def count_increments(current_idx: Index, obj: DataItem, step=1):
    return current_idx + step


size_increments: IndexUpdater


def size_increments(current_idx, obj: DataItem, size_func=len):
    return current_idx + size_func(obj)


current_time: IndexUpdater


def current_time(current_idx, obj):
    """Doesn't even look at current_idx or obj. Just gives the current time"""
    return time.time()


@dataclass
class DynamicIndexer:
    """
    :param start: The index to start at (the first data item will have this index)
    :param idx_updater: The (Index, DataItem) -> Index

    Let's take a finite stream of finite iterables (strings here):

    >>> stream = ['stream', 'of', 'different', 'sized', 'chunks']

    The default ``DynamicIndexer`` just does what ``enumerate`` does:

    >>> counter_index = DynamicIndexer()
    >>> list(map(counter_index, stream))
    [(0, 'stream'), (1, 'of'), (2, 'different'), (3, 'sized'), (4, 'chunks')]

    That's because it uses the default ``idx_updater`` function just increments by one.
    This function, DynamicIndexer.count_increments, is shown below

    >>> def count_increments(current_idx, data_item, step=1):
    ...     return current_idx + step

    To get the index starting at 10, we can specify ``start=10``, and to step the
    index by 3 we can partialize ``count_increments``:

    >>> from functools import partial
    >>> step3 = partial(count_increments, step=3)
    >>> list(map(DynamicIndexer(start=10, idx_updater=step3), stream))
    [(10, 'stream'), (13, 'of'), (16, 'different'), (19, 'sized'), (22, 'chunks')]

    You can specify any custom ``idx_updater`` you want: The requirements being that
    this function should take ``(current_idx, data_item)`` as the input, and
    return the next "current index", that is, what the index of the next data item will
    be.
    Note that ``count_increments`` ignored the ``data_item`` completely, but sometimes
    you want to take the data item into account.
    For example, your data item may contain several elements, and you want your
    index to index these elements, therefore you should update your index by
    incrementing it with the number of elements.

    We have ``DynamicIndexer.size_increments`` for that, the code is shown below:

    >>> def size_increments(current_idx, data_item, size_func=len):
    ...     return current_idx + size_func(data_item)
    >>> size_index = DynamicIndexer(idx_updater=DynamicIndexer.size_increments)
    >>> list(map(size_index, stream))
    [(0, 'stream'), (6, 'of'), (8, 'different'), (17, 'sized'), (22, 'chunks')]

    Q: What if I want the index of a data item to be a function of the data item itself?

    A: Then you would use that function to make the ``(idxof(data_item), data_item)``
    pairs directly. ``DynamicIndexer`` is for the use case where the index of an item
    depends on the (number of, sizes of, etc.) items that came before it.

    """

    start: Index = 0
    idx_updater: IndexUpdater = count_increments

    count_increments = staticmethod(count_increments)
    size_increments = staticmethod(size_increments)

    def __post_init__(self):
        self.current_idx = self.start

    def __call__(self, x):
        _current_idx = self.current_idx
        self.current_idx = self.idx_updater(_current_idx, x)
        return _current_idx, x


def dynamically_index(iterable: Iterable, start=0, idx_updater=count_increments):
    """Generalization of `enumerate(iterable)` that allows one to specify how the
    indices should be updated.

    The default is the sae behavior as `enumerate`: Starts with 0 and increments by 1.

    >>> stream = ['stream', 'of', 'different', 'sized', 'chunks']
    >>> assert (list(dynamically_index(stream, start=2))
    ...     == list(enumerate(stream, start=2))
    ...     == [(2, 'stream'), (3, 'of'), (4, 'different'), (5, 'sized'), (6, 'chunks')]
    ... )

    Say we wanted to increment the indices according to the size of the last item
    instead of just incrementing by 1 at every iteration tick...

    >>> def size_increments(current_idx, data_item, size_func=len):
    ...     return current_idx + size_func(data_item)
    >>> size_index = DynamicIndexer(idx_updater=DynamicIndexer.size_increments)
    >>> list(map(size_index, stream))
    [(0, 'stream'), (6, 'of'), (8, 'different'), (17, 'sized'), (22, 'chunks')]

    """
    dynamic_indexer = DynamicIndexer(start, idx_updater)
    return map(dynamic_indexer, iterable)


# Alternative to the above implementation:

from itertools import accumulate


def _dynamic_indexer(stream, idx_updater: IndexUpdater = count_increments, start=0):
    index_func = partial(accumulate, func=idx_updater, initial=start)
    obj = zip(index_func(stream), stream)
    return obj


def alt_dynamically_index(idx_updater: IndexUpdater = count_increments, start=0):
    """Alternative to dynamically_index using itertools and partial

    >>> def size_increments(current_idx, data_item, size_func=len):
    ...     return current_idx + size_func(data_item)
    ...
    >>> stream = ['stream', 'of', 'different', 'sized', 'chunks']
    >>> indexer = alt_dynamically_index(size_increments)
    >>> t = list(indexer(stream))
    >>> assert t == [(0, 'stream'), (6, 'of'), (8, 'different'), (17, 'sized'),
    ...              (22, 'chunks')]
    """
    return partial(_dynamic_indexer, idx_updater=idx_updater, start=start)


# ---------------------------------------------------------------------------------------
# Slicing index segment streams


def segment_overlaps(bt_tt_segment, query_bt, query_tt):
    """Returns True if, and only if, bt_tt_segment overlaps query interval.

    A `bt_tt_segment` will need to be of the ``(bt, tt, *data)`` format.
    That is, an iterable of at least two elements (the ``bt`` and ``tt``) followed with
    more elements (the actual segment data).

    This function is made to be curried, as shown in the following example:

    >>> from functools import partial
    >>> overlapping_segments_filt = partial(segment_overlaps, query_bt=4, query_tt=8)
    >>>
    >>> list(filter(overlapping_segments_filt, [
    ...     (1, 3, 'completely before'),
    ...     (2, 4, 'still completely before (upper bounds are strict)'),
    ...     (3, 6, 'partially before, but overlaps bottom'),
    ...     (4, 5, 'totally', 'inside'),  # <- note this tuple has 4 elements
    ...     (5, 8),  # <- note this tuple has only the minimum (2) elements,
    ...     (7, 10, 'partially after, but overlaps top'),
    ...     (8, 11, 'completely after (strict upper bound)'),
    ...     (100, 101, 'completely after (obviously)')
    ... ]))  # doctest: +NORMALIZE_WHITESPACE
    [(3, 6, 'partially before, but overlaps bottom'),
    (4, 5, 'totally', 'inside'),
    (5, 8),
    (7, 10, 'partially after, but overlaps top')]

    """
    bt, tt, *segment = bt_tt_segment
    return (
        query_bt < tt <= query_tt  # the top part of segment intersects
        or query_bt <= bt < query_tt  # the bottom part of the segment intersects
        # If it's both, the interval is entirely inside the query
    )


Stats = Any
_no_value_specified_sentinel = cast(int, object())


def always_true(x):
    return True


class BufferStats(deque):
    """A callable (fifo) buffer. Calls add input to it, but also returns some results
    computed from it's contents.

    What "add" means is configurable (through ``add_new_val`` arg). Default
    is append, but can be extend etc.

    >>> bs = BufferStats(maxlen=4, func=sum)
    >>> list(map(bs, range(7)))
    [0, 1, 3, 6, 10, 14, 18]

    See what happens when you feed the same sequence again:

    >>> list(map(bs, range(7)))
    [15, 12, 9, 6, 10, 14, 18]

    More examples:

    >>> list(map(BufferStats(maxlen=4, func=''.join), 'abcdefgh'))
    ['a', 'ab', 'abc', 'abcd', 'bcde', 'cdef', 'defg', 'efgh']

    >>> from math import prod
    >>> list(map(BufferStats(maxlen=4, func=prod), range(7)))
    [0, 0, 0, 0, 24, 120, 360]

    With a different ``add_new_val`` choice.

    >>> bs = BufferStats(maxlen=4, func=''.join, add_new_val=deque.appendleft)
    >>> list(map(bs, 'abcdefgh'))
    ['a', 'ba', 'cba', 'dcba', 'edcb', 'fedc', 'gfed', 'hgfe']

    With ``add_new_val=deque.extend``, data can be fed in chunks.
    In the following, also see how we use iterize to get a function that
    takes an iterator and returns an iterator

    >>> from creek.util import iterize
    >>> window_stats = iterize(BufferStats(
    ...     maxlen=4, func=''.join, add_new_val=deque.extend)
    ... )
    >>> chks = ['a', 'bc', 'def', 'gh']
    >>> for x in window_stats(chks):
    ...     print(x)
    a
    abc
    cdef
    efgh

    Note: To those who might think that they can optimize this for special
    cases: Yes you can.
    But SHOULD you? Is it worth the increase in complexity and reduction in
    flexibility?
    See https://github.com/thorwhalen/umpyre/blob/master/misc/performance_of_rolling_window_stats.md

    """

    # __name__ = 'BufferStats'

    def __init__(
        self,
        values=(),
        maxlen: int = _no_value_specified_sentinel,
        func: Callable = sum,
        add_new_val: Callable = deque.append,
        # *,
        # func_cond=always_true,
    ):
        """

        :param maxlen: Size of the buffer
        :param func: The function to be computed (on buffer contents) and
        returned when buffer is "called"
        :param add_new_val: The function that adds values on the buffer.
        Signature must be (self, new_val)
            Is usually a deque method (``deque.append`` by default, but could
            be ``deque.extend``, ``deque.appendleft`` etc.).
            Can also be any other function that
            has a valid (self, new_val) signature.
        """
        if maxlen is _no_value_specified_sentinel:
            raise TypeError('You are required to specify maxlen')
        if not isinstance(maxlen, int):
            raise TypeError(f'maxlen must be an integer, was: {maxlen}')

        super().__init__(values, maxlen=maxlen)
        self.func = func
        if isinstance(add_new_val, str):
            # assume add_new_val is a method of deque:
            add_new_val = getattr(self, add_new_val)
        self.add_new_val = add_new_val
        self.__name__ = 'BufferStats'
        # self.func_cond = func_cond

    def __call__(self, new_val) -> Stats:
        self.add_new_val(self, new_val)  # add the new value
        return self.func(self)
        # if self.func_cond(self):
        #     return self.func(self)


def is_not_none(x):
    return x is not None


def return_buffer_on_stats_condition(
    stats: Stats, buffer: Iterable, cond: Callable = is_not_none, else_val=None
):
    """

    >>> return_buffer_on_stats_condition(
    ... stats=3, buffer=[1,2,3,4], cond=lambda x: x%2 == 1
    ... )
    [1, 2, 3, 4]
    >>> return_buffer_on_stats_condition(
    ... stats=3, buffer=[1,2,3,4], cond=lambda x: x%2 == 0, else_val='3 is not even!'
    ... )
    '3 is not even!'
    """

    if cond(stats):
        return buffer
    else:
        return else_val


@dataclass
class Segmenter:
    """

    >>> gen = iter(range(200))
    >>> bs = BufferStats(maxlen=10, func=sum)
    >>> return_if_stats_is_odd = partial(
    ...     return_buffer_on_stats_condition,
    ...     cond=lambda x: x%2 == 1, else_val='The sum is not odd!'
    ... )
    >>> seg = Segmenter(buffer=bs, stats_buffer_callback=return_if_stats_is_odd)

    Since the sum of the values in the buffer [1] is odd, the buffer is returned:

    >>> seg(new_val=1)
    [1]

    Adding ``1 + 2`` is still odd so:

    >>> seg(new_val=2)
    [1, 2]

    Now since ``1 + 2 + 5`` is even, the ``else_val`` of ``return_if_stats_is_odd``
    is returned instead

    >>> seg(new_val=5)
    'The sum is not odd!'
    """

    buffer: BufferStats
    stats_buffer_callback: Callable[
        [Stats, Iterable], Any
    ] = return_buffer_on_stats_condition
    __name__ = 'Segmenter'

    def __call__(self, new_val):
        stats = self.buffer(new_val)
        return self.stats_buffer_callback(stats, list(self.buffer))
```

## util.py

```python
"""Utils for creek"""

from functools import (
    WRAPPER_ASSIGNMENTS,
    partial,
    update_wrapper as _update_wrapper,
    wraps as _wraps,
    singledispatch,
)
from itertools import islice

from typing import (
    Protocol,
    runtime_checkable,
    Tuple,
    Callable,
    Optional,
    Generator,
    Iterable,
    Any,
    Union,
    NewType,
    Iterator,
)


def iterate_skipping_errors(
    g: Iterable,
    error_callback: Optional[Callable[[BaseException], Any]] = None,
    caught_exceptions: Tuple[BaseException] = (Exception,),
) -> Generator:
    """
    Iterate over a generator, skipping errors and calling an error callback if provided.

    :param g: The generator to iterate over
    :param error_callback: A callback to call when an error is encountered.
    :param caught_exceptions: The exceptions to catch and skip.
    :return: A generator that yields the values of the original generator,
    skipping errors.

    >>> list(iterate_skipping_errors(map(lambda x: 1 / x, [1, 0, 2])))
    [1.0, 0.5]
    >>> list(iterate_skipping_errors(map(lambda x: 1 / x, [1, 0, 2]), print))
    division by zero
    [1.0, 0.5]

    See https://github.com/i2mint/creek/issues/6 for more info.

    """
    iterator = iter(g)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except caught_exceptions as e:
            if error_callback:
                error_callback(e)


def iterize(func, name=None):
    """From an In->Out function, makes a Iterator[In]->Itertor[Out] function.

    >>> f = lambda x: x * 10
    >>> f(2)
    20
    >>> iterized_f = iterize(f)
    >>> list(iterized_f(iter([1,2,3])))
    [10, 20, 30]

    """
    iterized_func = partial(map, func)
    if name is not None:
        iterized_func.__name__ = name
    return iterized_func


IteratorItem = Any


@runtime_checkable
class IterableType(Protocol):
    """An iterable type that can actually be used in singledispatch

    >>> assert isinstance([1, 2, 3], IterableType)
    >>> assert not isinstance(2, IterableType)
    """

    def __iter__(self) -> Iterable[IteratorItem]:
        pass


@runtime_checkable
class IteratorType(Protocol):
    """An iterator type that can actually be used in singledispatch

    >>> assert isinstance(iter([1, 2, 3]), IteratorType)
    >>> assert not isinstance([1, 2, 3], IteratorType)
    """

    def __next__(self) -> IteratorItem:
        pass


@runtime_checkable
class CursorFunc(Protocol):
    """An argument-less function returning an iterator's values"""

    def __call__(self) -> IteratorItem:
        """Get the next iterator's item and increment the cursor"""


IterType = NewType('IterType', Union[IteratorType, IterableType, CursorFunc])
IterType.__doc__ = 'A type that can be made into an iterator'

wrapper_assignments = (*WRAPPER_ASSIGNMENTS, '__defaults__', '__kwdefaults__')
update_wrapper = partial(_update_wrapper, assigned=wrapper_assignments)
wraps = partial(_wraps, assigned=wrapper_assignments)


# ---------------------------------------------------------------------------------------
# iteratable, iterator, cursors
# TODO: If bring i2 as dependency, use mk_sentinel here
no_sentinel = type('no_sentinel', (), {})()
no_default = type('no_default', (), {})()


class IteratorExit(BaseException):
    """Raised when an iterator should quit being iterated on, signaling this event
    any process that cares to catch the signal.
    We chose to inherit directly from `BaseException` instead of `Exception`
    for the same reason that `GeneratorExit` does: Because it's not technically
    an error.

    See: https://docs.python.org/3/library/exceptions.html#GeneratorExit
    """


DFLT_INTERRUPT_EXCEPTIONS = (StopIteration, IteratorExit, KeyboardInterrupt)


def iterate_until_exception(iterator, interrupt_exceptions=DFLT_INTERRUPT_EXCEPTIONS):
    while True:
        try:
            next(iterator)
        except interrupt_exceptions:
            print('ending')
            break


def iterable_to_iterator(iterable: Iterable, sentinel=no_sentinel) -> Iterator:
    """Get an iterator from an iterable

    >>> iterable = [1, 2, 3]
    >>> iterator = iterable_to_iterator(iterable)
    >>> assert isinstance(iterator, Iterator)
    >>> assert list(iterator) == iterable

    You can also specify a sentinel, which will result in the iterator stoping just
    before it encounters that sentinel value

    >>> iterable = [1, 2, 3, 4, None, None, 7]
    >>> iterator = iterable_to_iterator(iterable, None)
    >>> assert isinstance(iterator, Iterator)
    >>> list(iterator)
    [1, 2, 3, 4]
    """
    if sentinel is no_sentinel:
        return iter(iterable)
    else:
        return iter(iter(iterable).__next__, sentinel)


def iterator_to_cursor(iterator: Iterator, default=no_default) -> CursorFunc:
    """Get a cursor function for the input iterator.

    >>> iterator = iter([1, 2, 3])
    >>> cursor = iterator_to_cursor(iterator)
    >>> assert callable(cursor)
    >>> assert cursor() == 1
    >>> assert list(cursor_to_iterator(cursor)) == [2, 3]

    Note how we consumed the cursor till the end; by using cursor_to_iterator.
    Indeed, `list(iter(cursor))` wouldn't have worked since a cursor isn't a iterator,
    but a callable to get the items an the iterator would give you.

    You can specify a default. The default has the same role that it has for the
    `next` function: It makes the cursor function return that default when the iterator
    has been "consumed" (i.e. would raise a `StopIteration`).

    >>> iterator = iter([1, 2])
    >>> cursor = iterator_to_cursor(iterator, None)
    >>> assert callable(cursor)
    >>> cursor()
    1
    >>> cursor()
    2

    And then...

    >>> assert cursor() is None
    >>> assert cursor() is None

    forever.

    """
    if default is no_default:
        return partial(next, iterator)
    else:
        return partial(next, iterator, default)


def cursor_to_iterator(cursor: CursorFunc, sentinel=no_sentinel) -> Iterator:
    """Get an iterator from a cursor function.

    A cursor function is a callable that you call (without arguments) to get items of
    data one by one.

    Sometimes, especially in live io contexts, that's the kind interface you're given
    to consume a stream.

    >>> cursor = iter([1, 2, 3]).__next__
    >>> assert not isinstance(cursor, Iterator)
    >>> assert not isinstance(cursor, Iterable)
    >>> assert callable(cursor)

    If you want to consume your stream as an iterator instead, use `cursor_to_iterator`.

    >>> iterator = cursor_to_iterator(cursor)
    >>> assert isinstance(iterator, Iterator)
    >>> list(iterator)
    [1, 2, 3]

    If you want your iterator to stop (without a fuss) as soon as the cursor returns a
    particular element (called a sentinel), say it:

    >>> cursor = iter([1, 2, None, None, 3]).__next__
    >>> iterator = cursor_to_iterator(cursor, sentinel=None)
    >>> list(iterator)
    [1, 2]

    """
    return iter(cursor, sentinel)


def iterable_to_cursor(iterable: Iterable) -> CursorFunc:
    """Get a cursor function from an iterable."""
    iterator = iterable_to_iterator(iterable)
    return iterator_to_cursor(iterator)


@singledispatch
def to_iterator(x: IteratorType, sentinel=no_sentinel):
    """Get an iterator from an iterable or a cursor function

    >>> from typing import Iterator
    >>> it = to_iterator([1, 2, 3])
    >>> assert isinstance(it, Iterator)
    >>> list(it)
    [1, 2, 3]
    >>> list(it)
    []

    >>> cursor = iter([1, 2, 3]).__next__
    >>> assert isinstance(cursor, CursorFunc)
    >>> it = to_iterator(cursor)
    >>> assert isinstance(it, Iterator)
    >>> list(it)
    [1, 2, 3]
    >>> list(it)
    []

    You can use sentinels too

    >>> list(to_iterator([1, 2, None, 4], sentinel=None))
    [1, 2]
    >>> cursor = iter([1, 2, 3, 4, 5]).__next__
    >>> list(to_iterator(cursor, sentinel=4))
    [1, 2, 3]
    """
    if sentinel is no_sentinel:
        return x
    else:
        cursor = x.__next__
        return iter(cursor, sentinel)


@to_iterator.register
def _(x: IterableType, sentinel=no_sentinel):
    return to_iterator.__wrapped__(iter(x), sentinel)
    # TODO: Use of __wrapped__ seems hacky. Better way?
    # TODO: Why does to_iterator(iter(x), sentinel) lead to infinite recursion?


@to_iterator.register
def _(x: CursorFunc, sentinel=no_sentinel):
    return iter(x, sentinel)


# ---------------------------------------------------------------------------------------

no_such_item = type('NoSuchItem', (), {})()


class stream_util:
    def always_true(*args, **kwargs):
        return True

    def do_nothing(*args, **kwargs):
        pass

    def rewind(self, instance):
        instance.seek(0)

    def skip_lines(self, instance, n_lines_to_skip=0):
        instance.seek(0)


class PreIter:
    def skip_items(self, instance, n):
        return islice(instance, n, None)


def cls_wrap(cls, obj):
    if isinstance(obj, type):

        @wraps(obj, updated=())
        class Wrap(cls):
            @wraps(obj.__init__)
            def __init__(self, *args, **kwargs):
                wrapped = obj(*args, **kwargs)
                super().__init__(wrapped)

        # Wrap.__signature__ = signature(obj)

        return Wrap
    else:
        return cls(obj)


# TODO: Make identity_func "identifiable". If we use the following one, we can use == to detect it's use,
# TODO: ... but there may be a way to annotate, register, or type any identity function so it can be detected.


def identity_func(x):
    return x


static_identity_method = staticmethod(identity_func)


from inspect import signature, Signature


# Note: Pipe code is completely independent (with inspect imports signature & Signature)
#  If you only need simple pipelines, use this, or even copy/paste it where needed.
# TODO: Public interface mis-aligned with i2. funcs list here, in i2 it's dict. Align?
#  If we do so, it would be a breaking change since any dependents that expect funcs
#  to be a list of funcs will iterate over a iterable of names instead.
class Pipe:
    """Simple function composition. That is, gives you a callable that implements input -> f_1 -> ... -> f_n -> output.

    >>> def foo(a, b=2):
    ...     return a + b
    >>> f = Pipe(foo, lambda x: print(f"x: {x}"))
    >>> f(3)
    x: 5
    >>> len(f)
    2

    You can name functions, but this would just be for documentation purposes.
    The names are completely ignored.

    >>> g = Pipe(
    ...     add_numbers = lambda x, y: x + y,
    ...     multiply_by_2 = lambda x: x * 2,
    ...     stringify = str
    ... )
    >>> g(2, 3)
    '10'
    >>> len(g)
    3

    Notes:
        - Pipe instances don't have a __name__ etc. So some expectations of normal functions are not met.
        - Pipe instance are pickalable (as long as the functions that compose them are)

    You can specify a single functions:

    >>> Pipe(lambda x: x + 1)(2)
    3

    but

    >>> Pipe()
    Traceback (most recent call last):
      ...
    ValueError: You need to specify at least one function!

    You can specify an instance name and/or doc with the special (reserved) argument
    names ``__name__`` and ``__doc__`` (which therefore can't be used as function names):

    >>> f = Pipe(map, add_it=sum, __name__='map_and_sum', __doc__='Apply func and add')
    >>> f(lambda x: x * 10, [1, 2, 3])
    60
    >>> f.__name__
    'map_and_sum'
    >>> f.__doc__
    'Apply func and add'

    """

    funcs = ()

    def __init__(self, *funcs, **named_funcs):
        named_funcs = self._process_reserved_names(named_funcs)
        funcs = list(funcs) + list(named_funcs.values())
        self.funcs = funcs
        n_funcs = len(funcs)
        if n_funcs == 0:
            raise ValueError('You need to specify at least one function!')

        elif n_funcs == 1:
            other_funcs = ()
            first_func = last_func = funcs[0]
        else:
            first_func, *other_funcs = funcs
            *_, last_func = other_funcs

        self.__signature__ = Pipe._signature_from_first_and_last_func(
            first_func, last_func
        )
        self.first_func, self.other_funcs = first_func, other_funcs

    _reserved_names = ('__name__', '__doc__')

    def _process_reserved_names(self, named_funcs):
        for name in self._reserved_names:
            if (value := named_funcs.pop(name, None)) is not None:
                setattr(self, name, value)
        return named_funcs

    def __call__(self, *args, **kwargs):
        out = self.first_func(*args, **kwargs)
        for func in self.other_funcs:
            out = func(out)
        return out

    def __len__(self):
        return len(self.funcs)

    _dflt_signature = Signature.from_callable(lambda *args, **kwargs: None)

    @staticmethod
    def _signature_from_first_and_last_func(first_func, last_func):
        try:
            input_params = signature(first_func).parameters.values()
        except ValueError:  # function doesn't have a signature, so take default
            input_params = Pipe._dflt_signature.parameters.values()
        try:
            return_annotation = signature(last_func).return_annotation
        except ValueError:  # function doesn't have a signature, so take default
            return_annotation = Pipe._dflt_signature.return_annotation
        return Signature(tuple(input_params), return_annotation=return_annotation)
```