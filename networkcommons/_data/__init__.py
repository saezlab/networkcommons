import functools as ft
from pypath_common import data

load = ft.partial(data.load, module = 'networkcommons')
