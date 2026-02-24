import functools
from typing import Dict, Type, Any, Callable

class MatrixRegistry:
    """Central registry for multi-generational matrix solvers."""
    _solvers: Dict[str, Type] = {}
    _methods: Dict[str, Dict[str, Callable]] = {}

    @classmethod
    def register_solver(cls, name: str):
        """Decorator to register a class as a matrix solver."""
        def decorator(solver_cls: Type):
            cls._solvers[name.lower()] = solver_cls
            return solver_cls
        return decorator

    @classmethod
    def register_method(cls, solver_name: str, method_name: str):
        """Decorator to register a method within a solver."""
        def decorator(func: Callable):
            if solver_name.lower() not in cls._methods:
                cls._methods[solver_name.lower()] = {}
            cls._methods[solver_name.lower()][method_name.lower()] = func
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @classmethod
    def get_solver(cls, name: str) -> Type:
        return cls._solvers.get(name.lower())

    @classmethod
    def list_solvers(cls):
        return list(cls._solvers.keys())

# Global Instance
Registry = MatrixRegistry
solver = Registry.register_solver
method = Registry.register_method

