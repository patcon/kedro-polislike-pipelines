class EstimatorRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        """Decorator to register a factory callable that returns an estimator"""

        def decorator(factory_callable):
            cls._registry[name] = factory_callable
            return factory_callable

        return decorator

    @classmethod
    def get(cls, name, **kwargs):
        if name not in cls._registry:
            raise ValueError(f"Estimator {name} not registered")
        # Call the factory function now, passing kwargs
        return cls._registry[name](**kwargs)
