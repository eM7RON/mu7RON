import warnings


def ignore_warnings(*_warnings):
    """
    unittest likes to set warnings settings to default and 
    trigger some warnings. Use this as a decorator on your
    test methods to deactivate them.
    """

    def decorator(test_func):
        def wrapper(self, *args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", *_warnings)
                test_func(self, *args, **kwargs)

        return wrapper

    return decorator
