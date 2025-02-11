from ns import ns
import inspect

all_classes = [name for name in dir(ns) if inspect.isclass(getattr(ns, name, None))]
print(all_classes)
