import pkgutil
import langchain_core
import langchain_core.language_models.chat_models

functions = [name for name in dir(langchain_core.language_models.chat_models) 
             if callable(getattr(langchain_core.language_models.chat_models, name)) and not name.startswith("__")]
print(functions)

submodules = [name for _, name, _ in pkgutil.iter_modules(langchain_core.language_models.chat_models.__path__)]
print(submodules)

#    print(nam)
#langprompts
langchain_core.language_models
