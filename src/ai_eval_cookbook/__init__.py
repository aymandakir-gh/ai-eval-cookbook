"""ai-eval-cookbook: provider-agnostic, runnable offline recipes for evaluating LLM systems.

Each recipe is a small, importable, dependency-free module that operates on supplied
model outputs/references or an injected callable. Where a model, judge, embedder, runner,
or other LLM component is conceptually required, a deterministic offline default is used so
that everything runs without network access or API keys.
"""

__version__ = "0.1.0"
