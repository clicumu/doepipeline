# *doepipeline*

Optimize your data processing pipelines with *doepipeline*. The optimization strategy implemented in *doepipeline* is based on methods from statistical [Design of Experiments (DoE)](https://en.wikipedia.org/wiki/Design_of_experiments). Use it to optimize quantitative and/or qualitative factors of simple (single tool) or complex (multiple tool) pipelines.

![doepipeline overview](https://github.com/clicumu/doepipeline/blob/master/supporting/doepipeline_overview.png)

# Features
* Community developed: Users are welcome to contribute to add additional functionality.
* Installation: Easy installation through [conda](http://conda-forge.org/) or [PyPI](https://pypi.org/).
* Generic: The optimization is useful for all kinds of CLI applications.

# Quick start links
Take a look at the [wiki documentation](https://github.com/clicumu/doepipeline/wiki) to getting started using doepipeline. Briefly, the following steps are needed to start using doepipeline.

1. [Install doepipeline](https://github.com/clicumu/doepipeline/wiki/Installation)
2. [Create YAML configuration file](https://github.com/clicumu/doepipeline/wiki/Configuration-file)
3. [Run optimization](https://github.com/clicumu/doepipeline/wiki/Running)

Four example cases (including data and configuration files) are provided to as help getting started: 
1) [de-novo genome assembly](https://github.com/clicumu/doepipeline/wiki/Case-1)
2) [scaffolding of a fragmented genome assembly](https://github.com/clicumu/doepipeline/wiki/Case-2)
3) [k-mer taxonomic classification of ONT MinION reads](https://github.com/clicumu/doepipeline/wiki/Case-3) 
4) [genetic variant calling](https://github.com/clicumu/doepipeline/wiki/Case-4)

# Cite
__doepipeline: a systematic approach for optimizing multi-level and multi-step data processing workflows__ Svensson D, Sjögren R, Sundell D, Sjödin A, Trygg J BioRxiv doi: https://doi.org/10.1101/504050

# About this software
doepipeline is implemented as a Python package. It is open source software made available under the [MIT license](LICENSE).

If you experience any difficulties with this software, or you have suggestions, or want to contribute directly, you have the following options:

- submit a bug report or feature request to the 
  [issue tracker](https://github.com/clicumu/doepipeline/issues)
- contribute directly to the source code through the 
  [github](https://github.com/clicumu/doepipeline) repository. 'Pull requests' are
  especially welcome.
