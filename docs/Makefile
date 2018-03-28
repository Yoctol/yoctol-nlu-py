# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SPHINXPROJ    = yoctol-nlu
SOURCEDIR     = .
BUILDDIR      = _build
CODEDIR       = ../ynlu

# 'make' defaults to 'make html'
all:
	@echo ***Regenerating API stubs and building html files***
	make apidoc
	make html

apidoc:
	@echo ***Regenerating API stubs***
	rm -r $(SOURCEDIR)/reference || true
	$(SPHINXAPIDOC) -f -o $(SOURCEDIR)/reference $(CODEDIR) $(CODEDIR)/benchmark_dataset

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
