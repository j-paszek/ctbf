LATEXMK ?= latexmk
PYTHON ?= python
LATEXFLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all main main_a supplement clean-render baseline section review-smoke \
	figure-connect-then-complete figure-connect-then-complete-v2 \
	figure-connect-then-complete-owner-v1 figure-connect-then-complete-owner-v2 \
	figure-nonidentifiability-v1 figure-nonidentifiability-v2 \
	figure-trimmed-true-tree-v1 \
	figure-ktn152-empirical figure-ktn152-empirical-v2 figure-confirmation-v2 \
	figure-confirmation-v3 figure-confirmation-v4 figure-confirmation-v5 figure-confirmation-v6 figure-ktn152-empirical-v3 \
	figure-ktn152-empirical-v4

all: main supplement

main:
	mkdir -p build/main
	$(LATEXMK) $(LATEXFLAGS) -outdir=build/main \
		-jobname=ctbf-main-review main.tex

main_a:
	mkdir -p build/main_a
	$(LATEXMK) $(LATEXFLAGS) -outdir=build/main_a \
		-jobname=ctbf-main-a main_a.tex

supplement:
	mkdir -p build/supplement
	$(LATEXMK) $(LATEXFLAGS) -outdir=build/supplement \
		-jobname=ctbf-supplement-review supplement.tex

clean-render:
	mkdir -p build/clean-main build/clean-supplement
	$(LATEXMK) $(LATEXFLAGS) -outdir=build/clean-main \
		-jobname=ctbf-main-clean \
		-usepretex='\PassOptionsToPackage{clean}{ctbf-review}' main.tex
	$(LATEXMK) $(LATEXFLAGS) -outdir=build/clean-supplement \
		-jobname=ctbf-supplement-clean \
		-usepretex='\PassOptionsToPackage{clean}{ctbf-review}' supplement.tex

baseline:
	@test ! -e review/iteration-0/ctbf-main-iteration-0.pdf && \
		test ! -e review/iteration-0/ctbf-supplement-iteration-0.pdf || \
		(echo "Iteration 0 already exists and is immutable." >&2; exit 2)
	$(MAKE) all clean-render
	mkdir -p review/iteration-0
	cp build/main/ctbf-main-review.pdf \
		review/iteration-0/ctbf-main-iteration-0.pdf
	cp build/supplement/ctbf-supplement-review.pdf \
		review/iteration-0/ctbf-supplement-iteration-0.pdf

section:
	@test -n "$(SECTION)" || \
		(echo "SECTION is required (lowercase hyphenated slug)." >&2; exit 2)
	@test -n "$(ITERATION)" || \
		(echo "ITERATION is required (positive integer)." >&2; exit 2)
	@test -f "review/sections/$(SECTION)/iteration-$(ITERATION)/revision.tex" || \
		(echo "Missing review revision.tex for the requested section iteration." >&2; exit 2)
	mkdir -p build/review/$(SECTION)/iteration-$(ITERATION)
	$(LATEXMK) $(LATEXFLAGS) \
		-outdir=build/review/$(SECTION)/iteration-$(ITERATION) \
		-jobname=ctbf-$(SECTION)-iteration-$(ITERATION)-review \
		-usepretex='\def\CTBFReviewTitle{$(SECTION), iteration $(ITERATION)}\def\CTBFReviewSource{review/sections/$(SECTION)/iteration-$(ITERATION)/revision.tex}' \
		review/section-driver.tex
	$(LATEXMK) $(LATEXFLAGS) \
		-outdir=build/review/$(SECTION)/iteration-$(ITERATION) \
		-jobname=ctbf-$(SECTION)-iteration-$(ITERATION)-clean \
		-usepretex='\PassOptionsToPackage{clean}{ctbf-review}\def\CTBFReviewTitle{$(SECTION), iteration $(ITERATION), clean candidate}\def\CTBFReviewSource{review/sections/$(SECTION)/iteration-$(ITERATION)/revision.tex}' \
		review/section-driver.tex
	cp build/review/$(SECTION)/iteration-$(ITERATION)/ctbf-$(SECTION)-iteration-$(ITERATION)-review.pdf \
		review/sections/$(SECTION)/iteration-$(ITERATION)/
	cp build/review/$(SECTION)/iteration-$(ITERATION)/ctbf-$(SECTION)-iteration-$(ITERATION)-clean.pdf \
		review/sections/$(SECTION)/iteration-$(ITERATION)/

review-smoke:
	mkdir -p build/review-smoke
	$(LATEXMK) $(LATEXFLAGS) -outdir=build/review-smoke \
		-jobname=ctbf-review-smoke \
		-usepretex='\def\CTBFReviewTitle{Review macro smoke test}' \
		review/section-driver.tex
	$(LATEXMK) $(LATEXFLAGS) -outdir=build/review-smoke \
		-jobname=ctbf-clean-smoke \
		-usepretex='\PassOptionsToPackage{clean}{ctbf-review}\def\CTBFReviewTitle{Clean macro smoke test}' \
		review/section-driver.tex

figure-connect-then-complete:
	mkdir -p build/figures/connect-then-complete
	$(LATEXMK) $(LATEXFLAGS) \
		-outdir=build/figures/connect-then-complete \
		-jobname=connect-then-complete \
		figures/connect-then-complete.tex
	cp build/figures/connect-then-complete/connect-then-complete.pdf \
		figures/connect-then-complete.pdf

figure-connect-then-complete-v2:
	mkdir -p build/figures/connect-then-complete-v2
	$(LATEXMK) $(LATEXFLAGS) \
		-outdir=build/figures/connect-then-complete-v2 \
		-jobname=connect-then-complete-v2 \
		figures/connect-then-complete-v2.tex
	cp build/figures/connect-then-complete-v2/connect-then-complete-v2.pdf \
		figures/connect-then-complete-v2.pdf

figure-connect-then-complete-owner-v1:
	mkdir -p build/figures/connect-then-complete-owner-v1
	$(LATEXMK) $(LATEXFLAGS) \
		-outdir=build/figures/connect-then-complete-owner-v1 \
		-jobname=connect-then-complete-owner-v1 \
		figures/connect-then-complete-owner-v1.tex
	cp build/figures/connect-then-complete-owner-v1/connect-then-complete-owner-v1.pdf \
		figures/connect-then-complete-owner-v1.pdf

figure-connect-then-complete-owner-v2:
	mkdir -p build/figures/connect-then-complete-owner-v2
	$(LATEXMK) $(LATEXFLAGS) \
		-outdir=build/figures/connect-then-complete-owner-v2 \
		-jobname=connect-then-complete-owner-v2 \
		figures/connect-then-complete-owner-v2.tex
	cp build/figures/connect-then-complete-owner-v2/connect-then-complete-owner-v2.pdf \
		figures/connect-then-complete-owner-v2.pdf

figure-nonidentifiability-v1:
	mkdir -p build/figures/nonidentifiability-v1
	$(LATEXMK) $(LATEXFLAGS) \
		-outdir=build/figures/nonidentifiability-v1 \
		-jobname=nonidentifiability-v1 figures/nonidentifiability-v1.tex
	cp build/figures/nonidentifiability-v1/nonidentifiability-v1.pdf \
		figures/nonidentifiability-v1.pdf

figure-nonidentifiability-v2:
	mkdir -p build/figures/nonidentifiability-v2
	$(LATEXMK) $(LATEXFLAGS) \
		-outdir=build/figures/nonidentifiability-v2 \
		-jobname=nonidentifiability-v2 figures/nonidentifiability-v2.tex
	cp build/figures/nonidentifiability-v2/nonidentifiability-v2.pdf \
		figures/nonidentifiability-v2.pdf

figure-trimmed-true-tree-v1:
	mkdir -p build/figures/trimmed-true-tree-v1
	$(LATEXMK) $(LATEXFLAGS) \
		-outdir=build/figures/trimmed-true-tree-v1 \
		-jobname=trimmed-true-tree-v1 figures/trimmed-true-tree-v1.tex
	cp build/figures/trimmed-true-tree-v1/trimmed-true-tree-v1.pdf \
		figures/trimmed-true-tree-v1.pdf

figure-ktn152-empirical:
	@test -n "$(KTN152_COPYKIT_ROOT)" || \
		(echo "KTN152_COPYKIT_ROOT is required." >&2; exit 2)
	@test -n "$(KTN152_RECONSTRUCTION)" || \
		(echo "KTN152_RECONSTRUCTION is required." >&2; exit 2)
	python3 figures/build_ktn152_empirical_figure.py \
		--copykit-root "$(KTN152_COPYKIT_ROOT)" \
		--reconstruction "$(KTN152_RECONSTRUCTION)" \
		--output figures/ktn152-empirical-v1.svg
	inkscape --export-filename=figures/ktn152-empirical-v1.pdf \
		figures/ktn152-empirical-v1.svg

# New variants refuse to overwrite existing SVG/data derivatives.
figure-ktn152-empirical-v2:
	@test -n "$(KTN152_COPYKIT_ROOT)" && test -n "$(KTN152_RECONSTRUCTION)" && \
		test -n "$(KTN152_STATE_MAP)" || \
		(echo "Set KTN152_COPYKIT_ROOT, KTN152_RECONSTRUCTION and KTN152_STATE_MAP." >&2; exit 2)
	$(PYTHON) figures/build_ktn152_empirical_figure_v2.py \
		--copykit-root "$(KTN152_COPYKIT_ROOT)" \
		--reconstruction "$(KTN152_RECONSTRUCTION)" \
		--state-map "$(KTN152_STATE_MAP)" \
		--output figures/ktn152-empirical-v2.svg
	inkscape --export-filename=figures/ktn152-empirical-v2.pdf \
		figures/ktn152-empirical-v2.svg

figure-confirmation-v2:
	$(PYTHON) figures/build_confirmation_figure_v2.py \
		--source figures/v5-confirmation-v1 \
		--output figures/v5-confirmation-v2/main-figure.pdf

figure-confirmation-v3:
	@test -n "$(CONFIRMATION_RUN_ROOT)" || \
		(echo "CONFIRMATION_RUN_ROOT is required." >&2; exit 2)
	$(PYTHON) figures/build_confirmation_figure_v3.py \
		--run-root "$(CONFIRMATION_RUN_ROOT)" \
		--source figures/v5-confirmation-v1 \
		--output-root figures/v5-confirmation-v3

figure-confirmation-v4:
	@test -n "$(CONFIRMATION_RUN_ROOT)" || \
		(echo "Set CONFIRMATION_RUN_ROOT to the frozen internal run." >&2; exit 2)
	$(PYTHON) figures/build_confirmation_figure_v4.py \
		--run-root "$(CONFIRMATION_RUN_ROOT)" \
		--source figures/v5-confirmation-v1 --previous figures/v5-confirmation-v3 \
		--output-root figures/v5-confirmation-v4

figure-confirmation-v5:
	@test -n "$(PARTIAL_EXTENSION_ROOT)" || \
		(echo "Set PARTIAL_EXTENSION_ROOT to the completed uniform 2M extension." >&2; exit 2)
	cd .. && $(PYTHON) paper/figures/build_confirmation_figure_v5.py \
		--extension-root "$(PARTIAL_EXTENSION_ROOT)" \
		--output-root paper/figures/v5-confirmation-v5

figure-confirmation-v6:
	@test -n "$(PARTIAL_EXTENSION_ROOT)" || \
		(echo "Set PARTIAL_EXTENSION_ROOT to the completed uniform 2M extension." >&2; exit 2)
	cd .. && $(PYTHON) paper/figures/build_confirmation_figure_v6.py \
		--extension-root "$(PARTIAL_EXTENSION_ROOT)" \
		--output-root paper/figures/v5-confirmation-v6

figure-ktn152-empirical-v3:
	@test -n "$(KTN152_RECONSTRUCTION)" || \
		(echo "Set KTN152_RECONSTRUCTION to the frozen M2 tree." >&2; exit 2)
	$(PYTHON) figures/build_ktn152_empirical_figure_v3.py \
		--previous figures/ktn152-empirical-v2.svg \
		--previous-audit figures/ktn152-empirical-v2.audit.json \
		--reconstruction "$(KTN152_RECONSTRUCTION)" \
		--output figures/ktn152-empirical-v3.svg
	inkscape --export-filename=figures/ktn152-empirical-v3.pdf \
		figures/ktn152-empirical-v3.svg

# Display-label-only derivative; the SVG is the editable source.
figure-ktn152-empirical-v4:
	inkscape --export-filename=figures/ktn152-empirical-v4.pdf \
		figures/ktn152-empirical-v4.svg
