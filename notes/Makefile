help:
	@echo "Usage: make compile | make clean"

compile:
	@mkdir -p main
	@pdflatex -jobname=main -output-directory=main $(file)
	# @bibtex report/report.aux
	# @pdflatex -jobname=report -output-directory=report src/main.tex
	@pdflatex -jobname=main -output-directory=main $(file)
	@mv main/main.pdf .
	@open main.pdf

clean:
	@rm -r report

new-project:
	@mkdir $(name)
	@cp -r src images Makefile $(name)
