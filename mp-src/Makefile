
TARGETS := memory_figures.png


all: $(TARGETS)
	rm -f *.log *.tex *.mps *.mpx
	
	mv memory_figures_0.png ../src/sciml/memory_layout.png

%.png: %.mp
	mpost $<

test:
	mpost test

.PHONY: clean all
clean:
	rm -f *.log *.tex *.mps *.mpx
	rm -f *.png