PANDOC_FLAGS = --pdf-engine=tectonic \
	-V lang=pt-BR \
	-V geometry:margin=2cm \
	-V fontsize=11pt \
	-V colorlinks=true

FIGURES = $(wildcard figures/*.png)

.PHONY: all clean

all: relatorio.pdf

relatorio.pdf: relatorio.md $(FIGURES)
	pandoc $(PANDOC_FLAGS) $< -o $@

clean:
	rm -f relatorio.pdf
