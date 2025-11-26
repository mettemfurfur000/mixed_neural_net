all:
	gcc -o mixed_nn src/main.c src/nn.c src/funcs.c -lm