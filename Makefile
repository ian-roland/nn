all:	copy_model train test predict summary libnn.so

CFLAGS=-g # -Ofast

FLOATING_POINT_PROJECT_PATH= ../nnQuantized

copy_model:
	@echo "Searching for quantized model in: $(FLOATING_POINT_PROJECT_PATH)"
	@echo "Current directory: $(PWD)"
	@if [ -f "$(FLOATING_POINT_PROJECT_PATH)/quantized_model.txt" ]; then \
		cp "$(FLOATING_POINT_PROJECT_PATH)/quantized_model.txt" ./model.txt; \
		echo "Copied quantized model successfully"; \
	else \
		echo "Error: Quantized model not found in $(FLOATING_POINT_PROJECT_PATH)"; \
		echo "Possible solutions:"; \
		echo "1. Ensure floating-point project is trained and quantized"; \
		echo "2. Check the FLOATING_POINT_PROJECT_PATH variable"; \
		echo "3. Manually copy the model to this directory as 'model.txt'"; \
		exit 1; \
	fi

libnn.so: nn.o data_prep.o
	$(RM) $@
	$(AR) rcs $@ nn.o data_prep.o
	$(CC) -shared -Wl,-o libnn.so *.o

data_prep.o: data_prep.c data_prep.h
	$(CC) -Wall data_prep.c -c -march=native -flto $(CFLAGS) -fPIC

nn.o: nn.c nn.h
	$(CC) -Wall nn.c -c -march=native -flto $(CFLAGS) -fPIC

train: train.c nn.o data_prep.o
	$(CC) -Wall train.c data_prep.o nn.o -o train -lm -march=native $(CFLAGS)

test: test.c nn.o data_prep.o
	$(CC) -Wall test.c data_prep.o nn.o -o test -lm -march=native $(CFLAGS)

predict: predict.c nn.o
	$(CC) -Wall predict.c nn.o -o predict -lm -march=native $(CFLAGS)

summary: summary.c nn.o
	$(CC) -Wall summary.c nn.o -o summary -lm -march=native $(CFLAGS)

tags:
	ctags -R *

check:
	cppcheck --enable=all --inconclusive .

clean:
	$(RM) data_prep.o nn.o libnn.so train test predict summary model.txt tags nn.png
	$(RM) -r __pycache__
