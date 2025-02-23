all: train test test_quantized predict quantize summary libnn.so

CFLAGS = -Wall -Ofast -march=native -flto -fPIC
LDFLAGS = -lm 

libnn.so: nn.o data_prep.o
	$(RM) $@
	$(AR) rcs $@ $^
	$(CC) -shared -Wl,-soname,libnn.so -o $@ $^ $(LDFLAGS)

data_prep.o: data_prep.c data_prep.h
	$(CC) $(CFLAGS) -c $<

nn.o: nn.c nn.h
	$(CC) $(CFLAGS) -c $<

train: train.c nn.o data_prep.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)  

test: test.c nn.o data_prep.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test_quantized: test_quantized.c nn.o data_prep.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

predict: predict.c nn.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

summary: summary.c nn.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

quantize: quantize.c nn.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tags:
	ctags -R *

check:
	cppcheck --enable=all --inconclusive .

clean:
	$(RM) data_prep.o nn.o libnn.so train test test_quantized predict quantize summary model*.txt tags nn.png
	$(RM) -r __pycache__