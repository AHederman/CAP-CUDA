CC	= nvcc
LD	= $(CC)

CFLAGS	= -g -Wall 
CFLAGS	+= -I. 
LDFLAGS	= 

OBJS	= 
LIBS    = -lm #-llmpe -lmpe

SRCS	= $(patsubst %.o,%.c,$(OBJS))

PRGS	= gs_seq gs_cuda 

all: $(PRGS)

%.o: %.c $(HEADERS)
		$(CC) $(CFLAGS) -c $*.c $(INCLUDE) -o $@

$(PRGS): $(OBJS)
	$(PRGS): 
	$(PRGS): % : %.o
		$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS) $(LIBS)

clean:
		-rm -f *.o  *~ $(PRGS)
