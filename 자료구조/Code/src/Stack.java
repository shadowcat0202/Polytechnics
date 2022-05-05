import java.util.Arrays;
import java.util.EmptyStackException;

public class Stack<T> {
    private int top;
    private T[] stack;
    private static final int DEFAULT_SIZE = 10;
    Stack(){
        this(DEFAULT_SIZE);
    }
    Stack(int size){
        this.top = -1;   //중요!
        this.stack = (T[]) new Object[size];
    }


    private void resize(){
        if(this.top < this.stack.length / 2){
            int new_size = this.stack.length * 2;
            Arrays.copyOf(this.stack, new_size);
        }
        if(this.top < this.stack.length / 2){
            int new_size = this.stack.length / 2;
            Arrays.copyOf(this.stack, new_size);
        }
    }
    public boolean isEmpty(){
        return this.top == -1;
    }

    public boolean isFull(){
        return this.stack.length - 1 == this.top;
    }

    void push(T value){
        if(isFull())  resize();
        this.stack[++this.top] = value;
    }

    T pop() {
        if (isEmpty()) throw new EmptyStackException();
        T output = this.stack[this.top];
        this.stack[this.top--] = null;

        if(this.top < this.stack.length / 2)  resize();
        return output;
    }

    T peek(){
        if(isEmpty())   return null;
        return this.stack[this.top];
    }

    void delete(){
        if(isEmpty())   throw new EmptyStackException();
        this.stack[this.top--] = null;
        if(this.top < this.stack.length / 2)  resize();
    }





}
